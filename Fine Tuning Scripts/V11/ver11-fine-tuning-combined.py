#memory optimization
import os
from dotenv import load_dotenv
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Step 0: Login to Hugging Face
import huggingface_hub
load_dotenv()
api_key = os.getenv("HF_TOKEN")
huggingface_hub.login(token=api_key) 

# sanity-check inference
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map={"": 0}, #Specifically loads on the first GPU here
    trust_remote_code=True,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
)

# Sample prompt to verify the base model output format
sample_instruction = (
    "You are a medical NLP assistant. Extract ONLY explicitly stated adverse drug reactions,do not infer symptoms not directly mentioned. "
    "Output in the exact format: Drugs: <semicolon-separated list> | ADRs: <semicolon-separated list>"
)
sample_input = (
    "I feel a bit drowsy & have a little blurred vision, so far no gastric problems.\n"
    "I've been on Arthrotec 50 for over 10 years on and off, only taking it when I needed it."
)

messages = [
    {
        "role": "system",
        "content": f"{sample_instruction}\n",
    },

    {
        "role": "user",
        "content": f"\ntext: {sample_input}\n\nlabel: ",
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#Version 11 fix - generation needs to stop at end_of_text and eot_id
sample_terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

outputs = pipe(
    prompt, 
    max_new_tokens=180, 
    do_sample=False,
    eos_token_id=sample_terminators,
    pad_token_id=tokenizer.eos_token_id,
    )

print("=== Base-model sample output ===")
print(outputs[0]["generated_text"])

# Imports for fine-tuning
# Required packages:
# pip install -U transformers
# pip install -U accelerate
# pip install -U peft
# pip install -U trl

import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import json
import re
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          pipeline)
import accelerate

# Requires trl>=0.11 for SFTConfig + DataCollatorForCompletionOnlyLM.
# Recommended pins: trl==0.13.0, transformers==4.49.0,
# peft==0.14.0, accelerate==1.3.0

# Parsing helpers.
def _split_tokens(text):
    """Split a segment on separators and normalize each token."""
    tokens = set()
    for part in re.split(r"[;,\n|]+", text):
        token = part.strip().strip(".,:;\"'()!?-*•").strip().lower()
        if token:
            tokens.add(token)
    return tokens


def extract_adrs(text):
    #Made sure to match ADRs or ADR to account for any mistmatch
    if not text:
        return set()
    text = text.strip()
    if "|" in text:
        adr_part = text.split("|", 1)[1]
    else:
        match = re.search(r"adrs?\s*:", text, re.IGNORECASE)
        adr_part = text[match.end():] if match else None
    if adr_part is None:
        return set()
    match = re.search(r"adrs?\s*:", adr_part, re.IGNORECASE)
    if match:
        adr_part = adr_part[match.end():]
    return _split_tokens(adr_part)


def extract_drug(text):
    #Made sure to match Drugs or Drug to account for any data mismatch
    if not text:
        return set()
    text = text.strip()
    match = re.search(r"drugs?\s*:", text, re.IGNORECASE)
    if not match:
        return set()
    drug_part = text[match.end():]
    drug_part = drug_part.split("|", 1)[0]
    match = re.search(r"adrs?\s*:", drug_part, re.IGNORECASE)
    if match:
        drug_part = drug_part[:match.start()]
    return _split_tokens(drug_part)


# Load and process the JSON dataset
JSON_PATH = "combinedV5-cadec-psy-ade.json"  # path to dataset

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# raw_data is a list of dicts with keys: "instruction", "input", "output"
df = pd.DataFrame(raw_data)
print(f"Total records: {len(df)}")
print(df.head(2))

#Added shuffle to mix the three datasets
df = df.sample(frac=1, random_state=47).reset_index(drop=True)

# Train / eval / test split (80 / 10 / 10)
train_size = 0.8
eval_size  = 0.1

train_end = int(train_size * len(df))
eval_end  = train_end + int(eval_size * len(df))

X_train = df[:train_end].copy()
X_eval  = df[train_end:eval_end].copy()
X_test  = df[eval_end:].copy()

print(f"Train: {len(X_train)} | Eval: {len(X_eval)} | Test: {len(X_test)}")

#Check labels all types of labels for each split
def check_labels(df, name):
    """Abort early if any label in this split fails to parse for either entity type."""
    bad_adr_mask  = df["output"].apply(lambda x: not extract_adrs(x))
    bad_drug_mask = df["output"].apply(lambda x: not extract_drug(x))
    n_bad_adr  = bad_adr_mask.sum()
    n_bad_drug = bad_drug_mask.sum()

    if n_bad_adr or n_bad_drug:
        bad_indices = sorted(set(df.index[bad_adr_mask]) | set(df.index[bad_drug_mask]))
        raise SystemExit(
            f"{name} label check failed: {n_bad_adr} ADR-empty, "
            f"{n_bad_drug} Drug-empty out of {len(df)} rows. "
            f"Bad row indices (first 20): {bad_indices[:20]}"
        )
    print(f"{name} label check passed: all {len(df)} labels parse for drug and ADR.")

check_labels(X_train, "Train")
check_labels(X_eval,  "Eval")
check_labels(X_test,  "Test")

#Check for memory
print(f"Available GPUs: {torch.cuda.device_count()}")

max_memory = {0: "40GiB", 1: "40GiB", 2: "40GiB"} 

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.float16,

)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = tokenizer.eos_token_id


# move these function definitions to AFTER tokenizer is loaded:
def generate_prompt(row):
    messages = [
        {
            "role": "system",
            "content": (
                f"{row['instruction']}\n"
            )
        },
        {
            "role": "user",
            "content": f"\ntext: {row['input']}"
        },
        {
            "role": "assistant",
            "content": row["output"]
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

def generate_test_prompt(row):
    messages = [
        {
            "role": "system",
            "content": (
                "You are assisting a researcher with natural language processing in pharmacovigilance. Extract all adverse drug reactions (ADRS)"
                "explicitaly mentioned in the patient testimony."
                "Output your findings strictly in the format below: "
                "Drugs: <semicolon-separated list> | ADRs: <semicolon-separated list>"
            )
        },
        {
            "role": "user",
            "content": f"\ntext: {row['input']}"
        }
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

X_train.loc[:, "text"] = X_train.apply(generate_prompt, axis=1)
X_eval.loc[:,  "text"] = X_eval.apply(generate_prompt, axis=1)
y_true = X_test["output"].tolist()
X_test_prompts = pd.DataFrame(
    X_test.apply(generate_test_prompt, axis=1), columns=["text"])

# Convert to Hugging Face Dataset
train_data = Dataset.from_pandas(X_train[["text"]])
eval_data  = Dataset.from_pandas(X_eval[["text"]])


#Check part of the shuffled data for the correct template, abort if a majority are incorrect.
response_template = "<|start_header_id|>assistant<|end_header_id|>"
rt_ids = tokenizer.encode(response_template, add_special_tokens=False)
if not rt_ids:
    raise SystemExit("Response template tokenized to an empty sequence.")

n_checks = min(25, len(X_train))
found = 0
for _, row in X_train.head(n_checks).iterrows():
    ids = tokenizer.encode(row["text"], add_special_tokens=False)
    if any(ids[i:i + len(rt_ids)] == rt_ids for i in range(len(ids) - len(rt_ids) + 1)):
        found += 1

print(f"Response template check: found in {found}/{n_checks} checked training samples.")
if found <= n_checks * 0.5:
    raise SystemExit(
        "Response template check failed: only found in {found}/{n_checks} sampled data points."
    )


# Inference helper
def predict(test_df, model, tokenizer, y_true=None, tag="baseline"):
    malformed_file = open(f"error_analysis_adr_{tag}.txt", "w", encoding="utf-8")
    raw_file = open(f"raw_predictions_{tag}.txt", "w", encoding="utf-8")
    y_pred = []

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        return_full_text=False,
    )

    #Version 11 fix - generation needs to stop at end_of_text and eot_id
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    for i in tqdm(range(len(test_df))):
        prompt = test_df.iloc[i]["text"]
        result = pipe(
                prompt,
                truncation=True,
                max_new_tokens=180,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id
                )

        # Extract only the generated part after "label: "
        generated = result[0]["generated_text"].strip()
        y_pred.append(generated)

        gold = y_true[i] if y_true is not None else ""
        raw_file.write(f"--- Sample {i} ---\n")
        raw_file.write(f"Prompt (last 200 chars): ...{prompt[-200:]}\n")
        raw_file.write(f"Golden label: {gold}\n")
        raw_file.write(f"Raw output: {generated}\n\n")

        # Log only predictions that are genuinely malformed (empty, or missing
        # a parseable drug section, or missing a parseable ADR section).
        malformed = (
            not generated
            or not extract_drug(generated)
            or not extract_adrs(generated)
        )
        if malformed:
            malformed_file.write(f"--- Sample {i} ---\n")
            malformed_file.write(f"Prompt (last 200 chars): ...{prompt[-200:]}\n")
            malformed_file.write(f"Raw output: {generated}\n\n")

    malformed_file.close()
    raw_file.close()
    return y_pred


# Evaluation helper — uses the shared parsing module.
def evaluate_adr(y_true, y_pred):
    total_tp = total_fp = total_fn = 0
    drug_tp = drug_fp = drug_fn = 0
    drug_exact = 0
    empty_pred = 0

    for true_str, pred_str in zip(y_true, y_pred):
        # ADR token-level evaluation
        true_adrs = extract_adrs(true_str)
        pred_adrs = extract_adrs(pred_str)
        total_tp += len(true_adrs & pred_adrs)
        total_fp += len(pred_adrs - true_adrs)
        total_fn += len(true_adrs - pred_adrs)
        if not pred_adrs:
            empty_pred += 1

        # Drug set-level evaluation (multi-drug aware)
        true_drugs = extract_drug(true_str)
        pred_drugs = extract_drug(pred_str)
        drug_tp += len(true_drugs & pred_drugs)
        drug_fp += len(pred_drugs - true_drugs)
        drug_fn += len(true_drugs - pred_drugs)
        if true_drugs == pred_drugs:
            drug_exact += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    drug_precision = drug_tp / (drug_tp + drug_fp) if (drug_tp + drug_fp) > 0 else 0.0
    drug_recall    = drug_tp / (drug_tp + drug_fn) if (drug_tp + drug_fn) > 0 else 0.0
    drug_f1        = (
        2 * drug_precision * drug_recall / (drug_precision + drug_recall)
        if (drug_precision + drug_recall) > 0
        else 0.0
    )
    drug_acc = drug_exact / len(y_true) if y_true else 0.0

    print(f"ADR Precision  : {precision:.4f}")
    print(f"ADR Recall     : {recall:.4f}")
    print(f"ADR F1         : {f1:.4f}")
    print(f"Drug Precision : {drug_precision:.4f}")
    print(f"Drug Recall    : {drug_recall:.4f}")
    print(f"Drug F1        : {drug_f1:.4f}")
    print(f"Drug Exact Acc : {drug_acc:.4f}  ({drug_exact}/{len(y_true)})")
    print(f"Empty/unparseable predictions: {empty_pred}/{len(y_pred)}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "drug_precision": drug_precision,
        "drug_recall": drug_recall,
        "drug_f1": drug_f1,
        "drug_acc": drug_acc,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "empty_pred": empty_pred,
    }


# Load a fresh base model + adapter for inference, fully decoupled from the
# training-loop model object so no leftover state can leak into evaluation.
def load_inference_model(base_model, tokenizer, adapter_path):
    infer_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    infer_model.config.use_cache = True
    infer_model = PeftModel.from_pretrained(infer_model, adapter_path)
    return infer_model


# Baseline evaluation (before fine-tuning)
print("\n=== Baseline evaluation (no fine-tuning) ===")
y_pred_baseline = predict(X_test_prompts, model, tokenizer, y_true=y_true, tag="baseline_v11")
evaluate_adr(y_true, y_pred_baseline)

del pipe
torch.cuda.empty_cache()

#LoRA target-module discovery
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

modules = find_all_linear_names(model)
print("LoRA target modules:", modules)

# Hyperparameter sweep
output_dir = "llama-fine-tuned-combined-V11"
performance = []

# Smoke check: 1 training step on a 32-sample slice with the sweep config,
# so any GPU/kernel fault surfaces in seconds instead of after the sweep starts.
smoke_peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

smoke_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    optim="adamw_torch",
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.2,
    max_steps=1,
    overwrite_output_dir=True,
    warmup_steps=0,
    lr_scheduler_type="cosine",
    eval_strategy="no",
    max_seq_length=512,
    packing=False,
    dataset_text_field="text",
)

smoke_data = Dataset.from_pandas(X_train[["text"]].head(32))

smoke_trainer = SFTTrainer(
    model=model,
    args=smoke_config,
    tokenizer=tokenizer,
    train_dataset=smoke_data,
    data_collator=DataCollatorForCompletionOnlyLM(
        response_template="<|start_header_id|>assistant<|end_header_id|>",
        tokenizer=tokenizer,
    ),
    peft_config=smoke_peft_config,
)

smoke_trainer.train()
print("Smoke check passed: 1 training step completed without error.")

del smoke_trainer
torch.cuda.empty_cache()
gc.collect()

for batch in [1, 8]:
    for epoch in [1, 3]:
        print(f"\n{'='*60}")
        print(f"Training: batch_size={batch}, epochs={epoch}")
        print(f"{'='*60}")

        #Reload model between each test run so LoRA configs are not stacked between runs
        if 'trainer' in dir():
            del trainer
        if 'model' in dir():
            del model
        if 'pipe' in dir():
            del pipe
        torch.cuda.empty_cache()
        gc.collect()

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
        )

        training_arguments = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epoch,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=12,
            gradient_checkpointing=False,
            optim="adamw_torch",
            logging_steps=1,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.2,
            max_steps=-1,
            overwrite_output_dir=True,          # ensures safe reruns
            warmup_steps=80,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=0.1,
            max_seq_length=512,
            packing=False,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=DataCollatorForCompletionOnlyLM(
                response_template="<|start_header_id|>assistant<|end_header_id|>",
                tokenizer=tokenizer,
            ),
            peft_config=peft_config,
        )

        trainer.train()

        model.config.use_cache = True

        # Save adapter weights and tokenizer per run
        adapter_path = os.path.join(output_dir, f"b{batch}_e{epoch}")
        trainer.save_model(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        # Evaluate on the test set using a fresh base + adapter
        infer_model = load_inference_model(base_model, tokenizer, adapter_path)
        y_pred = predict(X_test_prompts, infer_model, tokenizer, y_true=y_true,
                         tag=f"v12_b{batch}_e{epoch}")
        metrics = evaluate_adr(y_true, y_pred)

        del infer_model
        gc.collect()
        torch.cuda.empty_cache()

        performance.append({
            "batch_size": batch,
            "epochs":     epoch,
            "precision":  metrics["precision"],
            "recall":     metrics["recall"],
            "f1":         metrics["f1"],
            "drug_precision": metrics["drug_precision"],
            "drug_recall":    metrics["drug_recall"],
            "drug_f1":        metrics["drug_f1"],
            "drug_acc":       metrics["drug_acc"],
            "empty_pred":     metrics["empty_pred"],
        })

# Summary table
perf_df = pd.DataFrame(performance)
print("\n=== Hyperparameter sweep results ===")
print(perf_df.sort_values("f1", ascending=False).to_string(index=False))
perf_df.to_csv("combined11_adr_sweep_results.csv", index=False)
print("\nResults saved to combined11_adr_sweep_results.csv")
