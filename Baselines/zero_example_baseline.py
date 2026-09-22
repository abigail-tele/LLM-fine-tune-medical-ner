"""
Zero-shot baseline: raw Llama 3.1 base model, no fine-tuning, no in-context
examples. Uses the SAME system prompt, test split, parsing, and eval code as
ver13-fine-tuning-combined.py, so results are directly comparable to your
fine-tuned model's numbers and to the few-shot baseline.

"""

import os
from dotenv import load_dotenv
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import huggingface_hub
load_dotenv()
huggingface_hub.login(token=os.getenv("HF_TOKEN"))

import json
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
JSON_PATH = "combinedV5-cadec-psy-ade.json"  # same dataset as fine-tuning script
TAG = "zeroshot"

# ---------------------------------------------------------------------------
# Parsing helpers — copied verbatim from ver13-fine-tuning-combined.py so
# scoring logic is identical across all three conditions (zero-shot,
# few-shot, fine-tuned).
# ---------------------------------------------------------------------------

def _split_tokens(text):
    tokens = set()
    for part in re.split(r"[;,\n|]+", text):
        token = part.strip().strip(".,:;\"'()!?-*•").strip().lower()
        if token:
            tokens.add(token)
    return tokens


def extract_adrs(text):
    if not text:
        return False, set()
    text = text.strip()
    if "|" in text:
        adr_part = text.split("|", 1)[1]
    else:
        match = re.search(r"adrs?\s*:", text, re.IGNORECASE)
        adr_part = text[match.end():] if match else None
    if adr_part is None:
        return False, set()
    match = re.search(r"adrs?\s*:", adr_part, re.IGNORECASE)
    if match:
        adr_part = adr_part[match.end():]
    return True, _split_tokens(adr_part)


def extract_drug(text):
    if not text:
        return False, set()
    text = text.strip()
    match = re.search(r"drugs?\s*:", text, re.IGNORECASE)
    if not match:
        return False, set()
    drug_part = text[match.end():]
    drug_part = drug_part.split("|", 1)[0]
    match = re.search(r"adrs?\s*:", drug_part, re.IGNORECASE)
    if match:
        drug_part = drug_part[:match.start()]
    return True, _split_tokens(drug_part)


def evaluate_adr(y_true, y_pred):
    total_tp = total_fp = total_fn = 0
    drug_tp = drug_fp = drug_fn = 0
    drug_exact = 0
    empty_pred = 0

    for true_str, pred_str in zip(y_true, y_pred):
        _, true_adrs = extract_adrs(true_str)
        _, pred_adrs = extract_adrs(pred_str)
        total_tp += len(true_adrs & pred_adrs)
        total_fp += len(pred_adrs - true_adrs)
        total_fn += len(true_adrs - pred_adrs)
        if not pred_adrs:
            empty_pred += 1

        _, true_drugs = extract_drug(true_str)
        _, pred_drugs = extract_drug(pred_str)
        drug_tp += len(true_drugs & pred_drugs)
        drug_fp += len(pred_drugs - true_drugs)
        drug_fn += len(true_drugs - pred_drugs)
        if true_drugs == pred_drugs:
            drug_exact += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    drug_precision = drug_tp / (drug_tp + drug_fp) if (drug_tp + drug_fp) > 0 else 0.0
    drug_recall = drug_tp / (drug_tp + drug_fn) if (drug_tp + drug_fn) > 0 else 0.0
    drug_f1 = (
        2 * drug_precision * drug_recall / (drug_precision + drug_recall)
        if (drug_precision + drug_recall) > 0 else 0.0
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
        "precision": precision, "recall": recall, "f1": f1,
        "drug_precision": drug_precision, "drug_recall": drug_recall,
        "drug_f1": drug_f1, "drug_acc": drug_acc, "empty_pred": empty_pred,
    }


# ---------------------------------------------------------------------------
# Reproduce the EXACT same train/eval/test split as the fine-tuning script
# (same file, same shuffle seed) so the test set here matches the one your
# fine-tuned model was scored on.
# ---------------------------------------------------------------------------

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)
df = df.sample(frac=1, random_state=47).reset_index(drop=True)

train_size = 0.8
eval_size = 0.1
train_end = int(train_size * len(df))
eval_end = train_end + int(eval_size * len(df))

X_test = df[eval_end:].copy()
print(f"Test set size: {len(X_test)}")

# ---------------------------------------------------------------------------
# Load raw base model (no LoRA adapter)
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.config.use_cache = True

# Same fixed evaluation system prompt used by generate_test_prompt() in the
# fine-tuning script — keep this identical across zero-shot/few-shot/fine-tuned.
EVAL_SYSTEM_PROMPT = (
    "You are assisting a researcher with natural language processing in pharmacovigilance. "
    "Extract all adverse drug reactions (ADRS) explicitaly mentioned in the patient testimony."
    "Output your findings strictly in the format below: "
    "Drugs: <semicolon-separated list> | ADRs: <semicolon-separated list>"
)


def generate_test_prompt(row):
    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": f"\ntext: {row['input']}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


y_true = X_test["output"].tolist()
X_test_prompts = pd.DataFrame(
    X_test.apply(generate_test_prompt, axis=1), columns=["text"]
)

# ---------------------------------------------------------------------------
# Inference — same predict() logic as the fine-tuning script
# ---------------------------------------------------------------------------

def predict(test_df, model, tokenizer, y_true=None, tag="zeroshot"):
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
            pad_token_id=tokenizer.eos_token_id,
        )

        generated = result[0]["generated_text"].strip()
        y_pred.append(generated)

        gold = y_true[i] if y_true is not None else ""
        raw_file.write(f"--- Sample {i} ---\n")
        raw_file.write(f"Prompt (last 200 chars): ...{prompt[-200:]}\n")
        raw_file.write(f"Golden label: {gold}\n")
        raw_file.write(f"Raw output: {generated}\n\n")

        drug_found, _ = extract_drug(generated)
        adr_found, _ = extract_adrs(generated)
        if not generated or not drug_found or not adr_found:
            malformed_file.write(f"--- Sample {i} ---\n")
            malformed_file.write(f"Prompt (last 200 chars): ...{prompt[-200:]}\n")
            malformed_file.write(f"Raw output: {generated}\n\n")

    malformed_file.close()
    raw_file.close()
    return y_pred


print("\n=== Zero-shot baseline evaluation ===")
y_pred = predict(X_test_prompts, model, tokenizer, y_true=y_true, tag=TAG)
metrics = evaluate_adr(y_true, y_pred)

pd.DataFrame([metrics]).to_csv(f"{TAG}_results.csv", index=False)
print(f"\nResults saved to {TAG}_results.csv")