
#Version 5 Changes:
#Split every drug and adverse reaction with semicolons
#Ensure that splits occur for every entity and not every word

import re
import json
import random
import argparse
import requests
import pandas as pd
from collections import defaultdict

INSTRUCTION = (
    "You are a medical NLP assistant. Extract ONLY explicitly stated adverse drug reactions,"
    "do not infer symptoms not directly mentioned. Output in the exact format: "
    "Drugs: <semicolon-separated list> | ADRs: <semicolon-separated list>"
)

ADE_URL = (
    "https://raw.githubusercontent.com/trunghlt/AdverseDrugReaction"
    "/master/ADE-Corpus-V2/DRUG-AE.rel"
)


# Shared formatter to extract drug entities at slashes when applicable
def split_drugs(drug):
    if isinstance(drug, str):
        parts = drug.split("/")
    else:
        parts = drug  # already an iterable of drug names
    return [d.strip() for d in parts if d and d.strip()]

#Shared formatter to extract ADR entities at commas when applicable
def split_adrs(adr):
    if isinstance(adr, str):
        parts = adr.split(",")
    else:
        parts = adr  # already an iterable of ADR phrases
    return [a.strip() for a in parts if a and a.strip()]


#Shared formatter to split both entities with semicolons
def make_example(text, drug, adrs):
    drug_str = "; ".join(split_drugs(drug)) or "Unknown"
    adr_str  = "; ".join(a.strip() for a in adrs if a.strip()) or "None identified"
    return {
        "instruction": INSTRUCTION,
        "input":       text.strip(),
        "output":      f"Drugs: {drug_str} | ADRs: {adr_str}",
    }

#Extra check to partition entities correctly
def normalize_output(output):
    if "|" not in output:
        return output
    drug_part, sep, adr_part = output.partition("|")

    drug_part = re.sub(r"^\s*Drugs?:\s*", "", drug_part)
    drug_str = "; ".join(split_drugs(drug_part)) or "Unknown"

    adr_part = re.sub(r"^\s*ADRs?:\s*", "", adr_part.strip())
    adr_str = "; ".join(split_adrs(adr_part)) or "None identified"

    return f"Drugs: {drug_str} | ADRs: {adr_str}"


# Source 1: CADEC 
def load_cadec(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for ex in data:
        ex["instruction"] = INSTRUCTION
        ex["output"] = normalize_output(ex["output"])
    print(f"CADEC:  {len(data):>5} examples")
    return data


# Source 2: ADE Corpus
def load_ade():
#Group by sentence so multiple ADRs become one example.
    print("ADE:    downloading...")
    r = requests.get(ADE_URL, timeout=30)
    r.raise_for_status()

    groups = defaultdict(lambda: {"drug": None, "adrs": []})
    for line in r.text.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 8:
            continue
        _, sentence, adr, _, _, drug, _, _ = parts
        key = (sentence.strip(), drug.strip())
        groups[key]["drug"] = drug.strip()
        groups[key]["adrs"].append(adr.strip())

    examples = [make_example(sent, v["drug"], v["adrs"]) for (sent, _), v in groups.items()]
    print(f"ADE:    {len(examples):>5} examples")
    return examples


# Source 3: PSYTAR
def load_psytar(path):
    #Fetch needed columns
    df = pd.read_excel(path, sheet_name="ADR_Identified", engine="openpyxl")

    adr_cols = [c for c in df.columns if c.startswith("ADR")]

    examples = []
    for _, row in df.iterrows():
        text = str(row["sentences"]).strip()
        #Remove extra dots in drugname
        drug = str(row["drug_id"]).split(".")[0]   # "lexapro.1" -> "lexapro"
        adrs = [str(row[c]).strip() for c in adr_cols
                if pd.notna(row[c]) and str(row[c]).strip().lower() != "nan"]
        if text and adrs:
            examples.append(make_example(text, drug, adrs))

    print(f"PSYTAR: {len(examples):>5} examples")
    return examples


# Combine & save
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cadec",  required=True, help="Path to your CADEC .json file")
    parser.add_argument("--psytar", required=True, help="Path to PsyTAR_dataset.xlsx")
    parser.add_argument("--out",    default="combinedV4_adr_dataset.json", help="Output file path")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    all_examples = []
    all_examples += load_cadec(args.cadec)
    all_examples += load_ade()
    all_examples += load_psytar(args.psytar)

    # Deduplicate on input text
    seen, unique = set(), []
    for ex in all_examples:
        key = ex["input"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    removed = len(all_examples) - len(unique)
    random.seed(args.seed)
    random.shuffle(unique)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    print(f"\nDone.")
    print(f"  Total combined : {len(all_examples)}")
    print(f"  After dedup    : {len(unique)} (removed {removed})")
    print(f"  Saved to       : {args.out}")
    print(f"\nSample:")
    print(json.dumps(unique[0], indent=2))


if __name__ == "__main__":
    main()