
#Recompute strict AND relaxed metrics for ADR/drug extraction from saved files
#Based on baseline paper to compare to


import re
import sys
import glob
import pandas as pd
from pathlib import Path


#Same parsing logic as the training script copied

def _split_tokens(text):
    """Split a segment on separators and normalize each token."""
    tokens = set()
    for part in re.split(r"[;,\n|]+", text):
        token = part.strip().strip(".,:;\"'()!?-*\u2022").strip().lower()
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


#File parsing

SAMPLE_RE = re.compile(
    r"--- Sample (\d+) ---\n"
    r"Prompt \(last 200 chars\): .*?\n"
    r"Golden label: (.*?)\n"
    r"Raw output: (.*?)(?=\n\n--- Sample |\n*\Z)",
    re.DOTALL,
)


def parse_raw_predictions(path):
#Return (y_true, y_pred) lists parsed from a raw_predictions_*.txt file.
    text = Path(path).read_text(encoding="utf-8")
    y_true, y_pred = [], []
    for match in SAMPLE_RE.finditer(text):
        _, gold, pred = match.groups()
        y_true.append(gold.strip())
        y_pred.append(pred.strip())
    return y_true, y_pred



#   Recall    = (Cor + 0.5*Par) / (Cor + Par + Mis)
#   Precision = (Cor + 0.5*Par) / (Cor + Par + Spu)

#Matching function based on the literature
def _phrases_overlap(a, b):
    if a in b or b in a:
        return True
    a_words = set(a.split())
    b_words = set(b.split())
    return bool(a_words & b_words)


def _relaxed_match_counts(true_set, pred_set):
    """Greedy bipartite match; returns (cor, par, mis, spu) for one sample."""
    true_remaining = set(true_set)
    pred_remaining = set(pred_set)
    cor = 0
    par = 0

    # Pass 1: exact matches first, so they aren't consumed by a partial pairing
    for p in list(pred_remaining):
        if p in true_remaining:
            cor += 1
            true_remaining.discard(p)
            pred_remaining.discard(p)

    # Pass 2: partial (overlapping, non-identical) matches among what's left
    for p in list(pred_remaining):
        match = next((t for t in true_remaining if _phrases_overlap(p, t)), None)
        if match is not None:
            par += 1
            true_remaining.discard(match)
            pred_remaining.discard(p)

    mis = len(true_remaining)   # gold phrases never matched
    spu = len(pred_remaining)   # predicted phrases never matched
    return cor, par, mis, spu


def _relaxed_prf1(cor, par, mis, spu):
    denom_r = cor + par + mis
    denom_p = cor + par + spu
    recall = (cor + 0.5 * par) / denom_r if denom_r > 0 else 0.0
    precision = (cor + 0.5 * par) / denom_p if denom_p > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _prf1(tp, fp, fn):
    #Strict precision/recall/F1 -- exact match only, no partial credit.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate(y_true, y_pred):
    strict_tp = strict_fp = strict_fn = 0
    relaxed_cor = relaxed_par = relaxed_mis = relaxed_spu = 0
    drug_tp = drug_fp = drug_fn = 0

    for true_str, pred_str in zip(y_true, y_pred):
        _, true_adrs = extract_adrs(true_str)
        _, pred_adrs = extract_adrs(pred_str)

        # strict: exact string match (this is what the original script computed)
        strict_tp += len(true_adrs & pred_adrs)
        strict_fp += len(pred_adrs - true_adrs)
        strict_fn += len(true_adrs - pred_adrs)

        # relaxed: half-credit partial-match scoring (standard SMM4H/CADEC definition)
        cor, par, mis, spu = _relaxed_match_counts(true_adrs, pred_adrs)
        relaxed_cor += cor
        relaxed_par += par
        relaxed_mis += mis
        relaxed_spu += spu

        # drug set stays exact-match (drug names don't usually need relaxed
        # matching the way multi-word ADR phrases do -- adjust if you want
        # relaxed drug matching too, same function applies)
        _, true_drugs = extract_drug(true_str)
        _, pred_drugs = extract_drug(pred_str)
        drug_tp += len(true_drugs & pred_drugs)
        drug_fp += len(pred_drugs - true_drugs)
        drug_fn += len(true_drugs - pred_drugs)

    strict_p, strict_r, strict_f1 = _prf1(strict_tp, strict_fp, strict_fn)
    relaxed_p, relaxed_r, relaxed_f1 = _relaxed_prf1(relaxed_cor, relaxed_par, relaxed_mis, relaxed_spu)
    drug_p, drug_r, drug_f1 = _prf1(drug_tp, drug_fp, drug_fn)

    return {
        "n_samples": len(y_true),
        "strict_precision": strict_p, "strict_recall": strict_r, "strict_f1": strict_f1,
        "relaxed_precision": relaxed_p, "relaxed_recall": relaxed_r, "relaxed_f1": relaxed_f1,
        "drug_precision": drug_p, "drug_recall": drug_r, "drug_f1": drug_f1,
    }


def main(paths, csv_path="relaxed_f1_results.csv"):
    if not paths:
        print("Usage: python recompute_relaxed_f1.py raw_predictions_*.txt [--csv output.csv]")
        return

    rows = []
    for path in paths:
        tag = Path(path).stem.replace("raw_predictions_", "")
        y_true, y_pred = parse_raw_predictions(path)
        if not y_true:
            print(f"WARNING: parsed 0 samples from {path} -- check the file format matches SAMPLE_RE.")
            continue
        metrics = evaluate(y_true, y_pred)
        metrics["tag"] = tag
        metrics["source_file"] = path
        rows.append(metrics)

        print(f"\n=== {tag} ({metrics['n_samples']} samples) ===")
        print(f"  ADR strict  P/R/F1 : {metrics['strict_precision']:.4f} / {metrics['strict_recall']:.4f} / {metrics['strict_f1']:.4f}")
        print(f"  ADR relaxed P/R/F1 : {metrics['relaxed_precision']:.4f} / {metrics['relaxed_recall']:.4f} / {metrics['relaxed_f1']:.4f}")
        print(f"  Drug        P/R/F1 : {metrics['drug_precision']:.4f} / {metrics['drug_recall']:.4f} / {metrics['drug_f1']:.4f}")

    if not rows:
        print("\nNo files parsed successfully -- nothing to write to CSV.")
        return

    if len(rows) > 1:
        print("\n=== Comparison across runs (sorted by relaxed F1) ===")
        rows.sort(key=lambda r: r["relaxed_f1"], reverse=True)
        for r in rows:
            print(f"  {r['tag']:<40s} relaxed_f1={r['relaxed_f1']:.4f}  strict_f1={r['strict_f1']:.4f}")

    # Compile every run's metrics into a single CSV, one row per tag.
    column_order = [
        "tag", "source_file", "n_samples",
        "strict_precision", "strict_recall", "strict_f1",
        "relaxed_precision", "relaxed_recall", "relaxed_f1",
        "drug_precision", "drug_recall", "drug_f1",
    ]
    df_out = pd.DataFrame(rows)[column_order]
    df_out.to_csv(csv_path, index=False)
    print(f"\nCompiled results for {len(rows)} run(s) written to {csv_path}")


if __name__ == "__main__":
    # Expand any glob patterns the shell didn't already expand, and pull out
    # an optional --csv output_path.csv argument.
    raw_args = sys.argv[1:]
    csv_out = "relaxed_f1_results.csv"
    if "--csv" in raw_args:
        idx = raw_args.index("--csv")
        csv_out = raw_args[idx + 1]
        raw_args = raw_args[:idx] + raw_args[idx + 2:]

    all_paths = []
    for arg in raw_args:
        expanded = glob.glob(arg)
        all_paths.extend(expanded if expanded else [arg])
    main(all_paths, csv_path=csv_out)