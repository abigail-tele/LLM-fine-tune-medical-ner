#!/usr/bin/env python3
"""Verify the golden-label pipeline of ver8-fine-tuning-combined-adr.py using
only the stdlib, so it runs on any machine (no pandas/GPU needed).

Usage: python3 verify_golden_labels.py [path-to-json]
"""
import json
import sys

JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else "combinedV5-cadec-psy-ade.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)          # list of dicts, order preserved

n = len(raw_data)
train_end = int(0.8 * n)
eval_end = train_end + int(0.1 * n)
test = raw_data[eval_end:]           # identical slicing to X_test = df[eval_end:]
print(f"Total: {n}  Train: {train_end}  Eval: {eval_end-train_end}  Test: {len(test)}")

# Exact copy of evaluate_adr.extract_adrs from the training script
def extract_adrs(text):
    text = text.strip()
    adr_part = text.split("|", 1)[1] if "|" in text else text
    if "ADRS:" in adr_part.lower():
        adr_part = adr_part.lower().split("ADRS:", 1)[1]
    return {t.strip().lower() for t in adr_part.split(";") if t.strip()}

problems, sizes = [], []
missing_keys = no_pipe = no_adrs = empty_adrs = plural = singular = 0

for i, rec in enumerate(test, start=eval_end):
    if not {"instruction", "input", "output"} <= set(rec):
        missing_keys += 1; problems.append((i, "missing keys", str(sorted(rec))))
        continue
    out = rec["output"].strip()
    if "|" not in out:
        no_pipe += 1; problems.append((i, "no '|'", out[:80]))
    elif "ADRS:" not in out.split("|", 1)[1].upper():
        no_adrs += 1; problems.append((i, "no ADRs: after pipe", out[:80]))
    adr = extract_adrs(out); sizes.append(len(adr))
    if not adr:
        empty_adrs += 1; problems.append((i, "empty extract_adrs", out[:80]))
    if "Drugs:" in out: plural += 1
    elif "Drug:" in out: singular += 1

print(f"Missing keys: {missing_keys}")
print(f"No '|': {no_pipe} | No 'ADRs:' after pipe: {no_adrs} | Empty ADR set: {empty_adrs}")
print(f"Labels 'Drugs:' (plural): {plural} | 'Drug:' (singular): {singular}")
if sizes:
    print(f"ADR tokens/label: min {min(sizes)}  max {max(sizes)}  mean {sum(sizes)/len(sizes):.1f}")
print("Test rows: %d - %d" % (eval_end, n - 1))
for rec in (test[:3] + test[-3:]):
    print("  ", rec["output"][:120])

if problems:
    print(f"\nPROBLEMS: {len(problems)}")
    for i, kind, s in problems[:10]:
        print(f"  row {i}: {kind} | {s!r}")
    sys.exit(1)
print("\nALL GOLDEN-LABEL CHECKS PASSED")
