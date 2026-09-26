# Combining Multi-Source Patient Narratives to Fine-Tune a Large Language Model for Drug and Adverse Drug Reaction Extraction

Fine-tunes an open-source LLM (Llama 3.1 8B) via LoRA to jointly extract **drug** and **adverse drug reaction (ADR)** entities from patient narratives, using a combined corpus built from three complementary datasets.

## Overview

Adverse drug reactions are frequently underreported in structured clinical data but are often described informally by patients online. This project combines three annotated ADR datasets into a single instruction-tuning corpus and fine-tunes Meta-Llama-3.1-8B-Instruct (via LoRA) to extract drug and ADR entities directly from that informal text.

**Key result:** the best V12 fine-tuning run improved F1 over the zero-shot baseline by ~15.5 points (ADR strict), ~13.4 points (ADR relaxed), and ~13.3 points (drugs). See [Results](#results) for the full comparison including the few-shot baseline.

## Results

Headline numbers below are the best V12 sweep configuration (`batch 8 / grad_accum 2 / epochs 3`, lr 2e-4, LoRA r16/alpha32; 750 test samples), taken from `fine_tuning/v12_final/combined12_adr_sweep_results.csv` and `fine_tuning/v12_final/recomputed_metrics_results.csv`. Baselines are from `baselines/zeroshot_calc.csv` (zero-shot, no in-context examples) and `baselines/fewshot_calc.csv` (few-shot, N=4 in-context examples from the train split only).

ADR scores are reported under two definitions: **strict** (exact-phrase match, no partial credit) and **relaxed** (half-credit partial match for overlapping phrases, per the standard SMM4H/CADEC definition used in `recompute-metrics.py`). Drug scores use exact token-set match.

| Condition | ADR strict P / R / F1 | ADR relaxed P / R / F1 | Drug P / R / F1 |
|---|---|---|---|
| Fine-tuned (V12 best, b8_ga2_e3) | 76.69% / 76.39% / **76.54%** | 84.39% / 84.07% / **84.23%** | 80.25% / 80.15% / **80.20%** |
| Zero-shot baseline | 63.57% / 58.67% / 61.02% | 73.82% / 68.13% / 70.86% | 75.39% / 60.05% / 66.85% |
| Few-shot baseline (N=4) | 65.61% / 45.68% / 53.86% | 76.40% / 53.19% / 62.72% | 72.81% / 60.42% / 66.04% |

Notes:

- `recomputed_metrics_results.csv` labels the V12 rows with a stale `v13_` tag prefix (e.g. `v13_b8_ga2_e3_lr0.0002_r16`); these rows correspond to the V12 raw-prediction files in `fine_tuning/v12_final/Raw Predictions/`.
- That CSV also contains a `baseline_v12` row that differs slightly from `baselines/zeroshot_calc.csv` (strict F1 60.83 vs 61.02). The table above uses `baselines/zeroshot_calc.csv` and `baselines/fewshot_calc.csv` as the canonical baseline sources.

## Repository Structure

```
.
├── baselines/
│   ├── zero_example_baseline.py   # zero-shot eval on the base model (same prompt/split/parsing as fine-tuning)
│   ├── few_example_baseline.py    # few-shot eval (N=4 train-split examples in context, no weight updates)
│   ├── zeroshot_calc.csv          # zero-shot metrics (canonical baseline source)
│   └── fewshot_calc.csv           # few-shot metrics (canonical baseline source)
├── dataset_preparation/
│   ├── final_build_adr_dataset.py # merges CADECv2, ADE Corpus, and PsyTAR into one instruction-tuning corpus
│   ├── verify_golden_labels.py    # stdlib-only sanity check of golden-label format and 80/10/10 slicing
│   └── Summer Project Datasets.pdf# background notes on the source datasets
├── fine_tuning/
│   ├── v11/                       # legacy run (LoRA r64/alpha16, 4-config sweep); superseded by v12_final
│   │   ├── ver11-fine-tuning-combined.py
│   │   ├── combined11_adr_sweep_results.csv
│   └── v12_final/                 # canonical run (LoRA r16/alpha32, 6-config sweep)
│       ├── ver12(final)-fine-tuning-combined.py
│       ├── recompute-metrics.py   # strict + relaxed P/R/F1 scoring from raw prediction files
│       ├── combined12_adr_sweep_results.csv
│       ├── recomputed_metrics_results.csv
├── requirements.txt
└── README.md
```

**Note:** the combined training file this project produces (default `combinedV4_adr_dataset.json`; the training and baseline scripts expect `combinedV5-cadec-psy-ade.json` in the working directory) is **not committed to this repository** — see [Datasets & Licensing](#datasets--licensing) for why, and how to regenerate it yourself.

## Installation

```bash
git clone https://github.com/abigail-tele/LLM-fine-tune-medical-ner.git
cd LLM-fine-tune-medical-ner
pip install -r requirements.txt
```

`requirements.txt` is split into two sections in one file:

1. **Dataset preparation** (CPU-only): `pandas`, `openpyxl`, `requests`.
2. **Fine-tuning, baselines, and scoring** (GPU): `torch` (CUDA-capable build), `transformers==4.49.0`, `peft==0.14.0`, `accelerate==1.3.0`, `trl>=0.20`, plus `datasets`, `tqdm`, `numpy`, `huggingface_hub`, `python-dotenv`.

Gated-model access: fine-tuning and baseline scripts load `meta-llama/Meta-Llama-3.1-8B-Instruct`, which requires Hugging Face access approval and an `HF_TOKEN` in a `.env` file (loaded via `python-dotenv`).

## Usage

1. **Build the combined dataset** (requires you to obtain CADECv2 and PsyTAR yourself first — see below):

   ```bash
   python dataset_preparation/final_build_adr_dataset.py --cadec <path/to/cadec.json> --psytar <path/to/PsyTAR_dataset.xlsx> --out combinedV5-cadec-psy-ade.json
   ```

   ADE Corpus is downloaded automatically from its public GitHub mirror. `load_psytar` reads the `ADR_Identified` sheet via `openpyxl`. Optional: `python dataset_preparation/verify_golden_labels.py <path/to/combined.json>` checks golden-label formatting (stdlib only, no GPU needed).

2. **Fine-tune the model** (CUDA GPU required):

   ```bash
   python "fine_tuning/v12_final/ver12(final)-fine-tuning-combined.py"
   ```

   The script reads `combinedV5-cadec-psy-ade.json` from the working directory (`JSON_PATH` is hardcoded — place or rename your built file accordingly) and writes `raw_predictions_<tag>.txt` files plus `combined12_adr_sweep_results.csv`. `fine_tuning/v11/ver11-fine-tuning-combined.py` is the legacy run and is not needed to reproduce the headline results.

3. **Score predictions** (strict + relaxed precision/recall/F1):

   ```bash
   python fine_tuning/v12_final/recompute-metrics.py "fine_tuning/v12_final/Raw Predictions/raw_predictions_*.txt" --csv results.csv
   ```

   Without `--csv`, results default to `relaxed_f1_results.csv`.

4. **Run baselines** (CUDA GPU required, same prompt/split/parsing as fine-tuning):

   ```bash
   python baselines/zero_example_baseline.py
   python baselines/few_example_baseline.py
   ```

## Datasets & Licensing

This project does not redistribute any source dataset directly. Each has its own license, and one (CADECv2) explicitly prohibits redistribution — so the merged training corpus (which contains CADECv2-derived text) is excluded from this repo. Get each dataset from its official source and regenerate the combined file locally with `dataset_preparation/final_build_adr_dataset.py`.

| Dataset | License | Redistribute? | Official source |
|---|---|---|---|
| PsyTAR | CC BY 4.0 | Yes, with attribution | [Mendeley Data](https://data.mendeley.com) / [BigBio](https://huggingface.co/datasets/bigbio/psytar) — confirm terms on the record you use |
| ADE Corpus V2 | CC BY 4.0 (BMC Bioinformatics, open access) | Yes, with attribution | [GitHub mirror](https://github.com/trunghlt/AdverseDrugReaction) (used directly by the build script) |
| CADECv2 | CSIRO Data Licence — **non-commercial, non-transferable** | **No** — do not redistribute raw files or any corpus built from it | [CSIRO Data Access Portal, DOI: 10.25919/3v5b-k950](https://doi.org/10.25919/3v5b-k950) |

### Citations

1. Zolnoori M, Fung KW, Patrick TB, Fontelo P, Kharrazi H, Faiola A, et al. The PsyTAR dataset: From patients generated narratives to a corpus of adverse drug events and effectiveness of psychiatric medications. Data Brief. 2019 Jun;24:103838.
2. Gurulingappa H, Rajput AM, Roberts A, Fluck J, Hofmann-Apitius M. Development of a benchmark corpus to support the extraction of entity associations and their relationships from biomedical literature. BMC Bioinformatics. 2012;13(1):1-10.
3. Dai X. CADECv2. Version 4 [dataset]. Canberra (AU): CSIRO; 2024. Available from: https://doi.org/10.25919/3v5b-k950
4. Dubey A, Jauhri A, Pandey A, Kadian A, Al-Dahle A, Letman A, et al.; Llama Team. The Llama 3 Herd of Models. arXiv [Preprint]. 2024.

## Model & Method

Results reported above are from `fine_tuning/v12_final` (the `fine_tuning/v11` run used LoRA r64/alpha16 with a 4-config sweep and is superseded):

- **Base model:** meta-llama/Meta-Llama-3.1-8B-Instruct
- **Fine-tuning:** LoRA (rank 16, alpha 32) across all linear layers; AdamW, cosine schedule, lr 2e-4, fp16
- **Hyperparameter sweep:** 6 configs varying batch size, gradient accumulation, and epochs (effective batch 16–32, 1–3 epochs)
- **Data split:** 80/10/10 train/eval/test slicing (same slicing as `verify_golden_labels.py`)
- **Evaluation:** strict (exact match) and relaxed (half-credit partial match) precision/recall/F1 for ADRs; exact token-set match for drugs

## Limitations & Future Work

- Model size was constrained by available compute — Llama 3.1 8B was the largest model that fit.
- Error analysis of the baseline showed most missed extractions came from PsyTAR sentences where the labeled drug was never present in the input text itself (it was only recorded at the post level). Future work should standardize entity-linked context for datasets like PsyTAR that label drugs at the post level but text at the sentence level.
- Generalization beyond the three combined training sources is untested; validating on an external dataset (e.g., SMM4H) is a natural next step.

## License

There is currently no `LICENSE` file in this repository. Treat the code as all-rights-reserved unless a license is added. **In any case this covers the code only**; the datasets used are governed by their own separate licenses listed above and are not included here.

## Contact

For questions about the dataset pipeline, fine-tuning setup, or results, please contact the repository owner, Abigail Televitckiy, via GitHub Issues on this repository. When reporting a problem, include the script and configuration you ran, your environment (GPU, package versions), and the exact error message or unexpected output so the issue can be reproduced.

## Acknowledgments

Abigail Televitckiy, Texas Academy of Mathematics and Science. Mentored by Dr. Heejun Kim, University of North Texas, School of Information Science.
