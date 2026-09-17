# Susru: Capsule Endoscopy Fine-Tuning Demo

Susru is a terminal-based assistant fine-tuned from **Qwen2.5-0.5B** using LoRA adapters. It accepts a capsule-endoscopy disease/finding and a numeric severity score, then generates an educational, structured report.

> Medical safety: Susru is for education and prototype testing only. It does not diagnose disease, replace a clinician, or prescribe treatment. The synthetic dataset must be reviewed and validated by qualified clinicians before any patient-facing or clinical use.

## What the project includes

- A memory-conscious Rust/Candle LoRA trainer for Qwen2.5-0.5B.
- A 100,000-record synthetic capsule-endoscopy fine-tuning dataset (JSONL).
- A converter that turns the JSONL training split into the TSV the trainer reads.
- A terminal UI for testing the saved fine-tuned adapter.
- Disease/finding examples including Crohn's disease, small-bowel bleeding, angiodysplasia, celiac disease, ulcers, polyps, tumors, strictures, capsule retention, and esophageal varices.

## Requirements

- Rust and Cargo (stable toolchain).
- Python 3 (used only for dataset generation/conversion).
- A CUDA-compatible GPU. The project is configured for the 4–6 GB VRAM training profile.
- Internet access on the first run, unless the base model has already been downloaded.

## Project layout

```text
data/generate_100k_jsonl.py    deterministic 100,000-record JSONL generator
data/capsule_dataset.json      full 100,000-record dataset as a single JSON array
data/capsule_train.jsonl       training split (90,000 records)
data/capsule_validation.jsonl  validation split (5,000 records)
data/capsule_test.jsonl        test split (5,000 records)
data/convert_jsonl_to_tsv.py   creates percentage TSV subsets for the Rust trainer
data/train_10.tsv ... train_90.tsv  10% through 90% Rust-training subsets
models/Qwen2.5-0.5B/           downloaded base-model files
qwen2.5-0.5b-qv-lora.safetensors  trained LoRA adapter
train_susru_unsloth.ipynb      Colab notebook: fine-tune with Unsloth/QLoRA
src/main.rs                    training and terminal UI commands
```

## 1. Generate the fine-tuning dataset

The repository includes the generated split files (`capsule_train.jsonl`,
`capsule_validation.jsonl`, `capsule_test.jsonl`). Regenerate the full
100,000-record corpus when needed:

```powershell
python .\data\generate_100k_jsonl.py
```

This writes the split files (`capsule_train.jsonl`, `capsule_validation.jsonl`,
`capsule_test.jsonl`). To build a single JSON array containing the full
100,000-record dataset (used by the Unsloth/Colab notebook), run:

```powershell
python .\data\merge_to_json.py
```

This writes `data/capsule_dataset.json`. The generator is deterministic and
produces records in batches of 1,000 with a live manifest. It covers 43
diseases/findings across inflammatory, vascular, bleeding, neoplastic,
malabsorption, infectious, structural, and polyposis categories, with exactly
33,333 mild / 33,333 moderate / 33,334 severe records.

The Rust trainer defaults to the deterministic 50% subset (45,000 of the
90,000 training records) to reduce training time. To rebuild a selected subset:

```powershell
python .\data\convert_jsonl_to_tsv.py
```

For example, create the 30% subset with:

```powershell
python .\data\convert_jsonl_to_tsv.py --percent 30
```

The TSV uses this internal format (one record per line; `\n` is stored as
literal characters so the trainer converts them into real newlines before
tokenization):

```text
Disease/finding: Crohn's disease\nSeverity: moderate\nResponse:<TAB>Findings: ...\nImpression: ...\nClinical significance: ...\nRecommendation: ...
```

## 2. Download the base model for offline use (optional)

This command downloads and saves the Qwen base-model files under `models/Qwen2.5-0.5B`:

```powershell
cargo run --release -- prepare-offline
```

After the files are downloaded, set `QWEN_OFFLINE=1` if you want to ensure the program does not try to access the network:

```powershell
$env:QWEN_OFFLINE = "1"
```

## 3. Train Susru

Run the LoRA fine-tuning process:

```powershell
cargo run --release -- train
```

The trainer reads `data/train_50.tsv` (45,000 records), performs three epochs by default, and writes the adapter to:

```text
qwen2.5-0.5b-qv-lora.safetensors
```

### Training options

```powershell
cargo run --release -- train --epochs 3 --batch-size 8
```

Select one of the generated dataset sizes with `--data-percent` (default: 50):

```powershell
cargo run --release -- train --epochs 3 --batch-size 8 --data-percent 30
```

Valid values are `10`, `20`, `30`, `40`, `50`, `60`, `70`, `80`, and `90`.

- `--epochs N` (default `3`, range 1–20). For a 0.5B model with rank-8 q/v
  LoRA on 45,000 examples, 2–4 epochs is the sweet spot; more than that tends
  to overfit and can cause the repeated-template output you may see in chat.
- `--batch-size N` (default `1`, range 1–64). Uses **gradient accumulation**:
  it sums `N` forward/backward passes into one optimizer update, so you get
  smoother gradients without using more VRAM. On a 6 GB GPU, batch 8–16 is
  comfortable.

The full 90,000-record dataset is intentionally much slower to train than a
small demo dataset. For a quick smoke test, create a temporary smaller copy of
the dataset first; use the full corpus for a longer training run.

## 3b. Train in Google Colab with Unsloth

Prefer a cloud GPU? Open `train_susru_unsloth.ipynb` in Google Colab
(File → Upload notebook, or upload it to Colab). It fine-tunes Qwen2.5-0.5B
with 4-bit QLoRA via Unsloth, which is much faster and more memory-efficient
than the local trainer.

Workflow:

1. Upload `data/capsule_dataset.json` to a Google Drive folder.
2. In the notebook's Configuration cell, set `DATASET_PATH` to that Drive path.
3. Run all cells; the LoRA adapter is saved back to Drive.

The notebook uses effective batch 16 (batch 2 × grad-accum 8), rank-16 LoRA on
`q/k/v/o`, and trains in minutes on a free T4 with the quick-test settings.

## 3c. Train with Python (QLoRA)

`finetune_capsule_qlora.py` fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` directly
from `data/capsule_dataset.json`. It preserves the structured input and trains
the model to generate only the structured report.

Install a CUDA-enabled PyTorch build appropriate for your GPU, then install the
remaining packages:

```powershell
python -m pip install -r requirements-finetune.txt
```

Run a short smoke test first:

```powershell
python .\finetune_capsule_qlora.py --max-steps 10
```

Then run the full two-epoch job:

```powershell
python .\finetune_capsule_qlora.py --epochs 2 --batch-size 2 --gradient-accumulation 8
```

The LoRA adapter and tokenizer are written to `outputs/susru-qwen-lora`. The
default 4-bit mode requires an NVIDIA CUDA GPU; use `--no-load-in-4bit` only
when you have enough RAM/VRAM for full-precision training.

## 4. Test the fine-tuned model
Start the terminal UI after training:

```powershell
cargo run --release -- chat
```

Susru starts with:

```text
I'm Susru, your assistant.
```

Type `/quit` or `/exit` to close the terminal UI.

## Input format

Enter one line using this exact shape:

```text
Diseases: <disease or finding>; Severity: <score from 0.0 to 1.0>
```

`Disease:` is also accepted in place of `Diseases:`. The severity score is converted internally to a training category:

| Score range | Category |
| --- | --- |
| 0.00–0.33 | mild |
| 0.34–0.66 | moderate |
| 0.67–1.00 | severe |

### Example input

```text
Diseases: Crohn's disease; Severity: 0.56
```

### Example output

```text
--- Susru Educational Capsule-Endoscopy Report ---
Finding: Crohn's disease
Severity score: 0.56
Severity category: moderate

Finding summary:
Moderate inflammatory changes, erosions, or ulcers may require timely
gastroenterology review. Capsule endoscopy alone does not establish a
diagnosis or treatment plan.

Limitations and next step:
This is educational information, not a diagnosis. Capsule images and symptoms
must be reviewed by a qualified clinician; seek urgent care for severe pain,
vomiting, black stools, vomiting blood, fainting, or worsening weakness.
--------------------------------------------------
```

## Name response

At the input prompt, entering either of the following returns Susru's name:

```text
What is your name?
Who are you?
```

Expected reply:

```text
I'm Susru, your assistant.
```

## Optional response-length setting

Susru generates up to 160 tokens by default. Use a different cap if needed:

```powershell
cargo run --release -- chat --max-new-tokens 96
```

## Verification commands

```powershell
cargo fmt --check
cargo check --offline
```
"# DynamicLLM_Runtime" 
