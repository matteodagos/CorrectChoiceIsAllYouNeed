# Correct Choice Is All You Need
### MCQA Fine-Tuning Pipeline for an EPFL-Specialized Tutor Based on Qwen3-0.6B

> My contribution to the group project: data collection, dataset curation, and supervised fine-tuning of a compact STEM multiple-choice question answering model.

> The full project (DPO, Quantization, RAG) is documented in the [paper](./pdf/Intelligent_Agents.pdf).

---

## My Contribution

This repository covers **my personal share** of the group project for the EPFL Intelligent Agents course:

- **Data retrieval & cleaning** — parsing and normalising three question banks (Canterbury CS, NLP4Education, EPFL M1 open-answer data) and assembling the final training/validation/test mixtures
- **SFT training** — full fine-tuning of `Qwen3-0.6B-Base` on a curated 27k-item MCQA mixture using the Unsloth + TRL stack
- **Benchmark generation** — building and uploading the evaluation splits to HuggingFace
- **Report writing** — sections on data, training, and results

The other components of the full pipeline (DPO by Jacopo, Quantization by Sajal, RAG by Michele) are not included here.

---

## Results

My SFT checkpoint raises normalised multi-token accuracy on the hidden `M3_TEST` set from **56.8% → 64.5%** (+7.7 pp) over the base model, with similar gains on two out-of-domain splits.

| Model | M3\_test | M3\_NLP4Education | M3\_Canterbury Code |
|---|---|---|---|
| Qwen3-0.6B-Base (baseline) | 0.5676 | 0.4140 | 0.3592 |
| Qwen3-1.7B-Base (baseline) | 0.7154 | 0.4938 | 0.4933 |
| **M3\_mcqa\_model (mine)** | **0.6445** | **0.4763** | **0.4188** |

*Multi-token normalised accuracy (acc\_norm)*

---

## Repository Structure

```
.
├── Intelligent_Agents.pdf           # Full group project report
│
├── Data cleaning & upload
│   ├── clean_canterbury.py          # Parse & upload Canterbury CS questionbank to HF
│   ├── clean_nlp4education.py       # Parse & upload NLP4Education MCQs to HF
│   ├── clean_M1_data.py             # Clean M1 EPFL open-answer data with GPT-assisted rationales
│   ├── upload_M3_dataset_train.py   # Build & upload the MCQA training mixture to HF
│   ├── upload_M3_dataset_validation.py  # Build & upload the validation mixture to HF
│   └── upload_M3_test.py            # Build & upload the test split to HF
│
└── Training
    ├── train_mcqa_model.py          # Full SFT training loop (Unsloth + TRL)
    ├── train_mcqa.sh                # End-to-end shell script: venv setup → train
    ├── mcqa_model.yaml              # LightEval config pointing to the HF model checkpoint
    └── requirements.txt             # Full dependency list
```

---

## Datasets

All datasets are public on HuggingFace:

| HF Repository | Split | Description |
|---|---|---|
| `matteodagos/MNLP_M3_mcqa_dataset` | train / validation | 27k MCQA training mixture |
| `matteodagos/M3_test` | test | 2 551-item evaluation split (ARC, MMLU, EPFL-MCQs) |
| `matteodagos/M3_canterbury_code` | test | Canterbury CS questionbank |
| `matteodagos/M3_nlp4education` | test | NLP4Education EPFL questions |
| `matteodagos/EPFL_MCQs` | — | In-house EPFL MCQ dataset |

### Training Mixture Composition

| Source | # Train | # Dev |
|---|---|---|
| SciQ | 11 679 | 1 000 |
| ScienceQA (text-only) | 506 | 190 |
| ARC-Easy | 2 241 | 567 |
| ARC-Challenge | 1 117 | 295 |
| EPFL-MCQs | 417 | 51 |
| MedMCQA | 10 000 | 500 |
| MMLU (aux train) | 1 569 | 296 |
| **Total** | **27 529** | **2 899** |

All items are normalised to exactly **four A–D alternatives**. Questions with fewer than four choices are discarded; questions with more than four have three distractors sampled uniformly from the wrong options and reshuffled to eliminate position bias.

---

## Training

### Setup

```bash
bash train_mcqa.sh
```

This script creates a virtual environment, installs all dependencies from `requirements.txt`, and runs `train_mcqa_model.py`.

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen3-0.6B-Base` |
| Full fine-tuning | ✅ |
| Effective batch size | 64 (4 × 16 grad. accum.) |
| Epochs | 2 |
| Peak learning rate | 2e-5 (linear decay) |
| Warmup | 5% |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Precision | bf16 |
| Hardware | Single A100 40GB |

### Prompt Format

```
The following are multiple choice questions (with answers) about knowledge
and skills in advanced master-level STEM courses.

{QUESTION}
A. {choice_1}
B. {choice_2}
C. {choice_3}
D. {choice_4}
Answer:
```

The completion is the gold letter followed by the choice text (e.g. `A. Paris`). No rationale is appended — appending rationales was found to dilute probability mass and hurt accuracy.

---

## Evaluation

Evaluation uses the [LightEval](https://github.com/huggingface/lighteval) framework. The model config is provided in `mcqa_model.yaml`:

```yaml
model_args: "pretrained=matteodagos/MNLP_M3_mcqa_model,revision=main"
dtype: "float16"
```

At inference, all choices are scored in parallel via their total log-probability; the choice with the highest score is selected as the prediction.

---

## Model Checkpoint

The trained model is available on HuggingFace: [`matteodagos/MNLP_M3_mcqa_model`](https://huggingface.co/matteodagos/MNLP_M3_mcqa_model)

---

## Full Project & Team

This repo is part of a larger group project. The complete pipeline also includes DPO, Quantization, and RAG modules developed by my teammates.

| Name | Contribution | Email |
|---|---|---|
| **Matteo D'Agostino** | **Data, SFT training, benchmarks** | matteo.dagostino@epfl.ch |
| Sajal Chaurasia | Quantization | sajal.chaurasia@epfl.ch |
| Michele Smaldone | RAG | michele.smaldone@epfl.ch |
| Jacopo Ferro | DPO | jacopo.ferro@epfl.ch |

*EPFL — Intelligent Agents course project*

---

## Citation

```
@misc{dagostino2025correctchoice,
  title   = {Correct Choice Is All You Need},
  author  = {D'Agostino, Matteo and Chaurasia, Sajal and Smaldone, Michele and Ferro, Jacopo},
  year    = {2025},
  note    = {EPFL Intelligent Agents course project}
}
```
