# Correct Choice Is All You Need
### A Compact EPFL-Specialized MCQA Tutor Based on Qwen3-0.6B

> Fine-tuning, preference alignment, quantization, and RAG-augmented inference for multiple-choice question answering in advanced STEM courses.

---

## Overview

This project transforms the open-source **Qwen3-0.6B-Base** model into a lightweight EPFL-style academic tutor capable of answering multiple-choice STEM questions. The full pipeline covers:

1. **Supervised Fine-Tuning (SFT)** on a curated 27k-item English MCQA mixture
2. **Direct Preference Optimisation (DPO)** on 37.5k answer pairs for cleaner, pedagogically aligned outputs
3. **Post-Training Quantization** (SmoothQuant + GPTQ) compressing the model to a ~510 MB W4A8 checkpoint
4. **Retrieval-Augmented Generation (RAG)** with multiple STEM corpora and embedding models

The SFT checkpoint raises normalised multi-token accuracy on the hidden `M3_TEST` set from **56.8% → 64.5%** (+7.7 pp) over the base model. DPO adds ~1 pp. The quantized model fits on student hardware with only a ~9% accuracy drop.

---

## Results

| Model | M3\_test | M3\_NLP4Education | M3\_Canterbury Code |
|---|---|---|---|
| Qwen3-0.6B-Base | 0.5676 | 0.4140 | 0.3592 |
| Qwen3-1.7B-Base | 0.7154 | 0.4938 | 0.4933 |
| **M3\_mcqa\_model (ours)** | **0.6445** | **0.4763** | **0.4188** |
| M3\_quantized\_model | 0.5527 | 0.4002 | 0.3487 |
| M3\_mcqa\_model + RAG | 0.6350 | 0.4730 | 0.4277 |

*Multi-token normalised accuracy (acc\_norm)*

---

## Repository Structure

```
.
├── clean_canterbury.py          # Parse & upload Canterbury CS questionbank to HF
├── clean_nlp4education.py       # Parse & upload NLP4Education MCQs to HF
├── clean_M1_data.py             # Clean M1 EPFL open-answer data with GPT-assisted rationales
├── upload_M3_dataset_train.py   # Build & upload the MCQA training mixture to HF
├── upload_M3_dataset_validation.py  # Build & upload the validation mixture to HF
├── upload_M3_test.py            # Build & upload the test split to HF
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

## Quantization

Post-training quantization is applied to the final SFT checkpoint using [LLM Compressor](https://github.com/vllm-project/llm-compressor):

- **SmoothQuant** (smoothing strength 0.7): transfers quantization difficulty from activations to weights
- **GPTQ**: W4 INT Static Symmetric Channel + A8 INT Dynamic Symmetric Token
- **Result**: ~510 MB checkpoint, ~9% accuracy drop vs. the full-precision model

The `lm_head` layer is intentionally left unquantized to preserve output reliability.

---

## RAG

The retrieval module indexes STEM-centric corpora (ArXiv MNLP, MIT Lectures, OpenStax, WIKI-STEM, a cyber-security book) into IVF-PQ indices. At inference, top-20 chunks are retrieved and concatenated with the question before being passed to the frozen MCQA model.

**Key finding:** Swapping the default `all-MiniLM-L12-v2` retriever for **LaBSE** yields a consistent +0.3–1.2 pp gain across all splits at zero training cost, attributed to LaBSE's stronger handling of technical jargon.

---

## DPO

Direct Preference Optimisation trains on 37.5k triplets `{question, chosen_answer, rejected_answer}` using Unsloth LoRA (rank 16) for two epochs. Interpolating the DPO weights with the SFT checkpoint at α = 0.5 gives the best combined accuracy (0.648 on `M3_test`).

---

## Model Checkpoint

The trained model is available on HuggingFace: [`matteodagos/MNLP_M3_mcqa_model`](https://huggingface.co/matteodagos/MNLP_M3_mcqa_model)

---

## Authors

| Name | Email |
|---|---|
| Matteo D'Agostino | matteo.dagostino@epfl.ch |
| Sajal Chaurasia | sajal.chaurasia@epfl.ch |
| Michele Smaldone | michele.smaldone@epfl.ch |
| Jacopo Ferro | jacopo.ferro@epfl.ch |

*EPFL — Intelligent Agents course project*

---

## Citation

If you use this work, please cite the accompanying report:

```
@misc{dagostino2025correctchoice,
  title   = {Correct Choice Is All You Need},
  author  = {D'Agostino, Matteo and Chaurasia, Sajal and Smaldone, Michele and Ferro, Jacopo},
  year    = {2025},
  note    = {EPFL Intelligent Agents course project}
}
```