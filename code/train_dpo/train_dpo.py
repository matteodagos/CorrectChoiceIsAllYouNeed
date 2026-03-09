# Training Script to recreate the DPO model

# ----

import os

os.environ["CC"] = os.path.expanduser("~/gcc_wrapper")
os.environ.pop("GCC_EXEC_PREFIX", None)

print("GCC wrapper set")

# -----

# Imports
import datetime
from datasets import load_dataset, Dataset, DatasetDict
import unsloth
import numpy as np
import torch
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel

print("Imports done")

# -----
# DATA and MODEL SETUP
# Load Dataset from Hub (use the dataset names from your data pipeline)
dataset_name = (
    "ciacco/MNLP_M3_dpo_dataset"  # Use the dataset repo ID from data_repo.json
)
print(f"Loading dataset: {dataset_name}")
dataset = load_dataset(dataset_name)

print(f"Train dataset: {len(dataset['train'])} samples")
print(f"Eval dataset: {len(dataset['eval'])} samples")
print(f"Test dataset: {len(dataset['test'])} samples")
# Setup Unsloth Model and LoRA
print("Loading model with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B-Base",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    # load_in_4bit=True, # note: even if this is commented out, and model is specified as base, unsloth will load a dynamic quantized 4 bit version of qwen3-0.6B-Base
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,  # "unsloth",
    random_state=3407,  # so will be reproducible
    use_rslora=False,
    loftq_config=None,
)

print(f"Model loaded successfully")
print(f"CUDA available: {torch.cuda.is_available()}")

# Training Configuration
experiment_name = "jferro_unsloth_dpo_large_combined"
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
run_name = f"{experiment_name}_{run_timestamp}"
output_dir = f"outputs/dpo_{run_name}"

print(f"Run name: {run_name}")
print(f"Output directory: {output_dir}")

# -------

# DPO Configuration


# Create DPO Trainer with Unsloth optimizations
training_args = DPOConfig(
    output_dir=output_dir,
    num_train_epochs=2,  # 2 epochs as planned
    per_device_train_batch_size=2,  # Try 2 with Unsloth optimizations
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Conservative but should work
    learning_rate=3e-6,  # Lower LR for DPO
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,  # More frequent evaluation
    save_strategy="steps",
    save_steps=50,  # Frequent saves for crash recovery
    save_total_limit=10,  # Keep more checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_8bit",
    weight_decay=0.01,
    seed=42,
    run_name=run_name,
    beta=0.1,  # DPO beta parameter
    report_to="tensorboard",  # might be hard to access from gnoto (I did not manage even with many tricks), but logs can be downloaded and seen in local with GUI
    logging_dir=f"logs/{run_name}",
    max_length=2048,
    max_prompt_length=1024,
    gradient_checkpointing=True,
    dataloader_drop_last=True,
    remove_unused_columns=False,
    bf16=True,  # Better stability than fp16
    dataloader_num_workers=0,  # Avoid multiprocessing issues
    push_to_hub=True,  # Auto-push checkpoints
    hub_model_id=f"ciacco/dpo_checkpoints_{run_name}",  # change this to your name, you can go check my repo if needed, was uploading, the final is checkpoint 2900 because of OOM
    hub_strategy="checkpoint",
    hub_private_repo=True,
)

# Initialize DPO trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Unsloth will handle reference model
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    tokenizer=tokenizer,
    processing_class=tokenizer,
)

# Enable Unsloth training mode
FastLanguageModel.for_training(model)

print("DPO Trainer initialized and ready for training")

# --------
# TRAINING

# Train Model
print(f"Starting Unsloth DPO training: {run_name}")
print("=" * 60)
print(f"Training samples: {len(dataset['train'])}")
print(f"Evaluation samples: {len(dataset['eval'])}")
print(
    f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
)
print("=" * 60)

try:
    dpo_trainer.train()
    print("Training completed successfully!")

except Exception as e:
    print(f"Training interrupted: {e}")
    print("Checking for checkpoints...")

    # Try to resume from latest checkpoint
    latest_checkpoint = dpo_trainer.get_last_checkpoint(output_dir)
    if latest_checkpoint:
        print(f"Resuming from: {latest_checkpoint}")
        dpo_trainer.train(resume_from_checkpoint=latest_checkpoint)

# ------

# after 11h+ it crashed with OOM, UI gave up even before but ok,
# looking at logs from tensorboard everything was smooth, you can check them in logs/unsloth_dpo_large_combined_20250609_1741
# tensorboard --logdir logs/unsloth_dpo_large_combined_20250609_1741

# Starting Unsloth DPO training: unsloth_dpo_large_combined_20250609_1741
# ============================================================
# Training samples: 30068
# Evaluation samples: 3758
# Effective batch size: 16
# ============================================================
# ==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
#    \\   /|    Num examples = 30,068 | Num Epochs = 2 | Total steps = 3,758
# O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 8
# \        /    Data Parallel GPUs = 1 | Total batch size (2 x 8 x 1) = 16
#  "-____-"     Trainable parameters = 10,092,544/600,000,000 (1.68% trained)
# Unsloth: Will smartly offload gradients to save VRAM!
#  [ 660/3758 2:31:52 < 11:55:01, 0.07 it/s, Epoch 0.35/2]
# Step    Training Loss   Validation Loss rewards / chosen    rewards / rejected  rewards / accuracies    rewards / margins   logps / chosen  logps / rejected    logits / chosen logits / rejected   eval_logits / chosen    eval_logits / rejected  nll_loss    aux_loss
# 50  0.691900    0.693741    0.003214    0.001493    0.482437    0.001721    -413.485474 -320.115845 -0.744739   -0.662467   0   0   0   0
# 100 0.685600    0.684637    0.028947    0.008401    0.565726    0.020546    -413.228119 -320.046753 -0.754971   -0.668336   No Log  No Log  No Log  No Log
# 150 0.662900    0.664290    0.083958    0.016223    0.618148    0.067735    -412.678009 -319.968567 -0.781449   -0.682863   No Log  No Log  No Log  No Log
# 200 0.634800    0.639306    0.161930    0.026551    0.647951    0.135379    -411.898254 -319.865295 -0.804779   -0.692837   No Log  No Log  No Log  No Log
# 250 0.616800    0.617253    0.233462    0.028904    0.670037    0.204558    -411.182953 -319.841736 -0.826070   -0.702554   No Log  No Log  No Log  No Log
# 300 0.605500    0.597410    0.283337    0.009806    0.674029    0.273530    -410.684204 -320.032684 -0.840275   -0.707741   No Log  No Log  No Log  No Log
# 350 0.611600    0.579649    0.344325    0.004025    0.693454    0.340300    -410.074371 -320.090546 -0.831071   -0.694419   No Log  No Log  No Log  No Log
# 400 0.576100    0.567288    0.399058    0.002979    0.693720    0.396080    -409.526978 -320.101013 -0.830184   -0.689354   No Log  No Log  No Log  No Log
# 450 0.563100    0.556626    0.436847    -0.009831   0.701171    0.446678    -409.149139 -320.229065 -0.837922   -0.694027   No Log  No Log  No Log  No Log
# 500 0.544000    0.544199    0.472517    -0.034629   0.703566    0.507146    -408.792419 -320.477112 -0.835320   -0.688411   No Log  No Log  No Log  No Log
# 550 0.517800    0.537025    0.502091    -0.049877   0.707025    0.551968    -408.496674 -320.629547 -0.845554   -0.696526   No Log  No Log  No Log  No Log
# 600 0.539600    0.529349    0.530914    -0.072034   0.708089    0.602948    -408.208466 -320.851105 -0.834885   -0.684805   No Log  No Log  No Log  No Log
# 650 0.504400    0.523724    0.550858    -0.089661   0.706493    0.640519    -408.009033 -321.027405 -0.827921   -0.679065   No Log  No Log  No Log  No Log

# ------


# Save and Push Final Model
print("Saving final model...")

# Save LoRA model locally
dpo_trainer.save_model()

# Merge LoRA with base model
print("Merging LoRA adapters...")
merged_model = model.merge_and_unload()

# Save merged model
final_model_name = f"ciacco/MNLP_M3_unsloth_dpo_{run_timestamp}"  # change this to your name, you can go check my repo if needed
lora_model_name = f"ciacco/MNLP_M3_unsloth_dpo_lora_{run_timestamp}"  # change this to your name, you can go check my repo if needed

try:
    # Push merged model
    merged_model.push_to_hub(final_model_name, private=True)
    tokenizer.push_to_hub(final_model_name, private=True)
    print(f"✅ Merged model pushed: {final_model_name}")

    # Push LoRA adapters separately
    model.push_to_hub(lora_model_name, private=True)
    print(f"✅ LoRA adapters pushed: {lora_model_name}")

except Exception as e:
    print(f"Error pushing to hub: {e}")
    print(f"Models saved locally in: {output_dir}")

print(f"\n🎉 Training completed!")
print(f"📊 Merged model: {final_model_name}")
print(f"🔧 LoRA adapters: {lora_model_name}")
print(f"📈 Training logs: {output_dir}")
print(f"📊 TensorBoard: tensorboard --logdir logs/")

# -----
# MERGE LORA WITH BASE MODEL IF CHECKPOINTS NEEDED - see README describing notebooks
