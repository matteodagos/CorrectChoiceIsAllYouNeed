import os
import torch, numpy as np
from tqdm import tqdm
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import unsloth_train
from trl import SFTConfig, SFTTrainer

del os.environ["GCC_EXEC_PREFIX"]
os.environ["CC"] = "/home/gcc_wrapper"

def preprocess_train(ex):
    prompt = f"The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n"
    prompt += ex["QUESTION"] + "\n"
    if len(ex["CHOICES"]) > 1:
        letters = "ABCDEFGHIJKLMNO"
        letters = letters[:len(ex["CHOICES"])]
        prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(letters, ex["CHOICES"])])
    prompt += "Answer:"
    completion = ex["ANSWER"]
    return {"prompt": prompt, "completion": completion}
def preprocess_validation(ex):
    prompt = f"The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n"
    prompt += ex["QUESTION"] + "\n"
    if len(ex["CHOICES"]) == 4:
        prompt += "".join([f"{key}. {choice}\n" for key, choice in zip("ABCD", ex["CHOICES"])])
    prompt += "Answer:"
    completion = ex["ANSWER"]
    return {"prompt": prompt, "completion": completion}   

train_dataset = load_dataset("matteodagos/MNLP_M3_mcqa_dataset", split = "train").map(preprocess_train)
validation_dataset = load_dataset("matteodagos/MNLP_M3_mcqa_dataset",split = "validation").map(preprocess_validation)

# load Qwen3-Base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-0.6B-Base",
    max_seq_length = 2048,   # Context length - can be longer, but uses more memory
    load_in_8bit = False,
    load_in_4bit = False,
    full_finetuning = True,
)

print(train_dataset.shape)
print(validation_dataset.shape)
print(validation_dataset[0])



def compute_metrics(eval_ds):
   model.eval()
   correct = 0
   total   = len(eval_ds)

   with torch.no_grad():
        for ex in eval_ds:
            prompt = ex["prompt"].rstrip()      
            option_scores = []

            # pre-tokenise the prompt once
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False
            ).input_ids
            plen = len(prompt_ids)

            for j, opt in enumerate(ex["choices"]):
                # build the *letter + text* exactly as it will appear
                candidate     = f" {chr(65+j)}. {opt}"
                cand_ids      = tokenizer(
                    candidate,
                    add_special_tokens=False
                ).input_ids
                ids           = torch.tensor([prompt_ids + cand_ids],
                                             device=model.device)

                # mask the prompt part
                labels        = ids.clone()
                labels[:, :plen] = -100          # ignore the prompt tokens

                out           = model(input_ids=ids, labels=labels)
                # log-probability of the complete option =
                # – mean-NLL × #option tokens
                option_scores.append(-out.loss.item() * len(cand_ids))

            # letter with the highest log-prob
            pred_letter = chr(65 + int(np.argmax(option_scores)))
            if pred_letter == ex["answer"]:
                correct += 1
                
   print(correct, total)
   return {"mcq_accuracy": correct / total}

def preprocess_eval(ex):
    # build the same prompt as in training
    formatted_options = ""
    for i, opt in enumerate(ex['choices']):
        formatted_options += f"\n- {chr(65+i)}. {opt}"
    prompt = (
        f"Question:\n{ex['question']}\n"
        f"Choices: {formatted_options}\n"
        "Answer:"
    )

    correct = ex["choices"][ord(ex["answer"]) - ord("A")]
    target = f"{ex['answer']}. {correct}"

    return {
        "prompt": prompt,
        "completion": target,       
        "choices": ex["choices"],
        "answer":  ex["answer"],
    }
test_ds = load_dataset("matteodagos/M3_test",split="test")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Formatting function that works for rows and batches
def formatting_func(example):
    """
    Returns a *list[str]* no matter whether we get 1 example or many.
    """
    prompt   = example["prompt"]
    complete = example["completion"]

    # Batch mode
    if isinstance(prompt, list):
        return [p + " " + c for p, c in zip(prompt, complete)]

    # Row mode
    else:
        return [prompt + " " + complete]

print(compute_metrics(test_ds.map(preprocess_eval)))

training_args = SFTConfig(
    output_dir="./mcqa_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,      # eff. batch 64
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,
    num_train_epochs=2,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    gradient_checkpointing=True,
    seed=42,
    #push_to_hub=True,
    #hub_model_id="matteodagos/MNLP_M3_mcqa_model",
    #hub_strategy="end",
)

trainer = SFTTrainer(
    model           = model,
    tokenizer = tokenizer,
    args            = training_args,
    train_dataset   = train_dataset,         
    eval_dataset    = validation_dataset,
    formatting_func = formatting_func
    
)

trainer_stats = unsloth_train(trainer)
model.save_pretrained("./mcqa_model") 

print(compute_metrics(test_ds.map(preprocess_eval)))

#model.push_to_hub("matteodagos/MNLP_M3_mcqa_model")
#tokenizer.push_to_hub("matteodagos/MNLP_M3_mcqa_model") 

model.eval()

# Helper that scores each choice and returns the best one
def ask_mcq(question: str, choices: list[str]):
    """
    Parameters
    ----------
    question : the stem
    choices  : list with option texts, e.g. ["Paris", "London", "Berlin", "Rome"]

    Returns
    -------
    best_letter  : "A" / "B" / ...
    best_text    : the option text itself
    all_scores   : dict {letter: total-log-prob}
    """
    device = model.device

    # build the common prompt (identical for all options)
    formatted = "\n".join([f"- {chr(65+i)}. {opt}" for i, opt in enumerate(choices)])
    prompt    = f"Question:\n{question}\nChoices:\n{formatted}\nAnswer:"
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    plen       = len(prompt_ids)

    option_scores = []
    for i, opt in enumerate(choices):
        # " A. Paris"  (note leading space so the model sees a new token)
        candidate   = f" {chr(65+i)}. {opt}"
        cand_ids    = tokenizer(candidate, add_special_tokens=False).input_ids

        ids    = torch.tensor([prompt_ids + cand_ids], device=device)
        labels = ids.clone()
        labels[:, :plen] = -100          # ignore prompt tokens in the loss

        with torch.no_grad():
            out   = model(input_ids=ids, labels=labels)
        # total log-probability of the option
        option_scores.append(-out.loss.item() * len(cand_ids))

    best_idx     = int(np.argmax(option_scores))
    letters      = [chr(65+i) for i in range(len(choices))]
    scores_dict  = dict(zip(letters, option_scores))

    return letters[best_idx], choices[best_idx], scores_dict


question = "Suppose you try to perform a binary search on a 5-element array sorted in the reverse order of what the binary search algorithm expects. How many of the items in this array will be found if they are searched for?"
choices  = ["5", "0", "1", "2"]

letter, answer, all_scores = ask_mcq(question, choices)

print("Model prediction:", f"{letter}. {answer}")
print("Option scores (higher = more likely):")
for k, v in all_scores.items():
    print(f"  {k}: {v:.2f}")

