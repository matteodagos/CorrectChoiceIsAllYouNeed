
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, Sequence
from huggingface_hub import login
import random, hashlib, json, os

allowed_mmlu =  ["abstract_algebra", "anatomy", "astronomy",
               "clinical_knowledge", "college_biology", "college_chemistry", 
               "college_computer_science", "college_mathematics", "college_physics", 
               "computer_security", "conceptual_physics", "electrical_engineering", 
               "formal_logic", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
               "high_school_mathematics", "high_school_physics",
               "machine_learning", "medical_genetics", "nutrition"]


def make_uid(ds_name, idx):
    return f"{ds_name}_{idx:07d}"      

def map_arc(row):
    # ARC has list-of-dicts choices
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = [c for c in row["choices"]["text"]]
    question = row['question']
    letter     = row["answerKey"]
    letter = letter if letter in alphabet else alphabet[int(letter) - 1]
    return {"question1": question, "choices1": choices, "answer1": str(letter)}

def map_epfl_mcqs(row):
    question = row["question"]
    choices = row["choices"]
    answer = row["answer"]
    return {"question1": question, "choices1": choices, "answer1": str(answer)}

def map_mmlu(row):
    choices = row["choices"]
    alphabet = "ABCD"
    correct_choice = row["answer"]
    answer = alphabet[correct_choice]
    row["answer"] = answer
    question = row["question"]
    return {"question1": question, "choices1": choices, "answer1": str(answer)}



SOURCES = [
    ("arc_easy",           "allenai/ai2_arc",                        "ARC-Easy",        map_arc),
    ("arc_challenge",      "allenai/ai2_arc",                        "ARC-Challenge",   map_arc),
    ("epfl_mcqs",          "matteodagos/EPFL_MCQs",          None,       map_epfl_mcqs),
    ("mmlu",               "cais/mmlu",                    "all",          map_mmlu),
]

all_ds = []
for short_name, hf_id, config, fn in SOURCES:

    print(f"Loading {short_name} …")
    ds = load_dataset(hf_id, config, split="test")
    if short_name == "mmlu":
        ds = ds.filter(lambda ex: ex["subject"] in allowed_mmlu)
        ds = ds.shuffle() 
        ds = ds.select(range(1000)) 
    elif short_name == "epfl_mcqs":
        ds = ds.filter(lambda ex: len(ex["choices"]) == 4)
    elif short_name == "arc_easy":
        ds = ds.shuffle()
        ds = ds.select(range(500)) 
    elif short_name == "arc_challenge":
        ds = ds.shuffle()
        ds = ds.select(range(1000))
    ds = ds.map(fn, remove_columns=ds.column_names)
    ds = ds.map(
        lambda ex, idx: {
            "dataset": short_name,
            "id": make_uid(short_name, idx),
            "question": ex["question1"],
            "choices": ex["choices1"],
            "answer": ex["answer1"],
        },
        with_indices=True,

    )
    ds = ds.remove_columns(["answer1", "choices1", "question1"])
    ds = ds.filter(lambda ex: len(ex["choices"]) == 4)
    all_ds.append(ds)
mix = concatenate_datasets(all_ds).shuffle(seed=42)
print("Final mixture size:", len(mix))
print(mix[0])


REPO_NAME = "matteodagos/M3_test" 

mix.push_to_hub(
    REPO_NAME,
    private=False,   
    split = "test",           
)
print("✓ Dataset pushed – view at https://huggingface.co/datasets/" + REPO_NAME)