
from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import login
import random, hashlib, json, os

allowed_mmlu =  ["abstract_algebra", "anatomy", "astronomy",
               "clinical_knowledge", "college_biology", "college_chemistry", 
               "college_computer_science", "college_mathematics", "college_physics", 
               "computer_security", "conceptual_physics", "electrical_engineering", 
               "formal_logic", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
               "high_school_mathematics", "high_school_physics",
               "machine_learning", "medical_genetics"]

def make_uid(ds_name, idx):
    return f"{ds_name}_{idx:07d}"         

def map_sciq(row):
    choices  = [row["correct_answer"],
                row["distractor1"],
                row["distractor2"],
                row["distractor3"]]
    random.shuffle(choices)            
    correct_choice = choices.index(row["correct_answer"])
    answer = f"{chr(65+correct_choice)}. " + choices[correct_choice]
    question = row["question"]
    return {"QUESTION": question, "ANSWER": answer, "CHOICES": choices, "RATIONALE": row["support"]}

def map_scienceqa(row):
    choices  = row["choices"]
    correct_choice = row["answer"]
    answer = f"{chr(65+correct_choice)}. " + choices[correct_choice]
    return {"QUESTION": row["lecture"] + "\n" + row["question"], "ANSWER": answer, "CHOICES": choices, "RATIONALE": row["solution"]}

def map_arc(row):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = [c for c in row["choices"]["text"]]
    letters = [c for c in row["choices"]["label"]]
    idx        = letters.index(row["answerKey"])
    letters = [l if l in alphabet else alphabet[i-1] for i, l in enumerate(letters)]
    letter     = row["answerKey"]
    letter = letter if letter in alphabet else alphabet[int(letter) - 1]
    answer = f"{letter}. {choices[idx]}"
    
    return {"QUESTION": row["question"], "ANSWER": answer, "CHOICES": choices, "RATIONALE": ""}

def map_aqua(row):
    choices  = [elem[2:] for elem in row["options"]]
    letters  = list("ABCDE")[:len(choices)]
    letter     = row["correct"] 
    idx        = letters.index(letter)
    answer = f"{letter}. {choices[idx]}"
    return {"QUESTION": row["question"], "ANSWER": answer, "CHOICES": choices, "RATIONALE": row["rationale"]}

def map_epfl_mcqs(row):
    choices = row["choices"]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    correct_choice = alphabet.index(row["answer"])
    answer = f"{chr(65+correct_choice)}. " + choices[correct_choice]
    return {"QUESTION": row["question"], "ANSWER": answer, "CHOICES": choices, "RATIONALE": row["rationale"]}

def map_mmlu(row):
    completion = row["output"]
    answer = completion[:completion.find("##EXPLANATION")]
    rationale = completion[completion.find("##EXPLANATION") + len("##EXPLANATION"):].strip()
    return {"QUESTION": row["instruction"], "ANSWER": answer, "CHOICES": [""], "RATIONALE": rationale}

def map_medmcqa(row):
    choices = []
    choices.append(row["opa"])
    choices.append(row["opb"])
    choices.append(row["opc"])
    choices.append(row["opd"])

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    correct_choice = row["cop"]
    answer = f"{alphabet[correct_choice]}. " + choices[correct_choice]
    return {"QUESTION": row["question"], "ANSWER": answer, "CHOICES": choices, "RATIONALE": row["exp"]}


SOURCES = [
    ("sciq",               "allenai/sciq",                   None,          map_sciq),
    ("scienceqa_textonly",          "tasksource/ScienceQA_text_only",         None,          map_scienceqa),
    ("arc_easy",           "allenai/ai2_arc",                        "ARC-Easy",        map_arc),
    ("arc_challenge",      "allenai/ai2_arc",                        "ARC-Challenge",   map_arc),
    ("aqua_rat",           "deepmind/aqua_rat",              "raw",          map_aqua),
    ("epfl_mcqs",          "matteodagos/EPFL_MCQs",          None,       map_epfl_mcqs),
    ("medmcqa",         "openlifescienceai/medmcqa",                None,          map_medmcqa),
    ("mmlu",               "bziemba/MNLP_M3_quantized_dataset",           None,      map_mmlu),
]

all_ds = []
for short_name, hf_id, config, fn in SOURCES:

    print(f"Loading {short_name} …")
    ds = load_dataset(hf_id, config, split="train")
    if short_name == "scienceqa_textonly":
        ds = ds.filter(lambda ex: ex["subject"] == "natural science" and ex["task"] == "closed choice")
    elif short_name == "aqua_rat":
        ds = ds.shuffle()  # shuffle the order of questions
        ds = ds.select(range(20000))  # only keep first 20000 rows
    elif short_name == "mmlu":
        ds = ds.filter(lambda ex: ex["dataset"] == "mmlu")
    elif short_name == "medmcqa":
        ds = ds.filter(lambda ex: ex["choice_type"] == "single")
        ds = ds.shuffle()  # shuffle the order of questions
        ds = ds.select(range(10000))  # only keep first 10000 rows

    ds = ds.map(fn, remove_columns=ds.column_names)
    ds = ds.map(
        lambda ex, idx: {
            "dataset": short_name,
            "id": make_uid(short_name, idx),
            "QUESTION": ex["QUESTION"],
            "ANSWER": ex["ANSWER"],
            "CHOICES": ex["CHOICES"],
            "RATIONALE": ex["RATIONALE"],
        },
        with_indices=True,
    )
    ds = ds.filter(lambda ex: len(ex["QUESTION"]) == 4)
    all_ds.append(ds)

mix = concatenate_datasets(all_ds).shuffle(seed=42)
print("Final mixture size:", len(mix))
print(mix[0])


REPO_NAME = "matteodagos/MNLP_M3_mcqa_dataset" 

mix.push_to_hub(
    REPO_NAME,
    private=False, 
    split = "train",                      
)
print("✓ Dataset pushed – view at https://huggingface.co/datasets/" + REPO_NAME)
