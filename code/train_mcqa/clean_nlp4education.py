import pandas as pd
from datasets import Features, Value, Sequence, Dataset
import random
import json

selected = []
with open("EPFL_questions.json", "r") as f:
      questions = json.load(f)
      for question in questions:
            if question["question_type"] == "mcq" and question["multiple_correct_answers"] == 0.0:
                  selected.append(question)
print(len(selected))

df = pd.read_json("EPFL_questions.json")
print(df.iloc[20])

ds = Dataset.from_list(selected)
ds = ds.filter(lambda ex: ex["question_type"] == "mcq" and len(ex["question_options"]) == 4)
print(ds)

def make_uid(ds_name, idx):
    return f"{ds_name}_{idx:07d}"  

ds = ds.map(
        lambda ex, idx: {
            "dataset": "nlp4education",
            "id": make_uid("nlp4education", idx),
            "question": ex["question_body"],
            "choices": ex["question_options"],
            "answer": chr(ex["mcq_answer_index"][0] + 65),
        },
        with_indices=True,
        remove_columns=ds.column_names
    )

REPO_NAME = "matteodagos/M3_nlp4education"

ds.push_to_hub(
    REPO_NAME,
    private=False,
    split="test",
)

print("✓ Dataset pushed – view at https://huggingface.co/datasets/" + REPO_NAME)