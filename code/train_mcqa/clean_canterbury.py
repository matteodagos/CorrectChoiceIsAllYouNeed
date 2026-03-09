import pandas as pd
from datasets import Features, Value, Sequence, Dataset
import random

random.seed(42)

df = pd.read_csv("canterbury_questions.csv")
print(df.head())
print(df.columns)

schema = Features({
      "Type": Value("string"),
      "ID": Value("string"),
      "Points": Value("float32"),
      "Question": Value("string"),
      "Correct Answer": Value("string"),
      "Choice 1": Value("string"),
      "Choice 2": Value("string"),
      "Choice 3": Value("string"),
      "Choice 4": Value("string"),
      "Choice 5": Value("string"),
      "Choice 6": Value("string"),
      "Choice 7": Value("string"),
      "Choice 8": Value("string"),
      "Choice 9": Value("string"),
      "Choice 10": Value("string"),
      "Explanation": Value("string"),
})

ds = Dataset.from_pandas(df, features = schema).filter(lambda ex: ex["Type"] == "MC")
print(ds)

def map_cs(row):
      question = row["Question"]
      question = question.replace("<p>", " ").strip()
      question = question.replace("</p>", " ").strip()
      choices = []
      answer = row["Correct Answer"]
      for i in range(1,11):
            if row["Choice " + str(i)] is not None:
                  choice = row["Choice " + str(i)]
                  choice = choice.replace("<p>", " ").strip()
                  choice = choice.replace("</p>", " ").strip()
                  choices.append(choice)
            else:
                  break
      right_choice = choices[ord(answer) - ord('A')]
      if ord(answer) > ord('D'):
            choices[0] = right_choice
            choices = choices[:4]
            random.shuffle(choices)
            answer = chr(ord('A') + choices.index(right_choice))
      else:
            choices = choices[:4]

      answer = answer + ". " + choices[ord(answer) - ord('A')]
      rationale = row["Explanation"]
      rationale = rationale.replace("<p>", " ").strip()
      rationale = rationale.replace("</p>", " ").strip()
      return {"question": question, "choices": choices, "answer": answer, "rationale": rationale}

ds = ds.filter(lambda ex: "img" not in ex["Question"])
ds = ds.filter(lambda ex: ex["Correct Answer"] is not None)
ds = ds.map(map_cs, remove_columns=ds.column_names)
ds = ds.filter(lambda ex: len(ex["choices"]) == 4)
print(ds)

def make_uid(ds_name, idx):
    return f"{ds_name}_{idx:07d}"  
    
ds = ds.map(
        lambda ex, idx: {
            "dataset": "canterbury_cs",
            "id": make_uid("canterbury_cs", idx),
            "question": ex["question"],
            "choices": ex["choices"],
            "answer": ex["answer"],
            "rationale": ex["rationale"],
        },
        with_indices=True,
    )
ds.shuffle(seed=42)
print(ds[0])


REPO_NAME = "matteodagos/M3_canterbury_code"

ds.push_to_hub(
    REPO_NAME,
    private=False,
    split="test",
)

print("✓ Dataset pushed – view at https://huggingface.co/datasets/" + REPO_NAME)