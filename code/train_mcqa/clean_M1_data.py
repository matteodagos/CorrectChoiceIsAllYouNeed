import json
from credentials import API_KEY, API_BASE
import gpt_wrapper
from gpt_wrapper.chat import Chat
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Value, Sequence

with open('m1_preference_data.json', 'r') as f:
    data = json.load(f)
print(f"Loaded {len(data)} questions")
open_questions = []

for question in data:
    if question['question_type'] == 'open_answer':
        open_questions.append(question)

print(f"There are {len(open_questions)} open-answer questions")

golden_answers = []
ids = []
questions = []
rationales = []

for question in tqdm(open_questions):

   complete_answer = question['question_answer']
   body = question['question_body']
   id = f"{question['course_id']}_{question['question_id']}"
   
   ids.append(id)
   questions.append(body)
   golden_answers.append(complete_answer)
   
   rationale_candidates = []
   for preference in question['preferences']:
      preferred = preference['ranking_criteria']['overall']
      if len(preferred) > 1:
         continue
      candidate = preference[preferred]
      rationale_candidates.append(candidate)
   
   content = f"Given the question: '{body}' "
   chat_for_rationale = Chat.create(name=f"Asking for rationale, {id}")
   instruction = "You are an expert professor who needs to gain all the \
                  answers from a group of students and try to formulate the \
                  final definitive rationale behind the answer to a question"
   content += f"By knowing the actual answer is {complete_answer} \n\
                              and given the following rationales behind:\n"
   for idx, rationale in enumerate(rationale_candidates[:10]):
      content += f"<START RATIONALE #{idx}>. {rationale} <END RATIONALE>\n"
   content += "Provide me with the summarizing, quality rationale by keeping the best from the best ones. \
      Avoid mentioning that it's a summarized version of given rationales, just go straight to the point. It should look as a direct answer to the given question."
   rationale = str(chat_for_rationale.ask(content=content, instruction=instruction))
   rationales.append(rationale)

   if len(rationales) % 20 == 0:
      print(Chat.budget())

df = pd.DataFrame({
   "id": ids,
   "question": questions,
   "answer": golden_answers,
   "rationale": rationales
})

dataset = df.to_dict(orient="records")

with open('openqs_m1_cleaned_dataset.json', 'w') as fout:
   json.dump(dataset, fout, indent=4)

with open("openqs_m1_cleaned_dataset.json", "r") as f:
   dataset1 = json.load(f)

print(len(dataset1))

train_indexes = np.random.choice(a=range(len(dataset)), size=int(0.9 * len(dataset)), replace=False)
validation_indexes = set(range(len(dataset))) - set(train_indexes)


train_set = [dataset[i] for i in train_indexes]
validation_set = [dataset[i] for i in validation_indexes]



# 1) Define the features (schema) of your table, marking "choices" as a Sequence of strings:
features = Features({
    "id":       Value("string"),
    "question": Value("string"),
    "answer":   Value("string"),
    "rationale": Value("string"),
})

# 2) Build the dataset with that schema:
ds = DatasetDict({
    "train":      Dataset.from_list(train_set,      features=features),
    "validation": Dataset.from_list(validation_set, features=features),
})

# 3) Now export to Parquet — the nested list type will be preserved:
ds["train"].to_parquet("EPFL_OPENQs/data/train-00000-of-00001.parquet")
ds["validation"].to_parquet("EPFL_OPENQs/data/validation-00000-of-00001.parquet")