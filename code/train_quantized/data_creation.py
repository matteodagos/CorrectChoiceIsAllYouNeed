import datasets
from datasets import load_dataset
import random
import argparse


# Max length of the sentences allowed.
MAX_SEQUENCE_LENGTH = 2048

def main(args):
    
    # For reproductibility
    random.seed(10)

    print('Creating the three datasets')
    c4_sentences = calibdata_c4(512) # The C4 en dataset with general english sentences
    epfl_mcqa = calibdata_mcqa_epfl_train(256) # The epfl m2 mcqa dataset
    nonepfl_mcqa = calibdata_mcqa_nonepfl_train(32) # The non-epfl mcqa dataset

    # Calibration Data Mixtures For Hyperparameter Tuning
    if not args.traindata_only:
        print('Creating all the different calibration data mixtures')
        
        # 64 C4 + 32 EPFL MCQA
        calib_64_32 = c4_sentences[:64] + epfl_mcqa[:32]
        save_calib_data(calib_64_32, 'calib_64_32')
    
        # 128 C4 + 64 EPFL MCQA
        calib_128_64 = c4_sentences[:128] + epfl_mcqa[:64]
        save_calib_data(calib_128_64, 'calib_128_64')
        
        # 256 C4 + 128 EPFL MCQA
        calib_256_128 = c4_sentences[:256] + epfl_mcqa[:128]
        save_calib_data(calib_256_128, 'calib_256_128')
    
        # 512 C4 + 256 EPFL MCQA
        calib_512_256 = c4_sentences[:512] + epfl_mcqa[:256]
        save_calib_data(calib_512_256, 'calib_512_256')

    # This is the final dataset used for calibration.
    # 64 C4 + 32 EPFL MCQA + 32 Non-EPFL MCQA => Best MCQA Achieved on it.
    calib_64_32_32 = c4_sentences[:64] + epfl_mcqa[:32] + nonepfl_mcqa[:32]
    save_calib_data(calib_64_32_32, 'calib_64_32_32')
    

def save_calib_data(datalist, dsname):
    # Save the calibration data into disk.
    print('Saving the ', dsname, ' to disk')
    dataset = datasets.DatasetDict({
                "train": datasets.Dataset.from_list(datalist)
                })
    dataset.save_to_disk(dsname)
    
    
def calibdata_c4(number_of_samples):
    # Load the c4 en data in streaming mode.
    dataset_name = 'allenai/c4'
    c4 = load_dataset(dataset_name, "en", split="train", streaming=True)

    # Load the first 30000 samples from it.
    c4 = c4.take(30000)

    # Convert it into a list and only keep 'text' and 'dataset' field.
    c4list = []
    for example in c4:
        if len(example['text']) < MAX_SEQUENCE_LENGTH:
            c4list.append({'text':example['text'], 'dataset':'allenai/c4 en train'})

    # Sample the number_of_samples examples from the final list
    calib_data = random.sample(c4list, number_of_samples)

    return calib_data
    
def calibdata_mcqa_nonepfl_train(number_of_samples):
    # Load the MCQA Training Data which does not have any epfl questions.
    ds_mcqa_nonepfl_train = load_dataset('sajal09/MNLP_MCQA_NonEPFL_Train_Data', split = 'train')

    # Format the dataset into a list of format (prompt + '' + completion, dataset)
    ds_mcqa_nonepfl_train = formatting_func(ds_mcqa_nonepfl_train)
    
    # Filter sentences greater than MAX_SEQUENCE_LENGTH
    pruned_list = []
    for i in ds_mcqa_nonepfl_train:
        if len(i[0]) < MAX_SEQUENCE_LENGTH:
            pruned_list.append({'text':i[0], 'dataset':i[1]})

    # Sample the number_of_samples examples from the final list
    calib_data = random.sample(pruned_list, number_of_samples)
    
    return calib_data

def calibdata_mcqa_epfl_train(number_of_samples):
    # Load the MCQA Training Data which only has the epfl questions.
    ds_mcqa_epfl_train = load_dataset('sajal09/MNLP_MCQA_EPFL_Train_Data', split = 'train')

    # Format the dataset into 'prompt', 'completion', 'dataset'
    ds_mcqa_epfl_train = ds_mcqa_epfl_train.map(preprocess_mcqa_epfl_train)

    # Format the dataset into a list of format (prompt + '' + completion, dataset)
    ds_mcqa_epfl_train = formatting_func(ds_mcqa_epfl_train)
    
    # Filter sentences greater than MAX_SEQUENCE_LENGTH
    pruned_list = []
    for i in ds_mcqa_epfl_train:
        if len(i[0]) < MAX_SEQUENCE_LENGTH:
            pruned_list.append({'text':i[0], 'dataset':i[1]})

    # Sample the number_of_samples examples from the final list
    calib_data = random.sample(pruned_list, number_of_samples)
    
    return calib_data

def formatting_func(example):
    # Returns a list of (prompt + '' + completion, dataset)
    prompt   = example["prompt"]
    complete = example["completion"]
    dataset = example["dataset"]

    # ── Batch mode ─────────────────────────────────────────────────
    if isinstance(prompt, list):
        return [(p + " " + c, d) for p, c, d in zip(prompt, complete, dataset)]

    # ── Row mode ───────────────────────────────────────────────────
    else:
        return [(prompt + " " + complete, dataset)]

def preprocess_mcqa_epfl_train(ex):
    # Format dataset into 'prompt', 'completion', 'dataset'
    
    formatted_options = ""
    for i, opt in enumerate(ex['choices']):
        formatted_options += f"\n- {chr(65+i)}. {opt}"
    prompt = (
        f"Question:\n{ex['question']}\n"
        f"Choices: {formatted_options}\n"
        "Answer:"
    )
    correct = ex["choices"][ord(ex["answer"]) - ord("A")]
    # append rationale
    target = f"{ex['answer']}. {correct} .Rationale: {ex['rationale']}"

    return {"prompt": prompt, "completion": target, "dataset": "epfl_m2_dataset"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data creation')
    parser.add_argument('--traindata_only', type=int, default=1) # Whether you want the calibration data used for the M3 model only or you want all the mixtures of the Calibration data used for Hyperparameter tuning.
    
    args = parser.parse_args()
    main(args)
