import datasets
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
import argparse
from unittest import mock
from compressed_tensors import CompressionFormat

# Max length of the sentences allowed.
MAX_SEQUENCE_LENGTH = 2048

def load_model_and_tokenizer(model_id):
    # Load the model and tokenizer from huggingface by model_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def tokenize(sample, tokenizer):
    # Tokenize the text with no padding and truncate if greater than MAX_SEQUENCE_LENGTH
    return tokenizer(sample["text"], 
                     padding=False, 
                     max_length=MAX_SEQUENCE_LENGTH, 
                     truncation=True, 
                     add_special_tokens=False)
    
def load_calibration_dataset(ds_name, tokenizer, from_disk = False):
    # Load from local disk if from_disk is True. 
    if from_disk:
        ds = load_from_disk(ds_name)
        ds = ds['train']
    else:
        ds = load_dataset(ds_name, split = 'train')

    # Set a seed for reproductibility
    ds = ds.shuffle(seed=42)
    # Tokenize the dataset using the tokenizer
    ds = ds.map(lambda x: tokenize(x, tokenizer), remove_columns=ds.column_names)
    return ds

def train(model_id, dataset_id, recipe, compression_format, qmodel_name, from_disk = False):
    # The function where calibration of model takes place with the calibration dataset.

    # Load the model and tokenizer.
    model, tokenizer = load_model_and_tokenizer(model_id)

    # Load and tokenize the calibration dataset using the tokenizer.
    ds_calibration = load_calibration_dataset(dataset_id, tokenizer, from_disk)

    # The pack quantization of int4 weights into int32 is not currently supported in llmcompressor library when the activation are 8 bits,
    # however I have raised a request and they have acknowledged it:
    # https://github.com/vllm-project/llm-compressor/issues/1506#issuecomment-2936154248
    # Until then, i am using 'mock' to override the default infer_quantization_format function to enforce pack quantization.
    with mock.patch("llmcompressor.transformers.sparsification.compressed_tensors_utils.infer_quantization_format") as mock_infer:
        mock_infer.return_value = compression_format

        # Run the calibration loop and store the model at qmodel_name.
        oneshot(
                model=model,
                dataset=ds_calibration,
                recipe=recipe,
                max_seq_length=MAX_SEQUENCE_LENGTH,
                num_calibration_samples=len(ds_calibration),
                output_dir = qmodel_name,
                save_compressed=True,
                )

    # Load the compressed model and its tokenizer from qmodel_name
    qmodel = AutoModelForCausalLM.from_pretrained(
                qmodel_name,
                torch_dtype = 'auto',
                device_map = 'auto',
                local_files_only=True
            )
    qtokenizer = AutoTokenizer.from_pretrained(qmodel_name)

    # Check the compressed model's size.
    print(qmodel_name, 'saved to memory with size = {:.3f}MB'.format(model_size(qmodel)) )

def model_size(model):
    # Calculates the size of the model in MB.
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
    
def recipe_and_compression_generator(sm_strength, weight_bits, act_bits, weight_strategy, weight_symmetric, weight_block_size):
    # Returns the quantization recipe and compressed format to be used.
    # For Activations, we are always using 8 bits dynamic int symmetric token recipe.

    # Since all our recipes are INT quantized, we use the CompressionFormat.int_quantized by default.
    compression_format = CompressionFormat.int_quantized
    if weight_bits == 4 and weight_symmetric:
        # If the weights are going to be 4 int bits, then we can pack them into int32 datatype.
        # However, llm-compressor does not suppport packing of 4 bits int weights if they are quantized Asymmetric.
        compression_format = CompressionFormat.pack_quantized

    # For QuantizationStrategy.GROUP, we have to provide the weight_block_size.
    if weight_strategy == QuantizationStrategy.GROUP:
        gptq_recipe = GPTQModifier(
            ignore=[
                "lm_head",
            ],
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                    num_bits=weight_bits, # Quantized bits of weights
                    type=QuantizationType.INT, # Quantized dtype of weights
                    strategy=weight_strategy, # Strategy of quantization: GROUP or CHANNEL or TENSOR
                    group_size=weight_block_size, # Size of Blocks of weights to be quantized.
                    symmetric=weight_symmetric, # Symmetric or Asymmetric Quantization
                    dynamic=False, # Static or Dynamic Quantization of weights
                ),
                input_activations=QuantizationArgs(
                    num_bits=act_bits, # Quantized bits of activations
                    type=QuantizationType.INT, # Quantized dtype of activations
                    strategy=QuantizationStrategy.TOKEN, # Strategy of quantization: TOKEN is supported by most hardwares
                    symmetric=True, # Symmetric or Asymmetric Quantization
                    dynamic=True, # Static or Dynamic Quantization of activations
                    observer=None,
                ),
            )
            }
        )
    else:
        gptq_recipe = GPTQModifier(
            ignore=[
                "lm_head",
            ],
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                    num_bits=weight_bits,
                    type=QuantizationType.INT,
                    strategy=weight_strategy,
                    symmetric=weight_symmetric,
                    dynamic=False,
                ),
                input_activations=QuantizationArgs(
                    num_bits=act_bits,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.TOKEN,
                    symmetric=True,
                    dynamic=True,
                    observer=None,
                ),
            )
            }
        )

    # Whether to add Smoothing or not.
    if sm_strength:
        smoothing_recipe = SmoothQuantModifier(smoothing_strength=sm_strength)
        return ([smoothing_recipe, gptq_recipe], compression_format)

    return ([gptq_recipe], compression_format)

def main(args):

    if args.train:
        print('Starting the training process')

        # The best M3 MCQA model used.
        MODEL_ID = "matteodagos/MNLP_M3_mcqa_model"

        # The best calibration data based on Hyperparameter tuning is used.
        DATASET_ID = "sajal09/MNLP_M3_quantized_dataset"

        # Train Recipe used here is W4 Int Channel Static Symmetric A8 Int Token Dynamic Symmetric
        train_recipe, compression_format = recipe_and_compression_generator(0.7, 4, 8, QuantizationStrategy.CHANNEL, True, None) 
        
        train(MODEL_ID, 
             DATASET_ID, 
             train_recipe,
             compression_format,
             'MNLP_M3_quantized_model', 
             False)

    elif args.calib_data_tuning:
        print('Starting the Calibration Data Mixtues Hyperparameter Tuning')

        # We perform the Hyperparameter tuning on MNLP_M2_mcqa_model
        MODEL_ID = "matteodagos/MNLP_M2_mcqa_model"

        # Different calibration datasets to be used for deciding the best one.
        calib_datasets = ['calib_64_32', 
                          'calib_128_64',
                         'calib_256_128',
                         'calib_512_256',
                         'calib_64_32_32']

        # The recipe and compression_format is held same.
        recipe, compression_format = recipe_and_compression_generator(0.7, 4, 8, QuantizationStrategy.CHANNEL, True, None) 

        # Save the compressed model by their respective names.
        model_tokenizer_dict = {}
        for calib_dataset in calib_datasets:
            model_tokenizer_dict[calib_dataset] = train(MODEL_ID, 
                                                        calib_dataset,
                                                        recipe,
                                                        compression_format,
                                                        calib_dataset+'_quantized_model',
                                                        True)


    elif args.smoothing_strength_tuning:
        print('Starting the Smoothing Strengths Hyperparameter Tuning')

        # We perform the Hyperparameter tuning on MNLP_M2_mcqa_model 
        MODEL_ID = "matteodagos/MNLP_M2_mcqa_model"

        # The best calibration data based on Hyperparameter tuning is used. 
        DATASET_ID = "sajal09/MNLP_M3_quantized_dataset"

        # Different Smoothing strengths to be used for deciding the best one.
        smoothing_strengths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Save the compressed model by their respective names.
        model_tokenizer_dict = {}
        for s_strength in smoothing_strengths:
            recipe, compression_format = recipe_and_compression_generator(s_strength, 4, 8, QuantizationStrategy.CHANNEL, True, None)

            model_tokenizer_dict[str(s_strength)] = train(MODEL_ID, 
                                                            DATASET_ID,
                                                            recipe,
                                                            compression_format,
                                                            str(s_strength)+'_quantized_model',
                                                            False)


    elif args.recipe_tuning:
        print('Starting the Recipes Hyperparameter Tuning')

        # We perform the Hyperparameter tuning on MNLP_M2_mcqa_model 
        MODEL_ID = "matteodagos/MNLP_M2_mcqa_model"

        # The best calibration data based on Hyperparameter tuning is used. 
        DATASET_ID = "sajal09/MNLP_M3_quantized_dataset"

        # Different recipes to be used for deciding the best one.
        recipes = {
                    'WithoutSmoothing': recipe_and_compression_generator(0, 4, 8, QuantizationStrategy.CHANNEL, True, None),
                    'W4_Channel_Symmetric_A8_Token_Symmetric': recipe_and_compression_generator(0.7, 4, 8, QuantizationStrategy.CHANNEL, True, None),
                    'W4_Channel_Asymmetric_A8_Token_Symmetric': recipe_and_compression_generator(0.7, 4, 8, QuantizationStrategy.CHANNEL, False, None),
                    'W4_Group128_Symmetric_A8_Token_Symmetric': recipe_and_compression_generator(0.7, 4, 8, QuantizationStrategy.GROUP, True, 128),
                    'W4_Group256_Symmetric_A8_Token_Symmetric': recipe_and_compression_generator(0.7, 4, 8, QuantizationStrategy.GROUP, True, 256),
                    'W4_Tensor_Symmetric_A8_Token_Symmetric': recipe_and_compression_generator(0.7, 4, 8, QuantizationStrategy.TENSOR, True, None),
                    'W8_Channel_Symmetric_A8_Token_Symmetric': recipe_and_compression_generator(0.7, 8, 8, QuantizationStrategy.CHANNEL, True, None)
        }

        # Save the compressed model by their respective names.
        model_tokenizer_dict = {}
        for recipe_name in recipes:
            model_tokenizer_dict[recipe_name] = train(MODEL_ID, 
                                                            DATASET_ID,
                                                            recipes[recipe_name][0], 
                                                            recipes[recipe_name][1],
                                                            str(recipe_name)+'_quantized_model',
                                                            False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training the model')
    parser.add_argument('--train', type=int, default=0) # Whether you want to train your model or do Hyperparameter tuning.
    parser.add_argument('--calib_data_tuning', type=int, default=0) # Whether you want to do Hyperparameter tuning on calibration data mixtures.
    parser.add_argument('--smoothing_strength_tuning', type=int, default=0) # Whether you want to do Hyperparameter tuning on smoothing strengths.
    parser.add_argument('--recipe_tuning', type=int, default=0) # Whether you want to do Hyperparameter tuning on different recipes.
    
    args = parser.parse_args()
    main(args)
    
    
