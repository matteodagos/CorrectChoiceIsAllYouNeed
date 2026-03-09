cd train_quantized

# Install the required packages
pip install -r requirements.txt
pip install transformers==4.51.3
pip install llmcompressor

# For training the model with the chosen caibration dataset to get sajal09/MNLP_M3_quantized_model, run:
python train.py --train 1

# For generating the calibration dataset to get sajal09/MNLP_M3_quantized_dataset, run:
python data_creation.py --traindata_only 1

# For generating the different mixtures of calibration datasets for hyperparameter tuning, run:
python data_creation.py --traindata_only 0

# Hyperparameter tuning: For generating models calibrated on different mixtures of calibration datasets, run:
# Note: Please generate the different mixtures of calibration datasets before running this command.
python train.py --calib_data_tuning 1

# Hyperparameter tuning: For generating models calibrated on different smoothing strengths, run:
python train.py --smoothing_strength_tuning 1

# Hyperparameter tuning: For generating models calibrated on different recipes, run:
python train.py --recipe_tuning 1