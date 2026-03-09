The requirements.txt file lists all the Python libraries that the code depends on. On terminal, please run the following to install requirements.txt packages and some additional packages:

```
pip install -r requirements.txt
pip install transformers==4.51.3
pip install llmcompressor
```
---
## For getting the sajal09/MNLP_M3_quantized_model, please run:
```
python train.py --train 1
```
This model has been trained/calibrated on matteodagos/MNLP_M3_mcqa_model with the calibration dataset sajal09/MNLP_M3_quantized_dataset

---
## For getting the sajal09/MNLP_M3_quantized_dataset, please run:
```
python data_creation.py --traindata_only 1
```
The MNLP_M3_quantized_dataset consists of 64 C4 en sentences, 32 Epfl MQCAs and 32 Non-Epfl MCQAs.

---

## For Hyperparameter tuning: 
Hyperparameter tuning was done on matteodagos/MNLP_M2_mcqa_model model and the best configuration was used to quantize matteodagos/MNLP_M3_mcqa_model. You can create the models relevant to different hyperparameter settings by running the scripts below. For evaluation, you can use the lighteval-epfl-mnlp suite on the dataset sajal09/MNLP_Matteo_Val_Split (Validation split of MCQA data in the lighteval-epfl-mnlp format).

### On different mixtures of calibration datasets, please run:
```
python data_creation.py --traindata_only 0
python train.py --calib_data_tuning 1
```
The calibration datasets used are: 
* calib_64_32: 64 C4 en sentences, 32 Epfl MQCAs
* calib_128_64: 128 C4 en sentences, 64 Epfl MQCAs
* calib_256_128: 256 C4 en sentences, 128 Epfl MQCAs
* calib_512_256: 512 C4 en sentences, 256 Epfl MQCAs
* calib_64_32_32: 64 C4 en sentences, 32 Epfl MQCAs, 32 Non-Epfl MCQAs
---
### On different smoothing strengths, please run:
```
python train.py --smoothing_strength_tuning 1
```
The smoothing strengths used are: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

---
### On different recipes, please run:
```
python train.py --recipe_tuning 1
```
The recipes used are:
* No Smoothing, W4_Int_Symmetric_Channel_A8_Int_Symmetric_Token
* Smoothing strength 0.7, W4_Int_Symmetric_Channel_A8_Int_Symmetric_Token
* Smoothing strength 0.7, W4_Int_ASymmetric_Channel_A8_Int_Symmetric_Token
* Smoothing strength 0.7, W4_Int_Symmetric_Group128_A8_Int_Symmetric_Token
* Smoothing strength 0.7, W4_Int_Symmetric_Group256_A8_Int_Symmetric_Token
* Smoothing strength 0.7, W4_Int_Symmetric_Tensor_A8_Int_Symmetric_Token
* Smoothing strength 0.7, W8_Int_Symmetric_Channel_A8_Int_Symmetric_Token
