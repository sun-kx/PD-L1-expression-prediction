# Multiple Instance Learning-based Prediction of PD-L1 Expression from H&E-stained Histopathological Images in Breast Cancer
This project is based on the Multiple Instance Learning (MIL) method and predicts the expression of the PD-L1 gene by analyzing H&E (hematoxylin-eosin) stained slice images in histopathology.
Based on Pytorch Lightning. All aggregation modules can be loaded as model files in the classifier lightning module. Loss, models, optimizers, and schedulers can be specified as strings according to the PyTorch name in the config file.
## Setup
Setup and with your data paths and training configurations. All entries in brackets should be customized `.data_config.yaml` `config.yaml`.
We recommend that you use anaconda to configure PD-L1-expression-prediction’s dependency environment.
Use the environment configuration file located in to create a conda environment: `environment.yml`
```sh
conda env create -n pdl1 -f environment.yaml
```
## Data structure

* `clini_table.xlsx`: Table (Excel-file) with clinically important labels. Each patient has a unique entry, column names `PATIENT` and `TARGET` are required.

| PATIENT	| TARGET	| GENDER	| AGE |
| ---       | ---       | ---       | --- |
| ID_345    | positive	| female	| 61  |
| ID_459    | negative	| male	    | 67  |
| ID_697    | NA	    | female	| 42  |

* `slide.csv`: Table (csv-file) with patient id's matched with slide / file names (column names `FILENAME` and `PATIENT`). Patients can have multiple entries if they have multiple slides.

| FILENAME	| PATIENT	|
| ---       | ---       |
| ID_345_slide01    | ID_345    |
| ID_345_slide02    | ID_345    |
| ID_459_slide01    | ID_459    |

* folder with features as `.h5-files`. Filenames correspond to filenames in `slide.csv`

## Training

by running 
```
python train_k-fold.py --name <name> --data_config <path/to/data_config.yaml> --config <path/to/config.yaml>
```
## Testing

You can test a given model with the following command:
```
python test.py --model_path <path/to/model.ckpt> --name <custom-name> --config_file <path/to/config.yaml> --data_config <path/to/data_config.yaml>
```
