# AutoTsD: a Multi-Module Time Since Deposition (TsD) Prediction Tool
This repository contains two independent prediction modules: **Autoencoder Multi-Task Prediction** (for simultaneous tissue and time prediction) and **Boruta RFC Prediction** (for Random Forest models with Boruta feature selection) based on the GTEX v10 human samples.

## Prerequisites

Ensure you have the following Python libraries installed:

```bash
pip install -r requirements.txt
```

-----

## 1\. Autoencoder Multi-Task Prediction

Located in: `./autoencoder/predict.py`

This module utilizes a **Multi-Task Stacked Autoencoder (SAE)** architecture to perform **Tissue Classification** and **Time Regression** simultaneously. It employs an attention mechanism (`TimeAttention`) to enhance time feature extraction using tissue classification probabilities. Current version supports human skin and muscle samples.

### Overview
  * **Inputs:** Requires two separate CSV files (Time features and Tissue features).
  * **Logic:**
    1.  Aligns sample IDs between the two input files (intersection).
    2.  Normalizes data using pre-saved Scalers.
    3.  Loads the model architecture dynamically based on a JSON config.
    4.  Performs inference on GPU (if available) or CPU.
  * **Outputs:** Predicted Tissue Label and Time (in hours), saved to a CSV file.

### File Dependencies
To run this script, you need the following artifacts generated during training:

| File Type | Default Name | Description |
| :--- | :--- | :--- |
| **Model Weights** | `best_model.pth` | The PyTorch model state dictionary. |
| **Metadata** | `model_meta.pkl` | Pickle file containing Scalers, LabelEncoders, and feature column names. |
| **Config** | `config.json` | JSON file defining the model architecture (layer sizes, dropout, etc.). |

### Usage

#### Basic Example

```bash
python ./autoencoder/predict.py \
  --input_time data/time_features.csv \
  --input_tissue data/tissue_features.csv
```

#### Custom Paths

```bash
python ./autoencoder/predict.py \
  --input_time test/time.csv \
  --input_tissue test/tissue.csv \
  --model checkpoints/best_model.pth \
  --meta checkpoints/model_meta.pkl \
  --config checkpoints/config.json \
  --output results/sae_predictions.csv
```

#### Arguments

| Argument | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `--input_time` | ✅ | - | Input CSV for Time features (Col 0 must be ID). |
| `--input_tissue` | ✅ | - | Input CSV for Tissue features (Col 0 must be ID). |
| `--model` | ❌ | `best_model.pth` | Path to the trained `.pth` model file. |
| `--meta` | ❌ | `model_meta.pkl` | Path to the metadata `.pkl` file. |
| `--config` | ❌ | `config.json` | Path to the architecture `.json` config. |
| `--output` | ❌ | `predictions.csv` | Output filename for results. |

-----

## 2\. Boruta RFC Prediction

Located in: `./boruta_rfc/predict.py`

This module handles inference for machine learning models (e.g., Random Forest) where feature selection was performed using **Boruta**. It ensures that the input data strictly matches the features selected during training. Current version only supports human skin samples.

### Overview

  * **Inputs:** A single CSV file containing sample features.
  * **Logic:**
    1.  Loads the list of selected features (`selected_features.csv`).
    2.  Validates that the input CSV contains all required features.
    3.  Aligns the input columns to match the training order.
    4.  Runs prediction using the loaded model.
  * **Outputs:** Sample IDs and their predicted values.

### File Dependencies

| File Type | Default Name | Description |
| :--- | :--- | :--- |
| **Model** | `final_model.pkl` | The trained model object saved via `joblib`. |
| **Feature List** | `selected_features.csv` | A CSV containing the names of features selected by Boruta. |

### Usage

#### Basic Example

```bash
python ./boruta_rfc/predict.py --input data/new_samples.csv
```

#### Custom Paths

```bash
python ./boruta_rfc/predict.py \
  --input data/validation_set.csv \
  --model models/rf_classifier.pkl \
  --features results/boruta_features/selected_features.csv \
  --output results/boruta_predictions.csv
```

#### Arguments

| Argument | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `--input` | ✅ | - | Path to input CSV file (Col 0 must be ID). |
| `--output` | ❌ | `predictions.csv` | Path to save the results. |
| `--model` | ❌ | `./final_model.pkl` | Path to the trained model file. |
| `--features` | ❌ | `./selected_features.csv` | Path to the selected features list. |

-----

## Data Format

For all input CSV files, the **first column** must be the Sample ID (index). Please check the demo CSV files.

**Example:**

```csv
SampleID,Gene_A,Gene_B,Gene_C,...
S001,0.5,1.2,0.0,...
S002,0.1,0.9,1.5,...
```
