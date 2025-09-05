# Fraud Detection - Inference CLI

Pretrained fraud detection model with a simple CLI to score your own transaction data. The final, readable notebook is `final.ipynb` at the project root. Inference is performed by `src/predict_fraud.py` using the same cleaning and feature engineering defined in `src/fraud_features.py`.

## Repository layout
- `src/`
  - `predict_fraud.py`: Command-line script to generate predictions
  - `fraud_features.py`: Cleaning and feature engineering used at inference time
- `models/`
  - `model.joblib`: Trained model artifact
- `data/`
  - `fraud.csv`: Example dataset (optional; large files are usually gitignored)
  - `fraud_predictions.csv`: Example output (created by the CLI)
- `notebooks/`
  - `fraud.ipynb`: Earlier/archived notebook(s)
- `final.ipynb`: Final version of the notebook
- `requirements.txt`, `.gitignore`, `README.md`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt
```

If LightGBM fails to build from source on your platform, consider installing a prebuilt wheel or using conda:
```bash
# Example: conda
conda create -n fraud python=3.10 -y
conda activate fraud
pip install -r requirements.txt
```

## Run predictions
From the repository root, score a CSV and write predictions to `data/fraud_predictions.csv`:
```bash
python src/predict_fraud.py \
  --model models/model.joblib \
  --data data/fraud.csv \
  --output data/fraud_predictions.csv
```

Use your own dataset (any path):
```bash
python src/predict_fraud.py -m models/model.joblib -d /path/to/your_data.csv
```

The script also prints the first 10 predictions to stdout. If `--output` is omitted, predictions are written next to your input as `<input>_predictions.csv`.

## Input schema (raw columns)
Your CSV should include these columns (header row required):

- **step**: integer time step (e.g., hours since start)
- **customer**: string customer identifier
- **age**: string age bucket (e.g., `25-29`, `31-35`, or `U` for unknown)
- **gender**: string (e.g., `M`, `F`, or other)
- **zipcodeOri**: string original customer ZIP (optional; dropped during features)
- **merchant**: string merchant identifier
- **zipMerchant**: string merchant ZIP (optional; dropped during features)
- **category**: string merchant category (e.g., `es_barsandrestaurants`)
- **amount**: numeric transaction amount
- **fraud**: optional 0/1 label (ignored for prediction if present)

Notes:
- Missing `fraud` is fine at inference time.
- Extra columns are safely handled when aligning to the model.
- If expected model features are missing, they are added with zeros during alignment.

## Output
The prediction CSV contains:
- Any of `customer`, `merchant`, `step` if present in input (for reference)
- `pred_proba`: probability of the positive (fraud) class when available
- `pred_label`: binary label using the provided threshold (default 0.5)

## Reproducibility
All inference-time transformations are in `src/fraud_features.py`:
- Categorical cleanup and one-hot encoding
- Time-derived features and per-customer rolling stats
- Per-merchant aggregates (uses labels if present, otherwise counts)
- Column alignment to the model’s expected feature order