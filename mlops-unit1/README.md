# mlops-unit1

## Project objective

Build a minimal, MLOps-friendly ML project scaffold: load data, run basic data checks, train a simple model, print evaluation metrics, and save the trained model.

## Dataset used

- `data/sample.csv`: a small Iris-like dataset with 4 numeric feature columns and a `species` target column.

## Project structure

```
mlops-unit1/
├── data/
├── src/
├── models/
├── requirements.txt
└── README.md
```

## Steps to run

### Setup (Windows PowerShell)

```powershell
cd "c:\Users\ajita\Desktop\MLOps\mlops-unit1"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Basic dataset statistics

```powershell
python .\src\stats.py
python .\src\stats.py --path .\data\sample.csv
```

### Train, evaluate, and save model

```powershell
python .\src\train.py
```

Model output:
- `models/model.joblib`
