# Exercise 3: Dev Workflow & Project Structure

## Project objective

Create an MLOps-friendly project scaffold and document how to run data checks and a simple training workflow.

## Dataset used

- `exercise2/data/sample.csv`: a small Iris-like dataset with 4 numeric feature columns and a `species` target column.

## Project structure

```
mlops-unit1/
├── exercise2/
│   ├── data/
│   ├── models/
│   ├── src/
│   └── requirements.txt
└── exercise3/
    └── README.md
```

## Steps to run

### Setup (Windows PowerShell)

```powershell
cd "c:\Users\ajita\Desktop\MLOps\mlops-unit1\exercise2"
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
- `exercise2/models/model.joblib`

## Push to GitHub

```powershell
git push
```
