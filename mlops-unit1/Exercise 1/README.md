# mlops-unit1

## Project objective

Provide a minimal, MLOps-friendly ML project with reproducible setup using Python + Git.

The repo is split into:
- `exercise2/`: dataset + code to train and save a model
- `exercise3/README.md`: workflow documentation

## Dependencies

`requirements.txt` lists the Python dependencies needed to run the ML code.

## Recreate the environment (Exercise 4)

1) Clone the repository into a new folder:

```powershell
cd "c:\Users\ajita\Desktop\MLOps"
git clone https://github.com/Ajitabh-12-7/mlops-unit1.git mlops-unit1-ex4
```

2) Create and activate a virtual environment:

```powershell
cd "c:\Users\ajita\Desktop\MLOps\mlops-unit1-ex4\exercise2"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies:

```powershell
pip install -r requirements.txt
```

4) Verify the code runs:

```powershell
python .\src\stats.py
python .\src\train.py
```

Model output:
- `exercise2/models/model.joblib` (saved by `scripts/train.py`; gitignored)

Verified: after cloning into `mlops-unit1-ex4`, running the above commands printed metrics (including `Accuracy: 1.0000`) and created `exercise2/models/model.joblib`.
