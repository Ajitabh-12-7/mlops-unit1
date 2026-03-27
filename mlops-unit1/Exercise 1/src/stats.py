from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a CSV dataset and print basic statistics.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "sample.csv",
        help="Path to a CSV file (default: ./data/sample.csv).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.path)

    print(f"Dataset path: {args.path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
    print("\nColumns:")
    print(list(df.columns))

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > 0:
        print("\nNumeric summary (describe):")
        print(numeric.describe().T)
    else:
        print("\nNo numeric columns found.")

    print("\nMissing values per column:")
    print(df.isna().sum())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
