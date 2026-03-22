from pathlib import Path
import csv
import json
import sys

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "linear_mmm_model.pkl"
FEATURES_PATH = BASE_DIR / "mmm_model_features.json"


def load_features():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def read_rows(input_path, feature_names):
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing_columns = [name for name in feature_names if name not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(missing_columns)
            )

        rows = list(reader)
        if not rows:
            raise ValueError("Input CSV has no data rows.")

    matrix = []
    for index, row in enumerate(rows, start=1):
        try:
            matrix.append([float(row[name]) for name in feature_names])
        except ValueError as exc:
            raise ValueError(f"Row {index} has a non-numeric value: {exc}") from exc

    return rows, np.array(matrix)


def write_predictions(output_path, rows, predictions):
    fieldnames = list(rows[0].keys()) + ["predicted_revenue"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            result_row = dict(row)
            result_row["predicted_revenue"] = float(prediction)
            writer.writerow(result_row)


def main():
    input_name = sys.argv[1] if len(sys.argv) > 1 else "batch_input_example.csv"
    output_name = sys.argv[2] if len(sys.argv) > 2 else "batch_predictions.csv"

    input_path = BASE_DIR / input_name
    output_path = BASE_DIR / output_name

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    model = joblib.load(MODEL_PATH)
    feature_names = load_features()
    rows, matrix = read_rows(input_path, feature_names)
    predictions = model.predict(matrix)
    write_predictions(output_path, rows, predictions)

    print(f"Read {len(rows)} rows from: {input_path.name}")
    print(f"Wrote predictions to: {output_path.name}")


if __name__ == "__main__":
    main()
