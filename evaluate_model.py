from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "compressed_data.csv.gz"
MODEL_PATH = BASE_DIR / "linear_mmm_model.pkl"
FEATURES_PATH = BASE_DIR / "mmm_model_features.json"

DEFAULT_CUTOFF_DATE = "2024-01-01"
DEFAULT_REPORT_NAME = "evaluation_report.csv"
DEFAULT_SUMMARY_NAME = "evaluation_summary.json"
DEFAULT_TIMESERIES_PLOT_NAME = "evaluation_timeseries.png"
DEFAULT_ERROR_HIST_PLOT_NAME = "evaluation_error_hist.png"
DEFAULT_SCATTER_PLOT_NAME = "evaluation_scatter.png"

RAW_COLUMNS = [
    "DATE_DAY",
    "ALL_PURCHASES_ORIGINAL_PRICE",
    "ALL_PURCHASES_GROSS_DISCOUNT",
    "GOOGLE_PAID_SEARCH_SPEND",
    "GOOGLE_SHOPPING_SPEND",
    "GOOGLE_PMAX_SPEND",
    "META_FACEBOOK_SPEND",
    "META_INSTAGRAM_SPEND",
    "EMAIL_CLICKS",
    "ORGANIC_SEARCH_CLICKS",
    "DIRECT_CLICKS",
    "BRANDED_SEARCH_CLICKS",
]


def load_feature_names():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evaluation_frame(feature_names, cutoff_date):
    df = pd.read_csv(DATA_PATH, compression="gzip", usecols=RAW_COLUMNS)
    df["DATE_DAY"] = pd.to_datetime(df["DATE_DAY"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DATE_DAY"]).copy()

    base_features = [name for name in feature_names if name not in {"year", "month", "day_of_week"}]
    df[base_features] = df[base_features].fillna(0)
    df["year"] = df["DATE_DAY"].dt.year
    df["month"] = df["DATE_DAY"].dt.month
    df["day_of_week"] = df["DATE_DAY"].dt.dayofweek
    df["actual_revenue"] = (
        df["ALL_PURCHASES_ORIGINAL_PRICE"] - df["ALL_PURCHASES_GROSS_DISCOUNT"]
    )

    if cutoff_date:
        df = df[df["DATE_DAY"] >= pd.Timestamp(cutoff_date)].copy()

    if df.empty:
        raise ValueError("No rows available after applying the date filter.")

    return df


def build_summary(df, cutoff_date):
    valid_pct_error = df["abs_pct_error"].dropna()
    summary = {
        "row_count": int(len(df)),
        "cutoff_date": cutoff_date,
        "date_start": df["DATE_DAY"].min().date().isoformat(),
        "date_end": df["DATE_DAY"].max().date().isoformat(),
        "metrics": {
            "mae": float(df["abs_error"].mean()),
            "rmse": float(np.sqrt(np.mean(np.square(df["error"])))),
            "mape": float(valid_pct_error.mean()) if not valid_pct_error.empty else None,
            "mean_actual_revenue": float(df["actual_revenue"].mean()),
            "mean_predicted_revenue": float(df["predicted_revenue"].mean()),
        },
    }
    return summary


def generate_plots(df, timeseries_path, error_hist_path, scatter_path):
    daily = (
        df.groupby("DATE_DAY", as_index=False)[["actual_revenue", "predicted_revenue"]]
        .sum()
        .sort_values("DATE_DAY")
    )

    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily["DATE_DAY"], daily["actual_revenue"], label="Actual Revenue", linewidth=2)
    ax.plot(daily["DATE_DAY"], daily["predicted_revenue"], label="Predicted Revenue", linewidth=2)
    ax.set_title("Actual vs Predicted Revenue Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(timeseries_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["error"], bins=50, color="#4C78A8", edgecolor="white")
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Row Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(error_hist_path, dpi=150)
    plt.close(fig)

    scatter_df = df
    if len(scatter_df) > 3000:
        scatter_df = scatter_df.sample(n=3000, random_state=42)

    line_min = min(scatter_df["actual_revenue"].min(), scatter_df["predicted_revenue"].min())
    line_max = max(scatter_df["actual_revenue"].max(), scatter_df["predicted_revenue"].max())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        scatter_df["actual_revenue"],
        scatter_df["predicted_revenue"],
        alpha=0.35,
        s=18,
        color="#E45756",
    )
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="black", linewidth=1)
    ax.set_title("Actual vs Predicted Revenue")
    ax.set_xlabel("Actual Revenue")
    ax.set_ylabel("Predicted Revenue")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)


def main():
    cutoff_date = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CUTOFF_DATE
    report_name = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_REPORT_NAME
    summary_name = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_SUMMARY_NAME
    timeseries_name = (
        sys.argv[4] if len(sys.argv) > 4 else DEFAULT_TIMESERIES_PLOT_NAME
    )
    error_hist_name = (
        sys.argv[5] if len(sys.argv) > 5 else DEFAULT_ERROR_HIST_PLOT_NAME
    )
    scatter_name = sys.argv[6] if len(sys.argv) > 6 else DEFAULT_SCATTER_PLOT_NAME

    report_path = BASE_DIR / report_name
    summary_path = BASE_DIR / summary_name
    timeseries_path = BASE_DIR / timeseries_name
    error_hist_path = BASE_DIR / error_hist_name
    scatter_path = BASE_DIR / scatter_name

    feature_names = load_feature_names()
    model = joblib.load(MODEL_PATH)
    df = load_evaluation_frame(feature_names, cutoff_date)

    feature_frame = df[feature_names].astype(float)
    df["predicted_revenue"] = model.predict(feature_frame)
    df["error"] = df["predicted_revenue"] - df["actual_revenue"]
    df["abs_error"] = df["error"].abs()
    df["abs_pct_error"] = np.where(
        df["actual_revenue"].abs() > 1e-9,
        df["abs_error"] / df["actual_revenue"].abs(),
        np.nan,
    )

    report_columns = [
        "DATE_DAY",
        *feature_names,
        "ALL_PURCHASES_ORIGINAL_PRICE",
        "ALL_PURCHASES_GROSS_DISCOUNT",
        "actual_revenue",
        "predicted_revenue",
        "error",
        "abs_error",
        "abs_pct_error",
    ]
    df.loc[:, report_columns].to_csv(report_path, index=False, encoding="utf-8")

    summary = build_summary(df, cutoff_date)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    generate_plots(df, timeseries_path, error_hist_path, scatter_path)

    print(f"Evaluated {summary['row_count']} rows from {summary['date_start']} to {summary['date_end']}")
    print(f"Detailed report: {report_path.name}")
    print(f"Summary report: {summary_path.name}")
    print(f"Timeseries plot: {timeseries_path.name}")
    print(f"Error histogram: {error_hist_path.name}")
    print(f"Scatter plot: {scatter_path.name}")
    print(f"MAE: {summary['metrics']['mae']:.4f}")
    print(f"RMSE: {summary['metrics']['rmse']:.4f}")
    if summary["metrics"]["mape"] is None:
        print("MAPE: unavailable (actual revenue contained zeros only)")
    else:
        print(f"MAPE: {summary['metrics']['mape']:.4%}")


if __name__ == "__main__":
    main()
