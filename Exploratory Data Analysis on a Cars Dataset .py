import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
OUTPUT_DIR = Path("cars_eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
    "highway-mpg", "price"
]

NUMERIC_COLUMNS = [
    "symboling", "wheel-base", "length", "width", "height", "curb-weight",
    "engine-size", "bore", "stroke", "compression-ratio", "horsepower",
    "peak-rpm", "city-mpg", "highway-mpg", "price"
]

def load_data():
    raw = urllib.request.urlopen(DATA_URL, timeout=30).read()
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLUMNS, na_values="?")
    return df

def clean_data(df):
    cleaned = df.copy()

    # normalized-losses has about 20% missing values, so retaining it would
    # require substantial imputation. It is removed rather than guessed.
    cleaned = cleaned.drop(columns=["normalized-losses"])

    # Convert numeric columns to numeric dtype.
    for col in NUMERIC_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Convert clearly numeric categorical labels.
    door_map = {"two": 2, "four": 4}
    cyl_map = {
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "eight": 8, "twelve": 12
    }
    cleaned["num-of-doors"] = cleaned["num-of-doors"].map(door_map)
    cleaned["num-of-cylinders"] = cleaned["num-of-cylinders"].map(cyl_map)

    # The two missing door values are not guessed from body style; rows with
    # unresolved critical values are removed to avoid introducing assumptions.
    before_missing_drop = len(cleaned)
    critical = [
        "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
        "drive-wheels", "engine-location", "engine-size", "horsepower",
        "bore", "stroke", "peak-rpm", "price"
    ]
    cleaned = cleaned.dropna(subset=critical)

    # Remove exact duplicates.
    before_dupes = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_dupes - len(cleaned)

    return cleaned, before_missing_drop, duplicates_removed

def save_charts(df, chart_dir):
    plt.figure(figsize=(8, 5))
    df["fuel-type"].value_counts().plot(kind="bar")
    plt.title("Fuel Type Distribution")
    plt.xlabel("Fuel Type")
    plt.ylabel("Number of Cars")
    plt.tight_layout()
    plt.savefig(chart_dir / "01_fuel_type.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    df["body-style"].value_counts().sort_values(ascending=False).plot(kind="bar")
    plt.title("Cars by Body Style")
    plt.xlabel("Body Style")
    plt.ylabel("Number of Cars")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(chart_dir / "02_body_style.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["price"], bins=15)
    plt.title("Distribution of Car Prices")
    plt.xlabel("Price (USD)")
    plt.ylabel("Number of Cars")
    plt.tight_layout()
    plt.savefig(chart_dir / "03_price_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["engine-size"], df["price"], alpha=0.7)
    plt.title("Engine Size vs Car Price")
    plt.xlabel("Engine Size")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(chart_dir / "04_engine_price.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["horsepower"], df["price"], alpha=0.7)
    plt.title("Horsepower vs Car Price")
    plt.xlabel("Horsepower")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(chart_dir / "05_hp_price.png", dpi=180)
    plt.close()

    corr_cols = [
        "curb-weight", "engine-size", "horsepower", "city-mpg",
        "highway-mpg", "price"
    ]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation="nearest", aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(chart_dir / "06_correlation_heatmap.png", dpi=180)
    plt.close()

def print_findings(df):
    corr = df[["engine-size", "curb-weight", "horsepower",
               "city-mpg", "highway-mpg", "price"]].corr()["price"].sort_values(
                   ascending=False
               )
    print("\nKEY FINDINGS")
    print("1. Most cars use gas fuel:", df["fuel-type"].value_counts().idxmax())
    print("2. Most common body style:", df["body-style"].value_counts().idxmax())
    print("3. Median price:", round(df["price"].median(), 2))
    print("4. Engine-size vs price correlation:", round(corr["engine-size"], 3))
    print("5. Horsepower vs price correlation:", round(corr["horsepower"], 3))
    print("6. Highway-mpg vs price correlation:", round(corr["highway-mpg"], 3))

def main():
    raw = load_data()
    cleaned, before_missing_drop, duplicates_removed = clean_data(raw)
    chart_dir = OUTPUT_DIR / "charts"
    chart_dir.mkdir(exist_ok=True)

    print("RAW SHAPE:", raw.shape)
    print("RAW DATA TYPES:\n", raw.dtypes)
    print("\nMISSING VALUES BEFORE CLEANING:\n", raw.isna().sum().sort_values(ascending=False))
    print("\nDUPLICATE ROWS BEFORE CLEANING:", raw.duplicated().sum())
    print("\nCLEANED SHAPE:", cleaned.shape)
    print("DUPLICATES REMOVED:", duplicates_removed)

    numeric = cleaned.select_dtypes(include=np.number)
    categorical = cleaned.select_dtypes(exclude=np.number)

    print("\nNUMERICAL DESCRIPTIVE STATISTICS:\n", numeric.describe().round(2))
    print("\nCATEGORICAL SUMMARY:")
    for col in categorical.columns:
        print(f"\n{col}:")
        print(cleaned[col].value_counts().head(10))

    save_charts(cleaned, chart_dir)
    print_findings(cleaned)

    print("\nCharts saved in:", chart_dir.resolve())
    print("Project report can be prepared from these outputs.")

if __name__ == "__main__":
    main()
