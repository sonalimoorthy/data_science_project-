"""
Build a classification model to predict Risk_Level derived from Conflict_Level.

Risk mapping:
- 0 to 2   -> Low
- 3 to 6   -> Medium
- 7 to 10  -> High
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "final" / "master_country_year.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET = "Risk_Level"
CLASS_ORDER = ["Low", "Medium", "High"]

# Use features that are available in the table without directly using Conflict_Level
# (the source variable used to construct the target), to avoid trivial leakage.
NUMERIC_FEATURES = ["Num_Attacks", "Fatalities", "Injuries", "GDP_per_capita", "Population"]
CATEGORICAL_FEATURES = ["Region", "Year"]


def to_risk_level(conflict_level: float) -> str:
    if conflict_level <= 2:
        return "Low"
    if conflict_level <= 6:
        return "Medium"
    return "High"


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Conflict_Level"] = pd.to_numeric(out["Conflict_Level"], errors="coerce")
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Conflict_Level"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    out[TARGET] = out["Conflict_Level"].map(to_risk_level)
    out["Year"] = out["Year"].astype(int).astype(str)
    return out


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    classifier = DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, random_state=RANDOM_STATE)
    return Pipeline([("prep", preprocessor), ("clf", classifier)])


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    data = prepare_dataset(df)

    x = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, zero_division=0)

    print("Risk_Level Classification (Decision Tree)")
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
    print(f"Accuracy: {accuracy:.6f}")

    print("\nConfusion matrix (rows=true, cols=pred; order: Low, Medium, High):")
    print(cm)

    print("\nClassification report:")
    print(report)

    # Performance interpretation:
    # - Accuracy gives overall correctness.
    # - The confusion matrix shows which classes are being mixed up.
    # - Precision/recall/F1 by class indicate whether performance is balanced
    #   across Low/Medium/High risk levels.
    print("Performance note:")
    print(
        "- Check recall for each class in the report. Lower recall means the model "
        "is missing that risk level more often."
    )


if __name__ == "__main__":
    main()
