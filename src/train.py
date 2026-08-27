"""Train the used-car price model and save it as a reusable artifact.

The whole preprocessing chain lives inside a scikit-learn Pipeline, so the
exact transformations applied at training time are applied again at prediction
time. This removes the most common source of production bugs in ML systems:
training/serving skew.

Run:
    python src/train.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cars.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

TARGET = "price_million_toman"
NUMERIC_FEATURES = ["year", "mileage_km", "engine_size_l", "previous_owners"]
CATEGORICAL_FEATURES = ["model", "fuel_type", "transmission", "body_condition", "color"]

MAX_PLAUSIBLE_MILEAGE_KM = 700_000
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_data.py` first."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the problems that cannot be handled inside the pipeline.

    Row-level decisions (dropping duplicates, discarding impossible values)
    belong here, before the split. Column-level transformations that must be
    replayed at prediction time belong in the pipeline instead.
    """
    before = len(df)
    df = df.drop_duplicates()

    # Normalise category spelling: strip padding and lowercase everything.
    # The result is cast back to object dtype with numpy NaN, because
    # scikit-learn's imputer cannot interpret pandas' own NA marker.
    for column in CATEGORICAL_FEATURES:
        cleaned = df[column].astype("string").str.strip().str.lower()
        df[column] = cleaned.astype(object).where(cleaned.notna(), np.nan)

    # Mileage above the plausible ceiling is a data-entry error, not a signal.
    df.loc[df["mileage_km"] > MAX_PLAUSIBLE_MILEAGE_KM, "mileage_km"] = np.nan

    df = df.dropna(subset=[TARGET])
    print(f"Cleaning: {before} rows in, {len(df)} rows out.")
    return df


def build_pipeline() -> Pipeline:
    """Preprocessing + model as one object that can be pickled and served."""
    numeric_branch = SimpleImputer(strategy="median")

    categorical_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # Unseen categories at prediction time must not crash the API.
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, NUMERIC_FEATURES),
            ("categorical", categorical_branch, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = pipeline.predict(X_test)
    return {
        "mae_million_toman": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse_million_toman": round(
            float(root_mean_squared_error(y_test, predictions)), 2
        ),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }


def main() -> None:
    started = time.time()

    df = clean_data(load_data(DATA_PATH))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)

    # Cross-validation on the training set guards against a lucky split.
    cv_scores = cross_val_score(
        build_pipeline(), X_train, y_train, cv=5, scoring="r2", n_jobs=-1
    )
    metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_r2_std"] = round(float(cv_scores.std()), 4)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "categories": {
                column: sorted(df[column].dropna().unique().tolist())
                for column in CATEGORICAL_FEATURES
            },
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Training rows: {len(X_train)} | test rows: {len(X_test)}")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    print(f"Saved model to {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
