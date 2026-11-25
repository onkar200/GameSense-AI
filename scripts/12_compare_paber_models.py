
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

# GameSense AI - Chronological GTI vs PABER Comparison

INPUT_FILE = "data/features/gamesense_validation_paber.csv"

OUTPUT_FILE = "data/features/gamesense_paber_comparison.csv"
PREDICTIONS_FILE = "data/features/gamesense_paber_predictions.csv"

TARGET = "next_map_poor_performance"

print("Loading PABER validation dataset...")

df = pd.read_csv(INPUT_FILE)

# Validate required columns.
required = [
    "match_id",
    "utc_timestamp",
    TARGET,
    "gti_score",
    "paber_score"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["utc_timestamp"] = pd.to_datetime(
    df["utc_timestamp"],
    errors="coerce",
    utc=True
)

for col in [
    TARGET,
    "gti_score",
    "paber_score",
    "gti_data_available",
    "paber_data_available"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

df[TARGET] = df[TARGET].replace(
    [np.inf, -np.inf],
    np.nan
)

# Keep only observations where both scores are available.
# This ensures a fair comparison using identical records.
df = df.dropna(
    subset=[
        TARGET,
        "match_id",
        "utc_timestamp",
        "gti_score",
        "paber_score"
    ]
).copy()

df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

if "gti_data_available" in df.columns:
    df = df[df["gti_data_available"] == 1].copy()

if "paber_data_available" in df.columns:
    df = df[df["paber_data_available"] == 1].copy()

print("\nCOMMON EVALUATION DATA")

print("Rows:", len(df))
print("Players:", df["player_id"].nunique()
      if "player_id" in df.columns else "Unavailable")
print("Series:", df["match_id"].nunique())
print("Target distribution:")
print(df[TARGET].value_counts().sort_index())

if len(df) == 0:
    raise ValueError("No common valid GTI/PABER observations.")

# Assign entire series to train or test.
series_dates = (
    df.groupby("match_id")["utc_timestamp"]
    .max()
    .sort_values()
)

if len(series_dates) < 4:
    raise ValueError("Too few series for chronological evaluation.")

cutoff = int(len(series_dates) * 0.75)
cutoff = min(max(cutoff, 1), len(series_dates) - 1)

test_series = set(series_dates.index[cutoff:])

train_df = df[
    ~df["match_id"].isin(test_series)
].copy()

test_df = df[
    df["match_id"].isin(test_series)
].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Chronological split produced an empty partition.")

X_train_target = train_df[TARGET]
X_test_target = test_df[TARGET]

if X_train_target.nunique() < 2 or X_test_target.nunique() < 2:
    raise ValueError(
        "Train or test partition contains only one target class. "
        "Inspect the chronological cutoff."
    )

print("\nCHRONOLOGICAL SPLIT")

print("Training rows:", len(train_df))
print("Test rows:", len(test_df))
print("Training series:", train_df["match_id"].nunique())
print("Test series:", test_df["match_id"].nunique())

print(
    "Training period:",
    train_df["utc_timestamp"].min(),
    "to",
    train_df["utc_timestamp"].max()
)

print(
    "Test period:",
    test_df["utc_timestamp"].min(),
    "to",
    test_df["utc_timestamp"].max()
)

# Predictor groups.
# All are available at the end of the current map.
numeric_features = [
    "gti_score",
    "paber_score",
    "paber_persistence",
    "previous_maps_available",
    "previous_loss_streak",
    "map_number",
    "rating_all",
    "acs_all",
    "adr_all",
    "kd_ratio",
    "kast_all",
    "deaths_all",
    "first_death_rate",
    "multikill_total"
]

categorical_features = [
    "player_role",
    "map_name",
    "event",
    "patch"
]

numeric_features = [
    c for c in numeric_features if c in df.columns
]

categorical_features = [
    c for c in categorical_features if c in df.columns
]

# Do not include:
# next_map_rating
# next_map_acs
# next_map_game_id
# next_rating_change_pct
# next_acs_change_pct
# gti_group
# paber_level
# role-relative full-dataset features

def make_pipeline(numeric, categorical, estimator):

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric
        ))

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical
        ))

    preprocessor = ColumnTransformer(transformers)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])


# Identical observations and identical chronological split.
experiments = [
    {
        "name": "GTI only",
        "features": ["gti_score"],
        "model": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    },
    {
        "name": "PABER only",
        "features": ["paber_score"],
        "model": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    },
    {
        "name": "GTI + PABER",
        "features": [
            "gti_score",
            "paber_score"
        ],
        "model": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    },
    {
        "name": "Full features + GTI + PABER",
        "features": numeric_features + categorical_features,
        "model": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }
]

results = []
prediction_frames = []

def calculate_metrics(name, actual, predicted, probability):

    return {
        "model": name,
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        ),
        "roc_auc": roc_auc_score(actual, probability),
        "pr_auc": average_precision_score(
            actual, probability
        ),
        "brier_score": brier_score_loss(
            actual, probability
        ),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_series": train_df["match_id"].nunique(),
        "test_series": test_df["match_id"].nunique()
    }


# Majority-class baseline.
print("\nTraining majority baseline...")

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(
    np.zeros((len(train_df), 1)),
    X_train_target
)

baseline_probability = baseline.predict_proba(
    np.zeros((len(test_df), 1))
)[:, 1]

baseline_predictions = (
    baseline_probability >= 0.5
).astype(int)

results.append(
    calculate_metrics(
        "Majority baseline",
        X_test_target,
        baseline_predictions,
        baseline_probability
    )
)

# Train the four experiments.
for experiment in experiments:

    name = experiment["name"]
    features = experiment["features"]
    estimator = experiment["model"]

    print("\nTraining:", name)

    numeric = [
        c for c in features if c in numeric_features
    ]

    categorical = [
        c for c in features if c in categorical_features
    ]

    model = make_pipeline(
        numeric,
        categorical,
        estimator
    )

    model.fit(
        train_df[features],
        X_train_target
    )

    probabilities = model.predict_proba(
        test_df[features]
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    results.append(
        calculate_metrics(
            name,
            X_test_target,
            predictions,
            probabilities
        )
    )

    prediction_frames.append(
        pd.DataFrame({
            "model": name,
            "match_id": test_df["match_id"].to_numpy(),
            "game_id": (
                test_df["game_id"].to_numpy()
                if "game_id" in test_df.columns
                else np.nan
            ),
            "actual": X_test_target.to_numpy(),
            "predicted": predictions,
            "probability": probabilities
        })
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            X_test_target,
            predictions
        )
    )

# Save results.
comparison = pd.DataFrame(results)

Path("data/features").mkdir(
    parents=True,
    exist_ok=True
)

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

if prediction_frames:
    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    comparison.round(4).to_string(index=False)
)

print("\nSaved:", OUTPUT_FILE)
print("Saved:", PREDICTIONS_FILE)

print("\nPABER RESEARCH COMPARISON COMPLETE!")
