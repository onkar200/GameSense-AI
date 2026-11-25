
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

INPUT_FILE = "data/features/gamesense_validation.csv"
OUTPUT_FILE = "data/features/gamesense_model_comparison.csv"
TARGET = "next_map_poor_performance"

print("Loading validation data...")
df = pd.read_csv(INPUT_FILE)

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df["utc_timestamp"] = pd.to_datetime(
    df["utc_timestamp"], errors="coerce", utc=True
)

df = df.dropna(
    subset=[TARGET, "match_id", "utc_timestamp"]
).copy()

df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

numeric_features = [
    "gti_score",
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
    "multikill_total",
]

categorical_features = [
    "player_role",
    "map_name",
    "event",
    "patch",
]

numeric_features = [c for c in numeric_features if c in df.columns]
categorical_features = [
    c for c in categorical_features if c in df.columns
]

all_features = numeric_features + categorical_features

# Assign each series a timestamp, then hold out the latest 25% of series.
series_dates = (
    df.groupby("match_id")["utc_timestamp"]
    .max()
    .sort_values()
)

cutoff_position = int(len(series_dates) * 0.75)
cutoff_position = min(max(cutoff_position, 1), len(series_dates) - 1)

test_series = set(series_dates.index[cutoff_position:])

train_df = df[~df["match_id"].isin(test_series)].copy()
test_df = df[df["match_id"].isin(test_series)].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Chronological split produced an empty partition.")

X_train = train_df[all_features]
X_test = test_df[all_features]
y_train = train_df[TARGET]
y_test = test_df[TARGET]

print("Training rows:", len(train_df))
print("Test rows:", len(test_df))
print("Training series:", train_df["match_id"].nunique())
print("Test series:", test_df["match_id"].nunique())
print("Training period:", train_df["utc_timestamp"].min(), "to",
      train_df["utc_timestamp"].max())
print("Test period:", test_df["utc_timestamp"].min(), "to",
      test_df["utc_timestamp"].max())

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "The chronological split has only one target class in train or test. "
        "Use a different series-level cutoff and inspect the class distribution."
    )


def make_pipeline(features):
    nums = [c for c in numeric_features if c in features]
    cats = [c for c in categorical_features if c in features]

    transformers = []

    if nums:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            nums,
        ))

    if cats:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cats,
        ))

    return ColumnTransformer(transformers)


experiments = [
    (
        "Majority baseline",
        [],
        DummyClassifier(strategy="most_frequent"),
    ),
    (
        "GTI only",
        ["gti_score"],
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    ),
    (
        "Full Random Forest",
        all_features,
        RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
]

results = []

for name, features, estimator in experiments:
    print("\nTraining:", name)

    if features:
        model = Pipeline([
            ("preprocessor", make_pipeline(features)),
            ("classifier", estimator),
        ])
        model.fit(X_train[features], y_train)
        predicted = model.predict(X_test[features])
        probability = model.predict_proba(X_test[features])[:, 1]
    else:
        model = estimator
        model.fit(np.zeros((len(y_train), 1)), y_train)
        predicted = model.predict(np.zeros((len(y_test), 1)))
        probability = None

    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "precision": precision_score(
            y_test, predicted, zero_division=0
        ),
        "recall": recall_score(
            y_test, predicted, zero_division=0
        ),
        "f1": f1_score(y_test, predicted, zero_division=0),
        "roc_auc": (
            roc_auc_score(y_test, probability)
            if probability is not None and y_test.nunique() == 2
            else np.nan
        ),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_series": train_df["match_id"].nunique(),
        "test_series": test_df["match_id"].nunique(),
    }

    results.append(row)

comparison = pd.DataFrame(results)
print("\nCHRONOLOGICAL MODEL COMPARISON")
print(comparison.round(4).to_string(index=False))

Path("data/features").mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_FILE, index=False)

print("\nSaved comparison to:", OUTPUT_FILE)
print("Existing trained model was not changed.")
