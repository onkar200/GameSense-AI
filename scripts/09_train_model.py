
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import GroupShuffleSplit
import joblib

INPUT_FILE = "data/features/gamesense_validation.csv"
MODEL_FILE = "models/gamesense_rf_model.joblib"
OUTPUT_FILE = "data/features/gamesense_predictions.csv"

TARGET = "next_map_poor_performance"

print("Loading validation dataset...")
df = pd.read_csv(INPUT_FILE)

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET, "match_id"]).copy()
df[TARGET] = df[TARGET].astype(int)

# Only features available at the end of the current map.
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

features = numeric_features + categorical_features

if not features:
    raise ValueError("No usable predictor columns were found.")

X = df[features].copy()
y = df[TARGET]
groups = df["match_id"].astype(str)

# Keep each series entirely in either train or test.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training records:", len(X_train))
print("Test records:", len(X_test))
print("Training series:", groups.iloc[train_idx].nunique())
print("Test series:", groups.iloc[test_idx].nunique())
print("Features:", features)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

def report_metrics(name, actual, predicted, probability=None):
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(actual, predicted), 4))
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(actual, predicted), 4)
    )
    print("Precision:", round(precision_score(actual, predicted, zero_division=0), 4))
    print("Recall:", round(recall_score(actual, predicted, zero_division=0), 4))
    print("F1:", round(f1_score(actual, predicted, zero_division=0), 4))

    if probability is not None and len(np.unique(actual)) == 2:
        print("ROC-AUC:", round(roc_auc_score(actual, probability), 4))

    print("Confusion matrix:")
    print(confusion_matrix(actual, predicted))
    print(classification_report(actual, predicted, zero_division=0))

report_metrics(
    "MAJORITY-CLASS BASELINE",
    y_test,
    baseline_predictions
)

report_metrics(
    "RANDOM FOREST",
    y_test,
    predictions,
    probabilities
)

# Save model and its feature schema.
Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump({
    "model": model,
    "features": features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": TARGET
}, MODEL_FILE)

# Save held-out predictions for dashboard and analysis.
results = df.iloc[test_idx][
    [
        c for c in [
            "player_name",
            "team_name",
            "event",
            "match_id",
            "game_id",
            "map_name",
            "map_number",
            "gti_score",
            "gti_level"
        ] if c in df.columns
    ]
].copy()

results["actual_poor_performance"] = y_test.to_numpy()
results["predicted_poor_performance"] = predictions
results["poor_performance_probability"] = probabilities

results.to_csv(OUTPUT_FILE, index=False)

print("\nMODEL SAVED:", MODEL_FILE)
print("HELD-OUT PREDICTIONS SAVED:", OUTPUT_FILE)
