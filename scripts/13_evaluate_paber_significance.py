
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score
)

# ==========================================
# 1. LOAD PREDICTIONS
# ==========================================

INPUT_FILE = "data/features/gamesense_paber_predictions.csv"
OUTPUT_FILE = "data/features/gamesense_paber_bootstrap.csv"

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "model",
    "match_id",
    "actual",
    "probability"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("\nDATA LOADED")
print("Rows:", len(df))
print("Models:", df["model"].unique())


# ==========================================
# 2. EXTRACT GTI AND COMBINED MODELS
# ==========================================

gti = df[df["model"] == "GTI only"].copy()
combined = df[df["model"] == "GTI + PABER"].copy()

if gti.empty or combined.empty:
    raise ValueError(
        "Could not find GTI only or GTI + PABER models. "
        "Check the model names in the CSV."
    )

# Use player_id and game_id as additional matching
# keys when available.
keys = ["match_id"]

for col in ["game_id", "player_id"]:
    if col in df.columns:
        keys.append(col)

gti = gti.sort_values(keys).reset_index(drop=True)
combined = combined.sort_values(keys).reset_index(drop=True)

if len(gti) != len(combined):
    raise ValueError("The two models do not have equal row counts.")

if not gti[keys].equals(combined[keys]):
    raise ValueError("The two models do not contain identical observations.")

if not gti["actual"].reset_index(drop=True).equals(
    combined["actual"].reset_index(drop=True)
):
    raise ValueError("Actual labels do not match between models.")

print("\nPAIRED DATA CHECK")
print("GTI observations:", len(gti))
print("Combined observations:", len(combined))
print("Unique test series:", gti["match_id"].nunique())


# ==========================================
# 3. METRIC CALCULATION
# ==========================================

def calculate_metrics(actual, probability):

    prediction = (probability >= 0.5).astype(int)

    if len(np.unique(actual)) < 2:
        return None

    return {
        "f1": f1_score(actual, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(
            actual, prediction
        ),
        "roc_auc": roc_auc_score(actual, probability)
    }


y_true = gti["actual"].to_numpy()
gti_prob = gti["probability"].to_numpy()
combined_prob = combined["probability"].to_numpy()

gti_metrics = calculate_metrics(y_true, gti_prob)
combined_metrics = calculate_metrics(y_true, combined_prob)

print("\nOBSERVED RESULTS")

for metric in gti_metrics:
    difference = combined_metrics[metric] - gti_metrics[metric]

    print(f"\n{metric.upper()}")
    print(f"GTI only:       {gti_metrics[metric]:.4f}")
    print(f"GTI + PABER:    {combined_metrics[metric]:.4f}")
    print(f"Difference:     {difference:+.4f}")


# ==========================================
# 4. CLUSTER BOOTSTRAP BY MATCH/SERIES
# ==========================================

N_BOOTSTRAP = 3000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# Keep every player-map observation from a series together.
series_groups = {
    match_id: np.where(gti["match_id"].to_numpy() == match_id)[0]
    for match_id in gti["match_id"].unique()
}

series_ids = list(series_groups.keys())

print("\nBOOTSTRAP SETTINGS")
print("Iterations:", N_BOOTSTRAP)
print("Series clusters:", len(series_ids))

bootstrap_results = []

for iteration in range(N_BOOTSTRAP):

    sampled_series = rng.choice(
        series_ids,
        size=len(series_ids),
        replace=True
    )

    sampled_indices = np.concatenate([
        series_groups[series_id]
        for series_id in sampled_series
    ])

    actual_sample = y_true[sampled_indices]
    gti_sample = gti_prob[sampled_indices]
    combined_sample = combined_prob[sampled_indices]

    gti_result = calculate_metrics(
        actual_sample, gti_sample
    )

    combined_result = calculate_metrics(
        actual_sample, combined_sample
    )

    # Skip samples containing only one target class.
    if gti_result is None or combined_result is None:
        continue

    bootstrap_results.append({
        "iteration": iteration + 1,
        "f1_difference":
            combined_result["f1"] - gti_result["f1"],
        "balanced_accuracy_difference":
            combined_result["balanced_accuracy"]
            - gti_result["balanced_accuracy"],
        "roc_auc_difference":
            combined_result["roc_auc"] - gti_result["roc_auc"]
    })


# ==========================================
# 5. CONFIDENCE INTERVALS
# ==========================================

bootstrap_df = pd.DataFrame(bootstrap_results)

if bootstrap_df.empty:
    raise ValueError("No valid bootstrap iterations were produced.")

bootstrap_df.to_csv(OUTPUT_FILE, index=False)

print("\nBOOTSTRAP RESULTS")
print("Valid iterations:", len(bootstrap_df))


for metric in [
    "f1_difference",
    "balanced_accuracy_difference",
    "roc_auc_difference"
]:

    values = bootstrap_df[metric]

    lower = values.quantile(0.025)
    upper = values.quantile(0.975)
    fraction_positive = (values > 0).mean()

    print("\n" + "=" * 45)
    print(metric.upper())
    print("=" * 45)

    print(f"Observed difference: {values.mean():+.4f}")
    print(f"95% CI: [{lower:+.4f}, {upper:+.4f}]")
    print(f"Bootstrap fraction above zero: {fraction_positive:.4f}")

    if lower > 0:
        print("Interpretation: Positive difference across the 95% interval.")
    elif upper < 0:
        print("Interpretation: Negative difference across the 95% interval.")
    else:
        print("Interpretation: Interval includes zero; improvement is uncertain.")


# ==========================================
# 6. SAVE SUMMARY
# ==========================================

summary = []

for metric in [
    "f1_difference",
    "balanced_accuracy_difference",
    "roc_auc_difference"
]:

    summary.append({
        "metric": metric,
        "mean_difference": bootstrap_df[metric].mean(),
        "ci_lower": bootstrap_df[metric].quantile(0.025),
        "ci_upper": bootstrap_df[metric].quantile(0.975),
        "fraction_positive": (bootstrap_df[metric] > 0).mean()
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "data/features/gamesense_paber_bootstrap_summary.csv",
    index=False
)

print("\nFILES SAVED")
print(OUTPUT_FILE)
print("data/features/gamesense_paber_bootstrap_summary.csv")

print("\nPABER SIGNIFICANCE EVALUATION COMPLETE!")
