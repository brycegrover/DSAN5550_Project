#!/usr/bin/env python3
"""
Vessel and sea ice analysis (revised).

Reads:
    processed/features/features_with_targets.parquet

Performs, on *active* grid cells (cells that ever see vessels):
    - EDA: probability of vessel presence vs ice concentration
    - Logistic regression for next-month presence using ice + season only
    - Random Forest classifier for next-month presence using full features
    - Random Forest regressor for next-month counts

Writes:
    processed/analysis/
        eda_prob_vs_ice_active.png
        logistic_next_roc.png
        rf_classifier_roc.png
        rf_classifier_feature_importance.png
        rf_regressor_scatter.png
        metrics.txt
        emissions.csv   (if CodeCarbon is installed)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional: CodeCarbon for emissions tracking
try:
    from codecarbon import EmissionsTracker

    HAS_CODEC = True
except ImportError:
    HAS_CODEC = False

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "processed" / "features" / "features_with_targets.parquet"
OUT_DIR = PROJECT_ROOT / "processed" / "analysis"


# ---------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run scripts/build_targets.py first."
        )

    df = pd.read_parquet(DATA_PATH)

    expected = {
        "iy",
        "ix",
        "lat",
        "lon",
        "ice_conc_mean",
        "year",
        "month",
        "vessel_count",
        "t",
        "ice_conc_prev",
        "vessel_count_prev",
        "delta_ice",
        "has_vessel",
        "vessel_count_next",
        "has_vessel_next",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in features_with_targets: {missing}")

    # Drop any rows with obvious missing predictors or targets
    df = df.dropna(
        subset=[
            "ice_conc_mean",
            "ice_conc_prev",
            "delta_ice",
            "vessel_count",
            "vessel_count_prev",
            "has_vessel",
            "vessel_count_next",
            "has_vessel_next",
            "month",
            "t",
            "lat",
            "lon",
        ]
    ).reset_index(drop=True)

    # Ensure integer types for counts and logical flags
    df["vessel_count"] = df["vessel_count"].astype(int)
    df["vessel_count_prev"] = df["vessel_count_prev"].astype(int)
    df["vessel_count_next"] = df["vessel_count_next"].astype(int)
    df["has_vessel"] = df["has_vessel"].astype(int)
    df["has_vessel_next"] = df["has_vessel_next"].astype(int)

    # Month as categorical with all 12 categories, to stabilize dummy columns
    df["month"] = df["month"].astype(int)
    df["month"] = pd.Categorical(df["month"], categories=list(range(1, 13)))

    return df


def filter_active_cells(df: pd.DataFrame, min_total_vessels: int = 5) -> pd.DataFrame:
    """
    Restrict to grid cells that ever see at least min_total_vessels detections
    over the full time period. This removes vast regions of permanently empty ocean.
    """
    print("Filtering to active grid cells...")
    cell_totals = df.groupby(["iy", "ix"])["vessel_count"].sum()
    active_idx = cell_totals[cell_totals >= min_total_vessels].index

    before_rows = len(df)
    before_cells = df[["iy", "ix"]].drop_duplicates().shape[0]

    df_active = (
        df.set_index(["iy", "ix"])
        .loc[active_idx]
        .reset_index()
        .sort_values(["iy", "ix", "t"])
        .reset_index(drop=True)
    )

    after_rows = len(df_active)
    after_cells = df_active[["iy", "ix"]].drop_duplicates().shape[0]

    print(
        f"Active cells filter: {before_cells} cells -> {after_cells} cells, "
        f"{before_rows} rows -> {after_rows} rows."
    )

    return df_active


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def time_train_test_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Time based train test split using t as time index.
    """
    t_sorted = np.sort(df["t"].unique())
    cutoff_index = int(np.floor((1.0 - test_fraction) * len(t_sorted)))
    cutoff_t = t_sorted[cutoff_index]

    train_df = df[df["t"] <= cutoff_t].copy()
    test_df = df[df["t"] > cutoff_t].copy()

    print(
        f"Train period t <= {cutoff_t}, "
        f"train n = {len(train_df)}, test n = {len(test_df)}"
    )
    return train_df, test_df


# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------


def eda_prob_vs_ice(df: pd.DataFrame, n_bins: int = 20):
    """
    Plot probability of vessel presence (this month) as a function of ice concentration
    for active cells.
    """
    print("Running EDA: probability of vessel presence vs ice concentration (active cells)")

    ice = df["ice_conc_mean"].clip(0.0, 1.0)
    has_vessel = df["has_vessel"]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(ice, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    prob = []
    counts = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            prob.append(np.nan)
            counts.append(0)
        else:
            prob.append(has_vessel[mask].mean())
            counts.append(mask.sum())

    prob = np.array(prob)
    counts = np.array(counts)

    plt.figure()
    plt.plot(bin_centers, prob, marker="o")
    plt.xlabel("Ice concentration (fraction)")
    plt.ylabel("P(has_vessel = 1)")
    plt.title("Probability of vessel presence vs ice concentration (active cells)")
    plt.grid(True)

    out_path = OUT_DIR / "eda_prob_vs_ice_active.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved EDA plot to {out_path}")


# ---------------------------------------------------------------------
# Modeling helpers
# ---------------------------------------------------------------------


def build_full_feature_matrix(df: pd.DataFrame, next_month: bool = True):
    """
    Build X, y for RF models.
    If next_month is True, y is has_vessel_next or vessel_count_next; the caller
    chooses which column to use.
    """
    X = df[
        [
            "ice_conc_mean",
            "ice_conc_prev",
            "delta_ice",
            "vessel_count",
            "vessel_count_prev",
            "lat",
            "lon",
            "month",
        ]
    ].copy()

    X = pd.get_dummies(X, columns=["month"], drop_first=True)
    feature_names = X.columns.tolist()
    return X.values, feature_names


def build_ice_only_matrix(df: pd.DataFrame):
    """
    Features for the logistic baseline: ice + season only, predicting has_vessel_next.
    """
    X = df[["ice_conc_mean", "ice_conc_prev", "delta_ice", "month"]].copy()
    X = pd.get_dummies(X, columns=["month"], drop_first=True)
    feature_names = X.columns.tolist()
    return X.values, feature_names


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


def run_logistic_next(df: pd.DataFrame, metrics_file):
    """
    Logistic regression baseline:
        Predict has_vessel_next using only ice variables + season (no location,
        no vessel history). This isolates what ice alone can do.
    """
    print("Running logistic regression for has_vessel_next (ice + season only)")

    train_df, test_df = time_train_test_split(df)

    X_train, feat_names = build_ice_only_matrix(train_df)
    X_test, _ = build_ice_only_matrix(test_df)

    y_train = train_df["has_vessel_next"].values
    y_test = test_df["has_vessel_next"].values

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"Logistic (next-month, ice-only) AUC: {auc:.3f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(report)

    metrics_file.write(
        "\n=== Logistic Regression (has_vessel_next, ice + season only) ===\n"
    )
    metrics_file.write(f"AUC: {auc:.3f}\n")
    metrics_file.write("Confusion matrix:\n")
    metrics_file.write(str(cm) + "\n")
    metrics_file.write("Classification report:\n")
    metrics_file.write(report + "\n")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Logistic ROC (has_vessel_next, ice + season only)")
    plt.legend()
    plt.grid(True)

    out_path = OUT_DIR / "logistic_next_roc.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved logistic ROC to {out_path}")


def run_rf_classifier(df: pd.DataFrame, metrics_file):
    """
    Random Forest classifier:
        Predict has_vessel_next using full feature set (ice, history, lat, lon, season).
    """
    print("Running Random Forest classifier for has_vessel_next (full features)")

    train_df, test_df = time_train_test_split(df)

    X_train, feature_names = build_full_feature_matrix(train_df, next_month=True)
    X_test, _ = build_full_feature_matrix(test_df, next_month=True)

    y_train = train_df["has_vessel_next"].values
    y_test = test_df["has_vessel_next"].values

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"RF classifier AUC: {auc:.3f}")
    print("RF classifier confusion matrix:")
    print(cm)
    print("RF classifier classification report:")
    print(report)

    metrics_file.write("\n=== Random Forest Classifier (has_vessel_next, full) ===\n")
    metrics_file.write(f"AUC: {auc:.3f}\n")
    metrics_file.write("Confusion matrix:\n")
    metrics_file.write(str(cm) + "\n")
    metrics_file.write("Classification report:\n")
    metrics_file.write(report + "\n")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Random Forest ROC (has_vessel_next)")
    plt.legend()
    plt.grid(True)
    out_path = OUT_DIR / "rf_classifier_roc.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved RF classifier ROC to {out_path}")

    # Feature importance
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in idx]
    sorted_importances = importances[idx]

    plt.figure(figsize=(8, 6))
    top_k = min(20, len(sorted_features))
    plt.barh(sorted_features[:top_k][::-1], sorted_importances[:top_k][::-1])
    plt.xlabel("Feature importance")
    plt.title("Random Forest feature importance (top 20)")
    plt.tight_layout()
    out_path = OUT_DIR / "rf_classifier_feature_importance.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved RF classifier feature importance to {out_path}")

    metrics_file.write("Feature importances (descending):\n")
    for fname, imp in zip(sorted_features, sorted_importances):
        metrics_file.write(f"{fname}: {imp:.4f}\n")


def run_rf_regressor(df: pd.DataFrame, metrics_file):
    """
    Random Forest regressor for vessel_count_next.
    """
    print("Running Random Forest regressor for vessel_count_next")

    train_df, test_df = time_train_test_split(df)

    X_train, feature_names = build_full_feature_matrix(train_df, next_month=True)
    X_test, _ = build_full_feature_matrix(test_df, next_month=True)

    y_train = train_df["vessel_count_next"].values
    y_test = test_df["vessel_count_next"].values

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RF regressor RMSE: {rmse:.3f}")
    print(f"RF regressor R2: {r2:.3f}")

    metrics_file.write("\n=== Random Forest Regressor (vessel_count_next) ===\n")
    metrics_file.write(f"RMSE: {rmse:.3f}\n")
    metrics_file.write(f"R2: {r2:.3f}\n")

    # Scatter plot predicted vs actual
    plt.figure()
    plt.scatter(y_test, y_pred, s=5, alpha=0.5)
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Actual vessel_count_next")
    plt.ylabel("Predicted vessel_count_next")
    plt.title("RF regressor: predicted vs actual vessel_count_next")
    plt.grid(True)
    out_path = OUT_DIR / "rf_regressor_scatter.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved RF regressor scatter to {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    ensure_out_dir()
    df = load_data()
    df = filter_active_cells(df, min_total_vessels=5)

    metrics_path = OUT_DIR / "metrics.txt"

    tracker = None
    if HAS_CODEC:
        print("Starting CodeCarbon EmissionsTracker...")
        tracker = EmissionsTracker(
            project_name="dsan5550_ice_vessels",
            output_dir=str(OUT_DIR),
            output_file="emissions.csv",
            save_to_file=True,
        )
        tracker.start()
    else:
        print("CodeCarbon not installed; skipping emissions tracking.")

    with open(metrics_path, "w") as metrics_file:
        metrics_file.write("Vessel and sea ice model metrics (active cells only)\n")

        eda_prob_vs_ice(df)
        run_logistic_next(df, metrics_file)
        run_rf_classifier(df, metrics_file)
        run_rf_regressor(df, metrics_file)

        if tracker is not None:
            emissions_kg = tracker.stop()
            metrics_file.write("\n=== CodeCarbon ===\n")
            metrics_file.write(f"Total emissions (kg CO2eq): {emissions_kg:.6f}\n")
            print(
                f"CodeCarbon reported total emissions: {emissions_kg:.6f} kg CO2eq "
                f"(see {OUT_DIR / 'emissions.csv'})"
            )

    print(f"All metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
