#!/usr/bin/env python3

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from codecarbon import EmissionsTracker


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
FEATURES_PATH = FEATURES_DIR / "features_with_targets.parquet"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"

ACTIVE_CELL_MIN_TOTAL_VESSELS = 5  # min ships to make a cell active
TRAIN_FRACTION = 0.8 # cutoff

def ensure_output_dir(): # debug
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    print(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    # debug
    if "has_vessel" not in df.columns:
        df["has_vessel"] = (df["vessel_count"] > 0).astype(int)
    if "has_vessel_next" not in df.columns:
        df["has_vessel_next"] = (df["vessel_count_next"] > 0).astype(int)

    # one hot encoed
    if "month" in df.columns:
        df = pd.get_dummies(df, columns=["month"], prefix="month")

    return df


 # function to active cells
def filter_active_cells(df, min_total_vessels=ACTIVE_CELL_MIN_TOTAL_VESSELS):
    print("Filtering to active grid cells.")
    group = df.groupby(["iy", "ix"])["vessel_count"].sum()
    active_index = group[group >= min_total_vessels].index # filter index

    before_cells = df[["iy", "ix"]].drop_duplicates().shape[0] # drop duplicates
    before_rows = len(df)

    active_mask = df.set_index(["iy", "ix"]).index.isin(active_index) # mask
    df_active = df[active_mask].copy()

    after_cells = df_active[["iy", "ix"]].drop_duplicates().shape[0] # after drop duplicates
    after_rows = len(df_active)

    print(
        f"Active cells filter: {before_cells} cells -> {after_cells} cells, "
        f"{before_rows} rows -> {after_rows} rows."
    )
    return df_active


def time_split(df, target_col):
    df = df.dropna(subset=[target_col]).copy()
    t_cut = df["t"].quantile(TRAIN_FRACTION)
    train_mask = df["t"] <= t_cut
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()

    print(
        f"Train period t <= {t_cut:.0f}, "
        f"train n = {len(df_train):d}, test n = {len(df_test):d}"
    )
    return df_train, df_test

# proabbility that a vessel is present vs ice concentration plot
def plot_prob_vs_ice(df, out_path):
    print("Running probability of vessel vs ice concentration in active cells")
    bins = np.linspace(0.0, 1.0, 11)
    df = df.copy()
    df["ice_bin"] = pd.cut(df["ice_conc_mean"], bins=bins, include_lowest=True)

    grouped = df.groupby("ice_bin", observed=False)["has_vessel"].mean().reset_index() # avergae has_vessel per bin then reset index
    bin_centers = [
        interval.left + (interval.right - interval.left) / 2.0
        for interval in grouped["ice_bin"]
    ]

    plt.figure(figsize=(7, 5))
    plt.plot(bin_centers, grouped["has_vessel"], marker="o")
    plt.xlabel("Ice concentration (fraction)")
    plt.ylabel("P(has_vessel = 1)")
    plt.title("Probability of vessel presence vs ice concentration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_roc(y_true, y_prob, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="orange")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved ROC plot to {out_path}")
    return roc_auc


def plot_feature_importance(model, feature_names, out_path, top_n=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names)[idx]
    top_importances = importances[idx]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_features)), top_importances[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel("Feature importance")
    plt.title("Random Forest feature importance (top {})".format(top_n))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved RF feature importance to {out_path}")


def plot_regressor_scatter(y_true, y_pred, out_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, s=5, alpha=0.5)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Actual vessel_count_next")
    plt.ylabel("Predicted vessel_count_next")
    plt.title("RF regressor: predicted vs actual vessel_count_next")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved RF regressor scatter to {out_path}")


def run_logistic_next_ice_only(df, metrics_lines):
    target_col = "has_vessel_next"

    # ice & month dummies
    month_cols = [c for c in df.columns if c.startswith("month_")]
    feature_cols = ["ice_conc_mean", "ice_conc_prev", "delta_ice"] + month_cols

    # keep rows where both features and target are there
    cols_for_model = feature_cols + [target_col, "t"]
    df_model = df[cols_for_model].dropna().copy()

    # time split on the cleaned subset
    t_cut = df_model["t"].quantile(TRAIN_FRACTION)
    train_mask = df_model["t"] <= t_cut
    df_train = df_model[train_mask].copy()
    df_test = df_model[~train_mask].copy()

    print(
        f"Running logistic regression for has_vessel_next\n"
        f"Train period t <= {t_cut:.0f}, train n = {len(df_train)}, test n = {len(df_test)}"
    )

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=-1)),
        ]
    )

    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_path = ANALYSIS_DIR / "logistic_next_roc.png"
    auc_val = plot_roc(
        y_test,
        y_prob,
        "Logistic ROC (has_vessel_next, ice + season only)",
        roc_path,
    )

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Logistic (next-month, ice-only) AUC: {auc_val:.3f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(report)

    metrics_lines.append(f"Logistic next-month (ice + season only) AUC: {auc_val:.3f}")
    metrics_lines.append("Logistic confusion matrix (next-month, ice-only):")
    metrics_lines.append(str(cm))
    metrics_lines.append("Logistic classification report (next-month, ice-only):")
    metrics_lines.append(report)
    metrics_lines.append("")


def run_rf_classifier_next_full(df, metrics_lines):
    target_col = "has_vessel_next"

    drop_cols = [
        "has_vessel",
        "has_vessel_next",
        "vessel_count_next",
        "iy",
        "ix",
        "t",
        "year",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    # keep only rows where both features and target are present
    cols_for_model = feature_cols + [target_col, "t"]
    df_model = df[cols_for_model].dropna().copy()

    # time split on the cleaned subset
    t_cut = df_model["t"].quantile(TRAIN_FRACTION)
    train_mask = df_model["t"] <= t_cut
    df_train = df_model[train_mask].copy()
    df_test = df_model[~train_mask].copy()

    print(
        f"Train period t <= {t_cut:.0f}, "
        f"train n = {len(df_train):d}, test n = {len(df_test):d}"
    )

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )

    print("Running Random Forest classifier for has_vessel_next (all feats)")
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    roc_path = ANALYSIS_DIR / "rf_classifier_roc.png"
    auc_val = plot_roc(
        y_test,
        y_prob,
        "Random Forest ROC (has_vessel_next)",
        roc_path,
    )

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"RF classifier AUC: {auc_val:.3f}")
    print("RF classifier confusion matrix:")
    print(cm)
    print("RF classifier classification report:")
    print(report)

    metrics_lines.append(f"RF classifier (has_vessel_next) AUC: {auc_val:.3f}")
    metrics_lines.append("RF classifier confusion matrix:")
    metrics_lines.append(str(cm))
    metrics_lines.append("RF classifier classification report:")
    metrics_lines.append(report)
    metrics_lines.append("")

    fi_path = ANALYSIS_DIR / "rf_classifier_feature_importance.png"
    plot_feature_importance(clf, feature_cols, fi_path)


def run_rf_regressor_next(df, metrics_lines):
    target_col = "vessel_count_next"

    drop_cols = [
        "has_vessel",
        "has_vessel_next",
        "vessel_count_next",
        "iy",
        "ix",
        "t",
        "year",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    cols_for_model = feature_cols + [target_col, "t"]
    df_model = df[cols_for_model].dropna().copy()

    t_cut = df_model["t"].quantile(TRAIN_FRACTION)
    train_mask = df_model["t"] <= t_cut
    df_train = df_model[train_mask].copy()
    df_test = df_model[~train_mask].copy()

    print(
        f"Train period t <= {t_cut:.0f}, "
        f"train n = {len(df_train):d}, test n = {len(df_test):d}"
    )

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )

    print("Running Random Forest regressor for vessel_count_next")
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RF regressor RMSE: {rmse:.3f}")
    print(f"RF regressor R2: {r2:.3f}")

    metrics_lines.append(f"RF regressor vessel_count_next RMSE: {rmse:.3f}")
    metrics_lines.append(f"RF regressor vessel_count_next R2: {r2:.3f}")
    metrics_lines.append("")

    scatter_path = ANALYSIS_DIR / "rf_regressor_scatter.png"
    plot_regressor_scatter(y_test, y_pred, scatter_path)


def main():
    ensure_output_dir()
    df = load_features()
    df = filter_active_cells(df)

    # compute has_vessel for current month if needed
    if "has_vessel" not in df.columns:
        df["has_vessel"] = (df["vessel_count"] > 0).astype(int)

    # probability vs ice in active cells
    eda_path = ANALYSIS_DIR / "eda_prob_vs_ice_active.png"
    plot_prob_vs_ice(df, eda_path)

    metrics_lines = []

    # Code carbon
    tracker = EmissionsTracker(
        project_name="dsan5550_ice_vessels",
        output_dir=str(ANALYSIS_DIR),
        output_file="emissions.csv",
        save_to_file=True,
    )
    print("Start codecarbon tracking")
    tracker.start()

    try:
        run_logistic_next_ice_only(df, metrics_lines)
        run_rf_classifier_next_full(df, metrics_lines)
        run_rf_regressor_next(df, metrics_lines)
    finally:
        emissions_kg = tracker.stop()
        print(
            f"code carbon reports: {emissions_kg:.6f}"
            f"(see {ANALYSIS_DIR / 'emissions.csv'})"
        )
        metrics_lines.append(f"total emissions: {emissions_kg:.6f}")

    metrics_path = ANALYSIS_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        for line in metrics_lines:
            f.write(line.rstrip("\n") + "\n")
    print(f"All metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
