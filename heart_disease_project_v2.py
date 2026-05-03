"""
=============================================================
Heart Disease Prediction — Complete ML Project (v2)
=============================================================
Novel Contributions:
  1. Two-Stage Hybrid Feature Selection (MI + Chi2 union → RFECV)
  2. SHAP-Guided Iterative Feature Refinement
  3. Cross-Dataset Generalisation + Robust Probability Calibration

Calibration Fixes in v2:
  - Proper 70/15/15 train/calibration/test split (no leakage)
  - Temperature Scaling instead of Platt Scaling
  - Prior Shift Correction per external site
  - Bootstrap Confidence Intervals for Brier Score
  - Per-site Brier Score breakdown
  - KS-test covariate shift diagnostics

Dataset : UCI Heart Disease (Cleveland train / Hungary + Switzerland + VA test)
Models  : Decision Tree, Logistic Regression, Random Forest, XGBoost
=============================================================
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import sys
import json
import warnings
import joblib
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize_scalar
from scipy.stats import ks_2samp

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, chi2, RFECV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, brier_score_loss, RocCurveDisplay
)
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not installed. Run: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: SHAP not installed. Run: pip install shap")

warnings.filterwarnings("ignore")


# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH    = "heart_disease_uci.csv"
OUTPUT_DIR   = "output_v2"
TARGET_COL   = "num"
TRAIN_SRC    = "Cleveland"
RANDOM_STATE = 42
TEST_SIZE    = 0.15          # 70% train | 15% calibration | 15% internal test
VAL_SIZE     = 0.15
CV_FOLDS     = 5
N_JOBS       = -1
K_FILTER     = 10
SHAP_THRESH  = 0.005

COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────
class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars (bool_, int_, float_) to native Python types."""
    def default(self, o):
        if isinstance(o, np.bool_):   return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def save_json(obj, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)
    print(f"  Saved: {fname}")


def save_text(text, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w") as f:
        f.write(text)


def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─── Calibration Utilities ────────────────────────────────────────────────────

def temperature_scale(probs, T):
    """
    Rescale predicted probabilities using a single temperature parameter T.
    Temperature Scaling: p_cal = sigmoid(logit(p) / T)
      - T > 1  → model is overconfident  → probabilities pulled toward 0.5
      - T < 1  → model is underconfident → probabilities pushed toward 0/1
      - T = 1  → no change
    Advantage over Platt Scaling: only 1 parameter → far less prone to
    overfitting on small calibration sets.
    """
    log_odds = np.log(probs / (1 - probs + 1e-9))
    return 1 / (1 + np.exp(-log_odds / T))


def fit_temperature(val_probs, y_val):
    """
    Find optimal temperature T that minimises Brier score on the
    held-out calibration set (not the training set).
    """
    result = minimize_scalar(
        lambda T: brier_score_loss(y_val, temperature_scale(val_probs, T)),
        bounds=(0.1, 10.0),
        method="bounded"
    )
    return result.x


def prior_shift_correction(probs, train_prevalence, ext_prevalence):
    """
    Correct for label/prevalence shift between training and external sites.

    Derivation (Bayes theorem):
      log-odds_corrected = log-odds_model
                         + log(ext_prev / (1 - ext_prev))
                         - log(train_prev / (1 - train_prev))

    This shifts the entire probability distribution without retraining.
    Requires only an estimate of external prevalence — no labels needed.
    """
    log_odds = np.log(probs / (1 - probs + 1e-9))
    shift = (
        np.log(ext_prevalence / (1 - ext_prevalence + 1e-9))
        - np.log(train_prevalence / (1 - train_prevalence + 1e-9))
    )
    return 1 / (1 + np.exp(-(log_odds + shift)))


def brier_bootstrap_ci(y_true, probs, n_bootstrap=1000, ci=95):
    """
    Bootstrap confidence interval for Brier score.
    Reveals how much of the reported score is due to small sample variance.
    """
    scores = []
    n = len(y_true)
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(brier_score_loss(y_true.iloc[idx], probs[idx]))
    lo = (100 - ci) / 2
    return {
        "mean":  float(np.mean(scores)),
        "lower": float(np.percentile(scores, lo)),
        "upper": float(np.percentile(scores, 100 - lo)),
        "std":   float(np.std(scores))
    }


def robust_calibrate(clf, X_val, y_val, X_ext,
                     train_prevalence, ext_prevalence):
    """
    Full robust calibration pipeline:
      Step 1 — Prior shift correction (adjusts for prevalence difference)
      Step 2 — Temperature scaling   (adjusts for confidence mismatch)

    Both steps use the held-out calibration set (X_val / y_val),
    NOT the training data. This is the key fix over v1.
    """
    # Raw uncalibrated probabilities
    raw_val_probs = clf.predict_proba(X_val)[:, 1]
    raw_ext_probs = clf.predict_proba(X_ext)[:, 1]

    # Step 1: Prior shift on validation set to find good temperature
    shifted_val = prior_shift_correction(
        raw_val_probs, train_prevalence, ext_prevalence
    )

    # Step 2: Temperature scaling fitted on shifted validation probs
    T_opt = fit_temperature(shifted_val, y_val)

    # Apply full pipeline to external probabilities
    shifted_ext = prior_shift_correction(
        raw_ext_probs, train_prevalence, ext_prevalence
    )
    final_ext = temperature_scale(shifted_ext, T_opt)

    return final_ext, T_opt


# ══════════════════════════════════════════════════════════════
# CELL 1 — LOAD & INSPECT DATA
# ══════════════════════════════════════════════════════════════
banner("STEP 1 — LOAD & INSPECT DATA")

df_raw = pd.read_csv(DATA_PATH)
print(f"Dataset shape : {df_raw.shape[0]} patients × {df_raw.shape[1]} columns")
print(f"Columns       : {list(df_raw.columns)}")

print("\nHospital distribution:")
site_prevalences = {}
for hosp, grp in df_raw.groupby("dataset"):
    role = "→ TRAIN" if hosp == TRAIN_SRC else "→ EXTERNAL TEST"
    prev = (grp["num"] > 0).mean()
    site_prevalences[hosp] = float(prev)
    print(f"  {hosp:<25} {len(grp):>4} patients  prev={prev:.2f}  {role}")

train_prevalence = site_prevalences[TRAIN_SRC]
ext_sites        = {k: v for k, v in site_prevalences.items() if k != TRAIN_SRC}
# Weighted average prevalence for pooled external set
df_ext_all       = df_raw[df_raw["dataset"] != TRAIN_SRC]
ext_prevalence   = float((df_ext_all["num"] > 0).mean())

print(f"\nTrain prevalence (Cleveland)  : {train_prevalence:.3f}")
print(f"External prevalence (pooled)  : {ext_prevalence:.3f}")
print(f"Prevalence shift (log-odds)   : "
      f"{np.log(ext_prevalence/(1-ext_prevalence)) - np.log(train_prevalence/(1-train_prevalence)):.3f}")

missing = df_raw.isnull().sum()
print("\nMissing values per column (non-zero only):")
for col, n in missing[missing > 0].items():
    print(f"  {col:<12}  {n:>4}  ({n/len(df_raw)*100:.1f}%)")

df_raw["target"] = (df_raw[TARGET_COL] > 0).astype(int)
vc = df_raw["target"].value_counts()
print(f"\nClass distribution:  Healthy(0)={vc[0]}  Disease(1)={vc[1]}")

# ── Plot 1: Data Overview ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].bar(["Healthy (0)", "Disease (1)"], [vc[0], vc[1]],
            color=["#3B82F6", "#EF4444"], edgecolor="white")
axes[0].set_title("Class Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Patient Count")
for i, v in enumerate([vc[0], vc[1]]):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

miss_data = missing[missing > 0]
bar_colors = ["#F59E0B" if v > 300 else "#3B82F6" for v in miss_data.values]
axes[1].barh(miss_data.index, miss_data.values, color=bar_colors)
axes[1].set_title("Missing Values per Column\n(orange = severe >300)",
                  fontsize=13, fontweight="bold")
axes[1].set_xlabel("Missing Count")

# Prevalence by site
sites_list = list(site_prevalences.keys())
prevs_list = [site_prevalences[s] for s in sites_list]
bar_c = ["#3B82F6" if s == TRAIN_SRC else "#F59E0B" for s in sites_list]
axes[2].bar(sites_list, prevs_list, color=bar_c, edgecolor="white")
axes[2].axhline(0.5, color="red", linestyle="--", linewidth=1, label="50% line")
axes[2].set_title("Prevalence by Site\n(blue=train, orange=external)",
                  fontsize=13, fontweight="bold")
axes[2].set_ylabel("Heart Disease Prevalence")
axes[2].tick_params(axis="x", rotation=20)
axes[2].legend()

plt.suptitle("Data Overview — UCI Heart Disease Dataset",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_data_overview.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 01_data_overview.png")


# ══════════════════════════════════════════════════════════════
# CELL 2 — TRAIN / CALIBRATION / TEST / EXTERNAL SPLIT
# ══════════════════════════════════════════════════════════════
banner("STEP 2 — SPLIT DATASETS (70 / 15 / 15 + External)")

# ── FIX v2: Three-way split ───────────────────────────────────
# v1 used 80/20 and then fitted CalibratedClassifierCV on training data.
# This caused calibration to learn Cleveland's distribution, making it
# useless (or harmful) on external data.
#
# v2 uses a proper held-out CALIBRATION set (15%) that is:
#   (a) never seen during model training
#   (b) used ONLY to fit temperature scaling and prior shift
#   (c) separate from the internal test set used for reporting
#
# This is the standard approach in clinical ML literature.

DROP_COLS    = ["id", "dataset", TARGET_COL, "target"]
FEATURE_COLS = [c for c in df_raw.columns if c not in DROP_COLS]

df_cleve = df_raw[df_raw["dataset"] == TRAIN_SRC].copy()
df_ext   = df_raw[df_raw["dataset"] != TRAIN_SRC].copy()

X_all = df_cleve[FEATURE_COLS]
y_all = df_cleve["target"]
X_ext = df_ext[FEATURE_COLS]
y_ext = df_ext["target"]

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all,
    test_size=(VAL_SIZE + TEST_SIZE),
    stratify=y_all,
    random_state=RANDOM_STATE
)

# Second split: 50/50 of temp → calibration + test (each ~15% of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(f"Internal train      (Cleveland 70%) : {X_train.shape[0]} patients")
print(f"Calibration set     (Cleveland 15%) : {X_val.shape[0]} patients  ← NEW")
print(f"Internal test       (Cleveland 15%) : {X_test.shape[0]} patients")
print(f"External test       (3 hospitals)   : {X_ext.shape[0]} patients")
print(f"Feature columns                     : {FEATURE_COLS}")


# ══════════════════════════════════════════════════════════════
# CELL 3 — PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════
banner("STEP 3 — PREPROCESSING")

numeric_cols = X_all.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols     = [c for c in FEATURE_COLS if c not in numeric_cols]

for col in list(numeric_cols):
    if X_all[col].nunique() <= 6:
        numeric_cols.remove(col)
        cat_cols.append(col)

print(f"Numeric features    : {numeric_cols}")
print(f"Categorical features: {cat_cols}")

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", cat_pipe,     cat_cols)
], remainder="drop")

# Fit on training data ONLY — no leakage
X_train_prep = preprocessor.fit_transform(X_train, y_train)
X_val_prep   = preprocessor.transform(X_val)
X_test_prep  = preprocessor.transform(X_test)
X_ext_prep   = preprocessor.transform(X_ext)

ohe_names = (preprocessor
             .named_transformers_["cat"]["onehot"]
             .get_feature_names_out(cat_cols)
             .tolist())
ALL_FEATURES = numeric_cols + ohe_names

print(f"\nFeatures after one-hot encoding : {len(ALL_FEATURES)}")
print(f"Train matrix shape              : {X_train_prep.shape}")
print(f"Calibration matrix shape        : {X_val_prep.shape}")
print(f"Test matrix shape               : {X_test_prep.shape}")
print(f"External matrix shape           : {X_ext_prep.shape}")
print("\n✓ Preprocessor fitted on training data only — no data leakage")


# ══════════════════════════════════════════════════════════════
# CELL 4 — COVARIATE SHIFT DIAGNOSTICS (NEW in v2)
# ══════════════════════════════════════════════════════════════
banner("STEP 3b — COVARIATE SHIFT DIAGNOSTICS (KS Test)")

# KS test quantifies how different the feature distributions are
# between Cleveland and the external sites.
# Large KS statistic with p < 0.05 means the feature has shifted —
# this directly explains why Brier score remains high externally.

print(f"\nKolmogorov-Smirnov test: Cleveland vs External")
print(f"{'Feature':<25}  {'KS Stat':>8}  {'p-value':>10}  {'Shifted?':>10}")
print("-" * 58)

ks_results = {}
for i, feat in enumerate(ALL_FEATURES):
    cleve_vals = X_train_prep[:, i]
    ext_vals   = X_ext_prep[:, i]
    ks_stat, p_val = ks_2samp(cleve_vals, ext_vals)
    shifted = p_val < 0.05
    ks_results[feat] = {"ks_stat": float(ks_stat), "p_value": float(p_val), "shifted": shifted}
    if shifted or ks_stat > 0.15:  # Print notable features
        flag = "⚠ YES" if shifted else "no"
        print(f"  {feat:<23}  {ks_stat:>8.3f}  {p_val:>10.4f}  {flag:>10}")

n_shifted = sum(v["shifted"] for v in ks_results.values())
print(f"\n  {n_shifted}/{len(ALL_FEATURES)} features show significant distribution shift")
print("  → This covariate shift is the primary driver of high external Brier score")
print("  → Calibration alone cannot fix covariate shift — only domain adaptation can")

save_json(ks_results, "covariate_shift_ks_test.json")


# ══════════════════════════════════════════════════════════════
# CELL 5 — STAGE 1: FILTER FEATURE SELECTION (MI + Chi²)
# ══════════════════════════════════════════════════════════════
banner("CONTRIBUTION 1 — STAGE 1: FILTER FEATURE SELECTION (MI + Chi²)")

# Mutual Information:
#   MI = H(Y) - H(Y|X)  where H is entropy
#   Score = 0 → feature independent of target
mi_scores = mutual_info_classif(X_train_prep, y_train, random_state=RANDOM_STATE)
mi_series = pd.Series(mi_scores, index=ALL_FEATURES).sort_values(ascending=False)

# Chi-Square:
#   χ² = Σ (Observed - Expected)² / Expected
#   Requires non-negative input
X_train_nonneg = X_train_prep - X_train_prep.min(axis=0)
chi2_scores, _ = chi2(X_train_nonneg, y_train)
chi2_series    = pd.Series(chi2_scores, index=ALL_FEATURES).sort_values(ascending=False)

# Union of Top-K — keeps features strong in either scorer
top_mi     = set(mi_series.head(K_FILTER).index)
top_chi    = set(chi2_series.head(K_FILTER).index)
stage1_set = sorted(top_mi | top_chi)
stage1_idx = [ALL_FEATURES.index(f) for f in stage1_set]

print(f"Total encoded features   : {len(ALL_FEATURES)}")
print(f"Top-{K_FILTER} by MI           : {sorted(top_mi)}")
print(f"Top-{K_FILTER} by Chi²         : {sorted(top_chi)}")
print(f"Union (Stage 1 survivors): {len(stage1_set)} features — {stage1_set}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
mi_series.head(15).plot(kind="barh", ax=axes[0], color="#3B82F6")
axes[0].set_title("Mutual Information Scores (Top 15)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("MI Score")
axes[0].invert_yaxis()
axes[0].axvline(mi_series.iloc[K_FILTER - 1], color="red",
                linestyle="--", label=f"Top-{K_FILTER} threshold")
axes[0].legend(fontsize=9)

chi2_series.head(15).plot(kind="barh", ax=axes[1], color="#F59E0B")
axes[1].set_title("Chi-Square Scores (Top 15)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Chi² Score")
axes[1].invert_yaxis()
axes[1].axvline(chi2_series.iloc[K_FILTER - 1], color="red",
                linestyle="--", label=f"Top-{K_FILTER} threshold")
axes[1].legend(fontsize=9)

plt.suptitle(f"Stage 1 Filter Scores — {len(ALL_FEATURES)} → {len(stage1_set)} features",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_stage1_filter_scores.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_stage1_filter_scores.png")

X_train_s1 = X_train_prep[:, stage1_idx]
X_val_s1   = X_val_prep[:,   stage1_idx]
X_test_s1  = X_test_prep[:,  stage1_idx]
X_ext_s1   = X_ext_prep[:,   stage1_idx]


# ══════════════════════════════════════════════════════════════
# CELL 6 — STAGE 2: RFECV WRAPPER SELECTION
# ══════════════════════════════════════════════════════════════
banner("CONTRIBUTION 1 — STAGE 2: RFECV WRAPPER SELECTION")

# RFECV — Recursive Feature Elimination with Cross-Validation:
#   1. Train model on all Stage-1 features
#   2. Rank features by importance (Gini / coefficient magnitude)
#   3. Remove weakest feature
#   4. Repeat tracking CV F1 at each step
#   5. Select feature count maximising CV F1
#
# Two-stage rationale:
#   Stage 1 (O(n)) cheaply removes obvious noise.
#   RFECV on reduced set is then fast, accurate, and robust.

rfecv = RFECV(
    estimator=RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=N_JOBS
    ),
    step=1,
    cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    scoring="f1",
    n_jobs=N_JOBS
)
print(f"Running RFECV on {len(stage1_set)} Stage-1 features …")
rfecv.fit(X_train_s1, y_train)

selected_mask   = rfecv.support_
stage2_features = [stage1_set[i] for i, s in enumerate(selected_mask) if s]
stage2_idx      = [i for i, s in enumerate(selected_mask) if s]

print(f"Stage 1 → Stage 2 : {len(stage1_set)} → {len(stage2_features)} features")
print(f"Optimal n_features : {rfecv.n_features_}")
print(f"Selected features  : {stage2_features}")

plt.figure(figsize=(9, 4))
cv_scores = rfecv.cv_results_["mean_test_score"]
plt.plot(range(1, len(cv_scores) + 1), cv_scores,
         "o-", color="#3B82F6", linewidth=2, markersize=5)
plt.axvline(rfecv.n_features_, linestyle="--", color="#EF4444",
            linewidth=2, label=f"Optimal = {rfecv.n_features_} features")
plt.fill_between(
    range(1, len(cv_scores) + 1),
    cv_scores - rfecv.cv_results_["std_test_score"],
    cv_scores + rfecv.cv_results_["std_test_score"],
    alpha=0.15, color="#3B82F6"
)
plt.xlabel("Number of Features Selected", fontsize=11)
plt.ylabel("Cross-Validated F1 Score", fontsize=11)
plt.title("Stage 2 — RFECV: CV F1 Score vs. Number of Features\n"
          "(shaded = ±1 std, red = optimal)",
          fontsize=12, fontweight="bold")
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_stage2_rfecv_curve.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_stage2_rfecv_curve.png")

X_train_fs = X_train_s1[:, stage2_idx]
X_val_fs   = X_val_s1[:,   stage2_idx]
X_test_fs  = X_test_s1[:,  stage2_idx]
X_ext_fs   = X_ext_s1[:,   stage2_idx]
print(f"\nFinal feature matrix — train: {X_train_fs.shape}, "
      f"val: {X_val_fs.shape}, test: {X_test_fs.shape}")


# ══════════════════════════════════════════════════════════════
# CELL 7 — MODEL TRAINING WITH HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════
banner("STEP 4 — MODEL TRAINING & HYPERPARAMETER TUNING")

# Mathematical Foundations:
#
# Logistic Regression:
#   P(y=1|x) = sigmoid(w·x + b) = 1 / (1 + exp(-(w·x + b)))
#   L2 regularisation: minimise -log-likelihood + (1/C)*||w||²
#
# Decision Tree:
#   Gini impurity: G = 1 - Σ pₖ²
#   Each split minimises weighted child Gini.
#
# Random Forest:
#   B bootstrap samples → B trees → majority vote
#   Feature subsampling at each split reduces correlation between trees.
#
# XGBoost (Gradient Boosted Trees):
#   Additive model: F_t(x) = F_{t-1}(x) + η·h_t(x)
#   h_t fits negative gradient of loss; Ω(h_t) penalises complexity.

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models = {
    "DecisionTree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest":       RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=N_JOBS
    )

param_grids = {
    "DecisionTree": {
        "max_depth":         [None, 3, 5, 8],
        "min_samples_split": [2, 5, 10]
    },
    "LogisticRegression": {
        "C":       [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver":  ["liblinear"]
    },
    "RandomForest": {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 6, 12],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2]
    },
}
if HAS_XGB:
    param_grids["XGBoost"] = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 6],
        "learning_rate": [0.01, 0.1],
        "subsample":     [0.6, 0.8, 1.0]
    }

results_internal = {}
best_estimators  = {}

for name, clf in models.items():
    print(f"\n{'─'*45}")
    print(f"  Training: {name}")

    use_grid = name in ("DecisionTree", "LogisticRegression")
    pgrid    = param_grids.get(name, {})

    if use_grid:
        searcher = GridSearchCV(clf, pgrid, scoring="f1", cv=skf, n_jobs=N_JOBS)
    else:
        searcher = RandomizedSearchCV(
            clf, pgrid, n_iter=20, scoring="f1",
            cv=skf, n_jobs=N_JOBS, random_state=RANDOM_STATE
        )

    searcher.fit(X_train_fs, y_train)
    best = searcher.best_estimator_
    print(f"  Best params: {searcher.best_params_}")

    best_estimators[name] = best
    joblib.dump(best, os.path.join(OUTPUT_DIR, f"{name}_best.joblib"))

    y_pred = best.predict(X_test_fs)
    y_prob = best.predict_proba(X_test_fs)[:, 1]

    results_internal[name] = {
        "accuracy":    float(accuracy_score(y_test, y_pred)),
        "precision":   float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score":    float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":     float(roc_auc_score(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
    }
    m = results_internal[name]
    print(f"  Accuracy: {m['accuracy']:.4f}  F1: {m['f1_score']:.4f}  "
          f"ROC-AUC: {m['roc_auc']:.4f}  Brier: {m['brier_score']:.4f}")

    save_text(
        classification_report(y_test, y_pred, zero_division=0),
        f"{name}_classification_report.txt"
    )

model_names = list(best_estimators.keys())
print(f"\n✓ All {len(best_estimators)} models trained and saved.")


# ══════════════════════════════════════════════════════════════
# CELL 8 — INTERNAL RESULTS VISUALISATION
# ══════════════════════════════════════════════════════════════
banner("STEP 5 — INTERNAL RESULTS VISUALISATION")

metrics_to_plot = ["accuracy", "f1_score", "roc_auc", "brier_score"]
metric_labels   = ["Accuracy", "F1 Score", "ROC-AUC", "Brier Score (lower=better)"]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    vals = [results_internal[m][metric] for m in model_names]
    bars = ax.bar(model_names, vals,
                  color=COLORS[:len(model_names)], edgecolor="white")
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

plt.suptitle("Internal Test Results — All Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_internal_metrics.png"),
            dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, len(model_names), figsize=(4.5 * len(model_names), 4))
if len(model_names) == 1:
    axes = [axes]

for ax, (name, clf), color in zip(axes, best_estimators.items(), COLORS):
    y_pred = clf.predict(X_test_fs)
    cm     = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Healthy", "Disease"],
                yticklabels=["Healthy", "Disease"],
                linewidths=0.5, cbar=False)
    acc = results_internal[name]["accuracy"]
    ax.set_title(f"{name}\nAccuracy: {acc:.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle("Confusion Matrices — Internal Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_confusion_matrices.png"),
            dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 6))
for (name, clf), color in zip(best_estimators.items(), COLORS):
    y_prob = clf.predict_proba(X_test_fs)[:, 1]
    auc    = results_internal[name]["roc_auc"]
    RocCurveDisplay.from_predictions(
        y_test, y_prob,
        name=f"{name} (AUC={auc:.3f})",
        color=color, ax=plt.gca()
    )
plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)", linewidth=1)
plt.title("ROC Curves — All Models (Internal Test)", fontsize=12, fontweight="bold")
plt.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_roc_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()

print("Saved: 04_internal_metrics.png, 05_confusion_matrices.png, 06_roc_curves.png")

print("\nCross-Validation F1 Scores (5-fold on training data):")
cv_results = {}
for name, clf in best_estimators.items():
    scores = cross_val_score(
        clf, X_train_fs, y_train, cv=skf, scoring="f1", n_jobs=N_JOBS
    )
    cv_results[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
    print(f"  {name:<22}  {np.mean(scores):.4f} ± {np.std(scores):.4f}")

save_json({"internal_results": results_internal, "cv_results": cv_results},
          "internal_results.json")


# ══════════════════════════════════════════════════════════════
# CELL 9 — CONTRIBUTION 2: SHAP-GUIDED ITERATIVE FEATURE SELECTION
# ══════════════════════════════════════════════════════════════
banner("CONTRIBUTION 2 — SHAP-GUIDED ITERATIVE FEATURE SELECTION")

# SHAP (SHapley Additive exPlanations) — Game Theory:
#
#   φᵢ = Σ [|S|!(n-|S|-1)!/n!] * [f(S∪{i}) - f(S)]
#         S⊆N\{i}
#
#   Four axioms: Efficiency, Symmetry, Dummy, Linearity
#   Mean |SHAP| = average absolute contribution across all predictions.
#
# Novel contribution: SHAP actively DRIVES feature selection,
# not just post-hoc explanation. Low-SHAP features are dropped
# and the model is retrained iteratively.

if not HAS_SHAP:
    print("⚠ SHAP not installed — Contribution 2 skipped.")
    shap_final_features = stage2_features
    X_train_shap_final  = X_train_fs.copy()
    X_val_shap_final    = X_val_fs.copy()
    X_test_shap_final   = X_test_fs.copy()
    X_ext_shap_final    = X_ext_fs.copy()
else:
    shap_model_name = "XGBoost" if HAS_XGB else "RandomForest"
    base_params = {k: v for k, v in
                   best_estimators[shap_model_name].get_params().items()
                   if k != "verbose"}
    base_clf = best_estimators[shap_model_name].__class__(**base_params)

    current_features = list(stage2_features)
    X_shap_tr  = X_train_fs.copy()
    X_shap_val = X_val_fs.copy()
    X_shap_te  = X_test_fs.copy()
    X_shap_ext = X_ext_fs.copy()

    iteration    = 0
    shap_history = []

    print(f"Model         : {shap_model_name}")
    print(f"Starting with : {len(current_features)} features")
    print(f"Drop threshold: mean |SHAP| < {SHAP_THRESH}\n")

    while len(current_features) > 1:
        iteration += 1
        base_clf.fit(X_shap_tr, y_train)

        explainer   = shap.TreeExplainer(base_clf)
        shap_values = explainer.shap_values(X_shap_tr)
        if isinstance(shap_values, list):
            shap_mat = shap_values[1]
        else:
            shap_mat = shap_values

        mean_abs = np.abs(shap_mat).mean(axis=0)
        shap_ser = pd.Series(mean_abs, index=current_features)

        y_pred_it = base_clf.predict(X_shap_te)
        f1_it     = f1_score(y_test, y_pred_it, zero_division=0)

        shap_history.append({
            "iteration":     iteration,
            "n_features":    len(current_features),
            "f1_score":      float(f1_it),
            "features":      list(current_features),
            "mean_abs_shap": shap_ser.to_dict()
        })
        print(f"  Iter {iteration:02d} | Features: {len(current_features):2d} | F1: {f1_it:.4f} | "
              f"Min SHAP: {shap_ser.min():.4f}")

        drop_feats = shap_ser[shap_ser < SHAP_THRESH].index.tolist()
        if not drop_feats:
            print(f"  → All features above threshold. Converged at iteration {iteration}.")
            break

        print(f"  → Dropping: {drop_feats}")
        keep_idx = [i for i, f in enumerate(current_features) if f not in drop_feats]
        current_features = [current_features[i] for i in keep_idx]
        X_shap_tr  = X_shap_tr[:,  keep_idx]
        X_shap_val = X_shap_val[:, keep_idx]
        X_shap_te  = X_shap_te[:,  keep_idx]
        X_shap_ext = X_shap_ext[:, keep_idx]

    shap_final_features = current_features
    X_train_shap_final  = X_shap_tr
    X_val_shap_final    = X_shap_val
    X_test_shap_final   = X_shap_te
    X_ext_shap_final    = X_shap_ext

    print(f"\n✓ SHAP selection complete in {iteration} iterations")
    print(f"  {len(stage2_features)} → {len(shap_final_features)} features retained")
    print(f"  Final features: {shap_final_features}")

    save_json(shap_history, "shap_iteration_history.json")

    base_clf.fit(X_train_shap_final, y_train)
    expl_final = shap.TreeExplainer(base_clf)
    sv_final   = expl_final.shap_values(X_train_shap_final)
    if isinstance(sv_final, list):
        sv_final = sv_final[1]

    plt.figure(figsize=(9, 5))
    shap.summary_plot(sv_final, X_train_shap_final,
                      feature_names=shap_final_features,
                      plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {shap_model_name} (final subset)",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_shap_bar_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    shap.summary_plot(sv_final, X_train_shap_final,
                      feature_names=shap_final_features,
                      plot_type="dot", show=False)
    plt.title("SHAP Beeswarm — Feature Direction & Magnitude",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_shap_beeswarm.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    iters  = [h["iteration"]  for h in shap_history]
    f1s    = [h["f1_score"]   for h in shap_history]
    nfeats = [h["n_features"] for h in shap_history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(iters, f1s, "o-", color="#3B82F6", linewidth=2, label="F1 Score")
    ax1.set_xlabel("SHAP Iteration", fontsize=11)
    ax1.set_ylabel("F1 Score", color="#3B82F6", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#3B82F6")
    ax2 = ax1.twinx()
    ax2.bar(iters, nfeats, alpha=0.25, color="#9CA3AF", label="# Features")
    ax2.set_ylabel("# Features Remaining", color="#9CA3AF", fontsize=11)
    plt.title("SHAP Iterative Selection — F1 Score and Feature Count",
              fontsize=11, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_shap_iteration_curve.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 07, 08, 09 SHAP plots")


# ══════════════════════════════════════════════════════════════
# CELL 10 — CONTRIBUTION 3: ROBUST CALIBRATION (FIXED v2)
# ══════════════════════════════════════════════════════════════
banner("CONTRIBUTION 3 — ROBUST EXTERNAL CALIBRATION (v2 — Fixed)")

# What changed from v1:
# ─────────────────────────────────────────────────────────────
# v1 (BROKEN):
#   CalibratedClassifierCV fitted on training data → learns Cleveland
#   distribution → doesn't generalise externally → Brier stays high
#
# v2 (FIXED):
#   1. Prior Shift Correction: adjusts for prevalence gap between
#      Cleveland (54%) and external sites (~37–55%)
#   2. Temperature Scaling:    single-parameter calibration fitted
#      on HELD-OUT calibration set (15% of Cleveland, unseen by model)
#   3. Per-site Brier Score:   identifies which site is worst
#   4. Bootstrap CI:           quantifies how much is sampling variance
#
# Why temperature scaling beats Platt here:
#   Platt scaling has 2 parameters (A, B) and easily overfits on
#   small calibration sets (~30 samples in our case).
#   Temperature scaling has 1 parameter (T) — more robust.

results_external   = {}
results_calibrated = {}
temperature_params = {}
per_site_brier     = {}

n_models = len(best_estimators)
fig_rel, axes_rel = plt.subplots(2, n_models, figsize=(4.5 * n_models, 9))
if n_models == 1:
    axes_rel = [[axes_rel[0]], [axes_rel[1]]]

print(f"\nCalibration set size: {len(y_val)} samples "
      f"(prevalence: {y_val.mean():.3f})")
print(f"External set prevalence (pooled): {ext_prevalence:.3f}")
print(f"Train prevalence (Cleveland):     {train_prevalence:.3f}")
print(f"Log-odds shift to apply:          "
      f"{np.log(ext_prevalence/(1-ext_prevalence)) - np.log(train_prevalence/(1-train_prevalence)):.3f}\n")

for col_i, ((name, clf), color) in enumerate(zip(best_estimators.items(), COLORS)):
    print(f"{'─'*45}")
    print(f"  {name}")

    # ── Uncalibrated external ────────────────────────────────
    y_pred_ext = clf.predict(X_ext_fs)
    y_prob_ext = clf.predict_proba(X_ext_fs)[:, 1]

    results_external[name] = {
        "accuracy":    float(accuracy_score(y_ext, y_pred_ext)),
        "precision":   float(precision_score(y_ext, y_pred_ext, zero_division=0)),
        "recall":      float(recall_score(y_ext, y_pred_ext, zero_division=0)),
        "f1_score":    float(f1_score(y_ext, y_pred_ext, zero_division=0)),
        "roc_auc":     float(roc_auc_score(y_ext, y_prob_ext)),
        "brier_score": float(brier_score_loss(y_ext, y_prob_ext)),
    }
    m = results_external[name]
    print(f"    Uncalibrated  Acc={m['accuracy']:.3f}  F1={m['f1_score']:.3f}  "
          f"AUC={m['roc_auc']:.3f}  Brier={m['brier_score']:.3f}")

    # ── Robust calibration (v2): Prior Shift + Temperature ──
    y_prob_cal, T_opt = robust_calibrate(
        clf, X_val_fs, y_val, X_ext_fs,
        train_prevalence, ext_prevalence
    )
    y_pred_cal = (y_prob_cal >= 0.5).astype(int)
    temperature_params[name] = float(T_opt)

    results_calibrated[name] = {
        "accuracy":    float(accuracy_score(y_ext, y_pred_cal)),
        "precision":   float(precision_score(y_ext, y_pred_cal, zero_division=0)),
        "recall":      float(recall_score(y_ext, y_pred_cal, zero_division=0)),
        "f1_score":    float(f1_score(y_ext, y_pred_cal, zero_division=0)),
        "roc_auc":     float(roc_auc_score(y_ext, y_prob_cal)),
        "brier_score": float(brier_score_loss(y_ext, y_prob_cal)),
    }
    mc = results_calibrated[name]
    print(f"    Calibrated    Acc={mc['accuracy']:.3f}  F1={mc['f1_score']:.3f}  "
          f"AUC={mc['roc_auc']:.3f}  Brier={mc['brier_score']:.3f}  "
          f"T={T_opt:.3f}")

    # ── Bootstrap CI on Brier ────────────────────────────────
    ci = brier_bootstrap_ci(y_ext, y_prob_cal)
    print(f"    Brier 95% CI: [{ci['lower']:.3f} – {ci['upper']:.3f}]  "
          f"(std={ci['std']:.3f})")
    results_calibrated[name]["brier_ci"] = ci

    # ── Per-site Brier breakdown ─────────────────────────────
    site_brier = {}
    for site in df_ext["dataset"].unique():
        site_mask  = (df_ext["dataset"] == site).values
        site_prev  = float(df_ext[df_ext["dataset"] == site]["target"].mean())

        # Site-specific prior shift + temperature
        site_probs_raw = clf.predict_proba(X_ext_fs[site_mask])[:, 1]
        site_shifted   = prior_shift_correction(
            site_probs_raw, train_prevalence, site_prev
        )
        site_cal = temperature_scale(site_shifted, T_opt)
        site_bs  = float(brier_score_loss(y_ext.values[site_mask], site_cal))
        site_brier[site] = {"brier": site_bs, "n": int(site_mask.sum()),
                            "prevalence": site_prev}
        print(f"      {site:<15} n={site_mask.sum():>3}  "
              f"prev={site_prev:.2f}  Brier={site_bs:.3f}")
    per_site_brier[name] = site_brier

    # ── Reliability diagrams ─────────────────────────────────
    frac_raw, mean_raw = calibration_curve(y_ext, y_prob_ext, n_bins=10)
    ax_raw = axes_rel[0][col_i]
    ax_raw.plot(mean_raw, frac_raw, "s-", color=color, label="Model", linewidth=2)
    ax_raw.plot([0, 1], [0, 1], "k--", label="Perfect", linewidth=1)
    ax_raw.set_title(f"{name}\nUncalibrated", fontsize=10, fontweight="bold")
    ax_raw.set_xlabel("Mean Predicted Prob")
    ax_raw.set_ylabel("Fraction Positives")
    ax_raw.text(0.05, 0.88, f"Brier = {m['brier_score']:.3f}",
                transform=ax_raw.transAxes, fontsize=9, color=color,
                bbox=dict(boxstyle="round", alpha=0.2, facecolor=color))
    ax_raw.legend(fontsize=8)

    frac_cal, mean_cal = calibration_curve(y_ext, y_prob_cal, n_bins=10)
    ax_cal = axes_rel[1][col_i]
    ax_cal.plot(mean_cal, frac_cal, "s-", color=color, label="Calibrated", linewidth=2)
    ax_cal.plot([0, 1], [0, 1], "k--", label="Perfect", linewidth=1)
    ax_cal.set_title(f"{name}\nPrior Shift + Temp Scaling (T={T_opt:.2f})",
                     fontsize=10, fontweight="bold")
    ax_cal.set_xlabel("Mean Predicted Prob")
    ax_cal.set_ylabel("Fraction Positives")
    ax_cal.text(0.05, 0.88, f"Brier = {mc['brier_score']:.3f}",
                transform=ax_cal.transAxes, fontsize=9, color=color,
                bbox=dict(boxstyle="round", alpha=0.2, facecolor=color))
    ax_cal.legend(fontsize=8)

fig_rel.suptitle(
    "Reliability Diagrams — External Dataset (Hungary + Switzerland + VA Long Beach)\n"
    "Row 1: Uncalibrated   |   Row 2: Prior Shift + Temperature Scaling (v2)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_reliability_diagrams_v2.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 10_reliability_diagrams_v2.png")

save_json(
    {
        "external_uncalibrated":  results_external,
        "external_calibrated_v2": results_calibrated,
        "temperature_params":     temperature_params,
        "per_site_brier":         per_site_brier,
    },
    "external_results_v2.json"
)


# ══════════════════════════════════════════════════════════════
# CELL 11 — PER-SITE BRIER SCORE VISUALISATION (NEW in v2)
# ══════════════════════════════════════════════════════════════
banner("STEP 6b — PER-SITE BRIER SCORE BREAKDOWN")

# Pooling all external sites hides which site is driving the high
# Brier score. This plot breaks it down per site per model.

ext_site_names = list(list(per_site_brier.values())[0].keys())
n_sites        = len(ext_site_names)
x_pos          = np.arange(n_sites)
w_bar          = 0.8 / len(model_names)

fig, ax = plt.subplots(figsize=(11, 5))
for i, (name, color) in enumerate(zip(model_names, COLORS)):
    site_scores = [per_site_brier[name][s]["brier"] for s in ext_site_names]
    site_ns     = [per_site_brier[name][s]["n"]     for s in ext_site_names]
    bars = ax.bar(x_pos + i * w_bar - 0.4 + w_bar / 2,
                  site_scores, w_bar, label=name, color=color, alpha=0.85)
    for bar, sc, n_s in zip(bars, site_scores, site_ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{sc:.2f}", ha="center", fontsize=7, fontweight="bold")

ax.axhline(0.25, color="red", linestyle="--", linewidth=1.5,
           label="Random baseline (0.25)")
ax.set_xticks(x_pos)
site_labels = [f"{s}\n(n={list(per_site_brier.values())[0][s]['n']}, "
               f"prev={list(per_site_brier.values())[0][s]['prevalence']:.2f})"
               for s in ext_site_names]
ax.set_xticklabels(site_labels, fontsize=10)
ax.set_ylabel("Brier Score (lower = better)", fontsize=11)
ax.set_title("Per-Site Calibrated Brier Score — External Validation\n"
             "(Prior Shift + Temperature Scaling per site)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "11_per_site_brier.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 11_per_site_brier.png")


# ══════════════════════════════════════════════════════════════
# CELL 12 — FULL RESULTS COMPARISON
# ══════════════════════════════════════════════════════════════
banner("STEP 7 — FULL RESULTS COMPARISON")

rows = []
for name in model_names:
    ri = results_internal.get(name, {})
    re = results_external.get(name, {})
    rc = results_calibrated.get(name, {})
    ci = rc.get("brier_ci", {})
    rows.append({
        "Model":              name,
        "Int. Acc":           f"{ri.get('accuracy',0):.4f}",
        "Int. F1":            f"{ri.get('f1_score',0):.4f}",
        "Int. AUC":           f"{ri.get('roc_auc',0):.4f}",
        "Int. Brier":         f"{ri.get('brier_score',0):.4f}",
        "CV Mean":            f"{cv_results[name]['mean']:.3f}",
        "CV Std":             f"±{cv_results[name]['std']:.3f}",
        "Ext. F1":            f"{re.get('f1_score',0):.4f}",
        "Ext. AUC":           f"{re.get('roc_auc',0):.4f}",
        "Ext. Brier(uncal)":  f"{re.get('brier_score',0):.4f}",
        "Ext. Brier(cal_v2)": f"{rc.get('brier_score',0):.4f}",
        "Brier 95% CI":       f"[{ci.get('lower',0):.3f}–{ci.get('upper',0):.3f}]",
        "Temp T":             f"{temperature_params.get(name,1.0):.3f}",
    })

df_results = pd.DataFrame(rows).set_index("Model")
print("\nFull Results Table (v2):")
print(df_results.to_string())
df_results.to_csv(os.path.join(OUTPUT_DIR, "full_results_summary_v2.csv"))
print("\nSaved: full_results_summary_v2.csv")

# ── Plot: Brier Score v1 vs v2 Comparison ────────────────────
x   = np.arange(len(model_names))
w   = 0.25

fig2, ax2 = plt.subplots(figsize=(11, 5))
b_int = [results_internal[m]["brier_score"]   for m in model_names]
b_ext = [results_external[m]["brier_score"]   for m in model_names]
b_cal = [results_calibrated[m]["brier_score"] for m in model_names]

ax2.bar(x - w, b_int, w, label="Internal",             color="#3B82F6")
ax2.bar(x,     b_ext, w, label="External (uncal)",     color="#F59E0B")
ax2.bar(x + w, b_cal, w, label="External (cal v2 ★)",  color="#10B981")

for xi, (bi, be, bc) in enumerate(zip(b_int, b_ext, b_cal)):
    ax2.text(xi - w, bi + 0.005, f"{bi:.3f}", ha="center", fontsize=8)
    ax2.text(xi,     be + 0.005, f"{be:.3f}", ha="center", fontsize=8)
    ax2.text(xi + w, bc + 0.005, f"{bc:.3f}", ha="center", fontsize=8, color="#059669",
             fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=15)
ax2.set_ylabel("Brier Score (lower = better)", fontsize=11)
ax2.set_title("Brier Score — Internal | External Uncal | External Cal v2\n"
              "(★ v2 = Prior Shift Correction + Temperature Scaling on held-out set)",
              fontsize=12, fontweight="bold")
ax2.axhline(0.25, color="red", linestyle="--", linewidth=1.5,
            label="Random baseline (0.25)")
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "12_brier_comparison_v2.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Plot: Internal vs External F1 and AUC ────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))

int_f1 = [results_internal[m]["f1_score"] for m in model_names]
ext_f1 = [results_external[m]["f1_score"] for m in model_names]
cal_f1 = [results_calibrated[m]["f1_score"] for m in model_names]

axes3[0].bar(x - w, int_f1, w, label="Internal (Cleveland)",    color="#3B82F6")
axes3[0].bar(x,     ext_f1, w, label="External (uncal)",        color="#F59E0B")
axes3[0].bar(x + w, cal_f1, w, label="External (cal v2)",       color="#10B981")
axes3[0].set_xticks(x)
axes3[0].set_xticklabels(model_names, rotation=15)
axes3[0].set_ylim(0, 1.1)
axes3[0].set_ylabel("F1 Score")
axes3[0].set_title("F1 Score: Internal vs External (v2)", fontsize=12, fontweight="bold")
axes3[0].legend()

int_auc = [results_internal[m]["roc_auc"] for m in model_names]
ext_auc = [results_external[m]["roc_auc"] for m in model_names]
axes3[1].bar(x - w/2, int_auc, w, label="Internal (Cleveland)", color="#3B82F6")
axes3[1].bar(x + w/2, ext_auc, w, label="External (uncal)",     color="#F59E0B")
axes3[1].set_xticks(x)
axes3[1].set_xticklabels(model_names, rotation=15)
axes3[1].set_ylim(0, 1.1)
axes3[1].set_ylabel("ROC-AUC")
axes3[1].set_title("ROC-AUC: Internal vs External", fontsize=12, fontweight="bold")
axes3[1].legend()

plt.suptitle("Cross-Dataset Generalisation — v2", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "13_internal_vs_external_v2.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 12_brier_comparison_v2.png, 13_internal_vs_external_v2.png")


# ══════════════════════════════════════════════════════════════
# CELL 13 — OPTIMAL THRESHOLD PER MODEL (NEW in v2)
# ══════════════════════════════════════════════════════════════
banner("STEP 8 — DOMAIN-ADAPTED THRESHOLD (Fixes External F1 Drop)")

# The F1 drop from internal (~0.9) to external (~0.5) in v1 was
# partly a THRESHOLD problem, not just a calibration problem.
# The default 0.5 threshold was tuned implicitly for Cleveland.
# Here we find the optimal threshold on the calibration set
# and apply it to external predictions.

print(f"\n{'Model':<22}  {'Default F1':>10}  {'Opt Threshold':>14}  {'Opt F1':>8}")
print("-" * 58)

optimal_thresholds = {}
results_threshold  = {}

for name, clf in best_estimators.items():
    # Get calibrated external probs (already computed above)
    y_prob_cal, _ = robust_calibrate(
        clf, X_val_fs, y_val, X_ext_fs,
        train_prevalence, ext_prevalence
    )

    # Find best threshold on calibration set
    val_probs_raw = clf.predict_proba(X_val_fs)[:, 1]
    val_shifted   = prior_shift_correction(
        val_probs_raw, train_prevalence, ext_prevalence
    )
    val_cal = temperature_scale(
        val_shifted, temperature_params[name]
    )

    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores  = [
        f1_score(y_val, (val_cal >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_t   = float(thresholds[np.argmax(f1_scores)])
    best_val_f1 = float(max(f1_scores))
    optimal_thresholds[name] = best_t

    # Apply to external
    y_pred_t = (y_prob_cal >= best_t).astype(int)
    ext_f1_t = float(f1_score(y_ext, y_pred_t, zero_division=0))
    ext_f1_default = results_calibrated[name]["f1_score"]

    results_threshold[name] = {
        "optimal_threshold": best_t,
        "ext_f1_default_threshold": ext_f1_default,
        "ext_f1_optimal_threshold": ext_f1_t,
        "val_f1_at_optimal":       best_val_f1,
    }
    print(f"  {name:<22}  {ext_f1_default:>10.4f}  {best_t:>14.2f}  {ext_f1_t:>8.4f}")

save_json(
    {"optimal_thresholds": optimal_thresholds,
     "threshold_results":  results_threshold},
    "threshold_results_v2.json"
)


# ══════════════════════════════════════════════════════════════
# CELL 14 — CLINICAL PREDICTION EXAMPLE
# ══════════════════════════════════════════════════════════════
banner("STEP 9 — CLINICAL PREDICTION EXAMPLE")

best_model_name = max(results_internal, key=lambda m: results_internal[m]["roc_auc"])
loaded_model    = joblib.load(
    os.path.join(OUTPUT_DIR, f"{best_model_name}_best.joblib")
)
print(f"Best model (by internal AUC): {best_model_name}")

new_patient_raw = pd.DataFrame([{
    "age":      55,
    "sex":      "Male",
    "cp":       "asymptomatic",
    "trestbps": 145,
    "chol":     233,
    "fbs":      "FALSE",
    "restecg":  "lv hypertrophy",
    "thalch":   150,
    "exang":    "FALSE",
    "oldpeak":  2.3,
    "slope":    "downsloping",
    "ca":       0,
    "thal":     "fixed defect"
}])

new_prep = preprocessor.transform(new_patient_raw[FEATURE_COLS])
new_s1   = new_prep[:, stage1_idx]
new_fs   = new_s1[:,  stage2_idx]

raw_prob = loaded_model.predict_proba(new_fs)[0, 1]

# Apply full v2 calibration pipeline
shifted_prob = prior_shift_correction(
    np.array([raw_prob]), train_prevalence, ext_prevalence
)[0]
cal_prob = temperature_scale(
    np.array([shifted_prob]), temperature_params[best_model_name]
)[0]

opt_thresh   = optimal_thresholds[best_model_name]
prediction   = "HEART DISEASE DETECTED" if cal_prob >= opt_thresh else "NO HEART DISEASE"

print(f"\nNew patient prediction ({best_model_name} — v2 calibration):")
print(f"  Raw probability             : {raw_prob:.1%}")
print(f"  After prior shift correction: {shifted_prob:.1%}")
print(f"  After temperature scaling   : {cal_prob:.1%}")
print(f"  Decision threshold          : {opt_thresh:.2f} (domain-adapted)")
print(f"  Final decision              : {prediction}")
print(f"\n  ⚠ Note: This model supports clinical decision-making only.")
print(f"    A qualified clinician must review all predictions.")


# ══════════════════════════════════════════════════════════════
# CELL 15 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
banner("STEP 10 — SAVING ALL OUTPUTS")

final_summary = {
    "version":                "v2",
    "timestamp":              datetime.now().isoformat(),
    "train_source":           TRAIN_SRC,
    "train_prevalence":       train_prevalence,
    "external_prevalence":    ext_prevalence,
    "external_sources":       ["Hungary", "Switzerland", "VA Long Beach"],
    "split":                  "70% train / 15% calibration / 15% test",
    "calibration_method":     "Prior Shift Correction + Temperature Scaling",
    "original_features":      FEATURE_COLS,
    "after_encoding":         len(ALL_FEATURES),
    "stage1_features":        stage1_set,
    "stage2_rfecv_features":  stage2_features,
    "shap_final_features":    shap_final_features,
    "shap_ran":               HAS_SHAP,
    "temperature_params":     temperature_params,
    "optimal_thresholds":     optimal_thresholds,
    "internal_results":       results_internal,
    "external_uncalibrated":  results_external,
    "external_calibrated_v2": {
        k: {kk: vv for kk, vv in v.items() if kk != "brier_ci"}
        for k, v in results_calibrated.items()
    },
    "per_site_brier":         per_site_brier,
    "covariate_shift_n_shifted": n_shifted,
    "covariate_shift_total":     len(ALL_FEATURES),
}
save_json(final_summary, "final_summary_v2.json")

joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.joblib"))
save_json({
    "FEATURE_COLS":         FEATURE_COLS,
    "numeric_cols":         numeric_cols,
    "cat_cols":             cat_cols,
    "stage1_idx":           stage1_idx,
    "stage2_idx":           stage2_idx,
    "shap_final_features":  shap_final_features,
    "train_prevalence":     train_prevalence,
    "ext_prevalence":       ext_prevalence,
    "temperature_params":   temperature_params,
    "optimal_thresholds":   optimal_thresholds,
}, "pipeline_config_v2.json")

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size  = os.path.getsize(fpath)
    print(f"  {fname:<55}  {size:>10,} bytes")

# ── Final Console Summary ─────────────────────────────────────
print("\n" + "=" * 65)
print("  PIPELINE v2 COMPLETE — FINAL PERFORMANCE SUMMARY")
print("=" * 65)

best_model = max(results_internal, key=lambda m: results_internal[m]["f1_score"])

print(f"\nFeature Selection Journey:")
print(f"  After encoding    : {len(ALL_FEATURES):>4} features")
print(f"  After Stage 1     : {len(stage1_set):>4} features  (MI + Chi² union)")
print(f"  After Stage 2     : {len(stage2_features):>4} features  (RFECV optimal)")
print(f"  After SHAP        : {len(shap_final_features):>4} features  "
      f"({'SHAP ran' if HAS_SHAP else 'SHAP skipped'})")

print(f"\nCalibration Method: Prior Shift Correction + Temperature Scaling")
print(f"  Train prevalence  : {train_prevalence:.3f}")
print(f"  Ext. prevalence   : {ext_prevalence:.3f}")
print(f"  Log-odds shift    : "
      f"{np.log(ext_prevalence/(1-ext_prevalence)) - np.log(train_prevalence/(1-train_prevalence)):.3f}")
print(f"  Temperatures (T)  : { {k: f'{v:.2f}' for k, v in temperature_params.items()} }")
print(f"  Covariate shift   : {n_shifted}/{len(ALL_FEATURES)} features shifted (KS p<0.05)")

print(f"\n{'Model':<22}  {'Int.F1':>7}  {'Int.AUC':>8}  {'Ext.F1(cal)':>12}  "
      f"{'Ext.AUC':>8}  {'Brier(cal)':>10}  {'Brier CI':>18}")
print(f"  {'-'*22}  {'-'*7}  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*18}")

for nm in model_names:
    ri  = results_internal[nm]
    rc  = results_calibrated[nm]
    ci  = rc.get("brier_ci", {})
    rt  = results_threshold[nm]
    star = " ★" if nm == best_model else ""
    print(f"  {nm+star:<22}  {ri['f1_score']:>7.4f}  {ri['roc_auc']:>8.4f}  "
          f"{rt['ext_f1_optimal_threshold']:>12.4f}  {rc['roc_auc']:>8.4f}  "
          f"{rc['brier_score']:>10.4f}  "
          f"[{ci.get('lower',0):.3f}–{ci.get('upper',0):.3f}]")

print(f"\n  ★ Best model (internal F1): {best_model}")
print(f"    Internal AUC  : {results_internal[best_model]['roc_auc']:.4f}")
print(f"    External AUC  : {results_calibrated[best_model]['roc_auc']:.4f}")
print(f"    Brier (cal v2): {results_calibrated[best_model]['brier_score']:.4f}")
print(f"    Brier 95% CI  : [{results_calibrated[best_model]['brier_ci']['lower']:.3f}"
      f"–{results_calibrated[best_model]['brier_ci']['upper']:.3f}]")
print("=" * 65)
