import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================
# Config
# =========================================

DATA_PATH = "dataset_probe_tol20.csv"
MODEL_OUT = "trained_models/logreg_probe_tol20.joblib"

TARGET_COL = "label_bucket"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================================
# Feature list
# =========================================

STATIC_FEATURE_COLS = [
    "num_legal_moves",
    "is_check",
    "capture_ratio",
    "check_ratio",
    "num_promotions",
    "knight_mobility",
    "bishop_mobility",
    "rook_mobility",
    "queen_mobility",
    "num_attackers_on_king",
    "num_pinned_pieces",
    "side_to_move_white",
]

PROBE_FEATURE_COLS = [
    "probe_score_cp",
    "probe_abs_score_cp",
    "probe_best_second_gap",
    "probe_depth",
    "probe_seldepth",
    "probe_is_mate",
    "probe_score_delta_small",
    "probe_gap_delta_small",
    "probe_sign_flip",
    "probe_top_move_changed",
]

FEATURE_COLS = STATIC_FEATURE_COLS + PROBE_FEATURE_COLS

# =========================================
# Load data
# =========================================

df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in {DATA_PATH}")

missing_features = [c for c in FEATURE_COLS if c not in df.columns]
if missing_features:
    raise ValueError(
        "The dataset is missing required feature columns:\n"
        + "\n".join(missing_features)
    )

# Keep only rows that actually have a label
df = df[df[TARGET_COL].notna()].copy()

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].astype(int)

print("Dataset shape:", df.shape)
print("Number of features:", len(FEATURE_COLS))

print("\nStatic feature columns:")
print(STATIC_FEATURE_COLS)

print("\nProbe feature columns:")
print(PROBE_FEATURE_COLS)

print("\nOverall label distribution:")
print(y.value_counts().sort_index())

print("\nOverall label distribution (proportion):")
print(y.value_counts(normalize=True).sort_index())

# =========================================
# Train/test split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain label distribution:")
print(y_train.value_counts().sort_index())

print("\nTest label distribution:")
print(y_test.value_counts().sort_index())

# =========================================
# Preprocessing + model
# =========================================

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

clf = Pipeline(steps=[
    ("imputer", imputer),
    ("scaler", scaler),
    ("model", model),
])

# =========================================
# Train
# =========================================

clf.fit(X_train, y_train)

# =========================================
# Evaluate on original imbalanced test set
# =========================================

y_pred = clf.predict(X_test)

print("\n=== Evaluation on original test set (imbalanced) ===")
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================================
# Create a balanced evaluation subset from TEST ONLY
# =========================================

test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values

print("\nTest-set class counts before balancing:")
print(test_df[TARGET_COL].value_counts().sort_index())

min_test_class_size = test_df[TARGET_COL].value_counts().min()

balanced_parts = []
for label, group in test_df.groupby(TARGET_COL):
    sampled_group = group.sample(n=min_test_class_size, random_state=RANDOM_STATE)
    balanced_parts.append(sampled_group)

balanced_test_df = pd.concat(balanced_parts, axis=0).reset_index(drop=True)

print("\nBalanced test-set class counts:")
print(balanced_test_df[TARGET_COL].value_counts().sort_index())

X_test_balanced = balanced_test_df[FEATURE_COLS]
y_test_balanced = balanced_test_df[TARGET_COL].astype(int)

y_pred_balanced = clf.predict(X_test_balanced)

print("\n=== Evaluation on balanced test subset ===")
print("\nAccuracy:")
print(accuracy_score(y_test_balanced, y_pred_balanced))

print("\nClassification report:")
print(classification_report(y_test_balanced, y_pred_balanced, digits=4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test_balanced, y_pred_balanced))

# =========================================
# Evaluate only on hard positions
# =========================================

hard_test_df = test_df[test_df[TARGET_COL] != 25000].copy()

if len(hard_test_df) > 0:
    X_test_hard = hard_test_df[FEATURE_COLS]
    y_test_hard = hard_test_df[TARGET_COL].astype(int)
    y_pred_hard = clf.predict(X_test_hard)

    print("\n=== Evaluation on hard-position test subset (label != 25000) ===")
    print("\nAccuracy:")
    print(accuracy_score(y_test_hard, y_pred_hard))

    print("\nClassification report:")
    print(classification_report(y_test_hard, y_pred_hard, digits=4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test_hard, y_pred_hard))

# =========================================
# Save model
# =========================================

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

joblib.dump({
    "model": clf,
    "feature_cols": FEATURE_COLS,
    "static_feature_cols": STATIC_FEATURE_COLS,
    "probe_feature_cols": PROBE_FEATURE_COLS,
    "target_col": TARGET_COL,
}, MODEL_OUT)

print(f"\nSaved model to {MODEL_OUT}")