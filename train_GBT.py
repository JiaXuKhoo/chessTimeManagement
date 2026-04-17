import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# =========================================
# Config
# =========================================

FEATURE_MODE = "probe"   # "static" or "probe"

if FEATURE_MODE == "static":
    DATA_PATH = "dataset_static_tol20.csv"
    MODEL_OUT = "trained_models/gbt_static_tol20.joblib"
elif FEATURE_MODE == "probe":
    DATA_PATH = "dataset_probe_tol20.csv"
    MODEL_OUT = "trained_models/gbt_probe_tol20.joblib"
else:
    raise ValueError("FEATURE_MODE must be either 'static' or 'probe'")

TARGET_COL = "label_bucket"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================================
# Explicit feature lists
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

if FEATURE_MODE == "static":
    FEATURE_COLS = STATIC_FEATURE_COLS
else:
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
        f"The dataset for FEATURE_MODE='{FEATURE_MODE}' is missing required feature columns:\n"
        + "\n".join(missing_features)
    )

# Keep only rows with a valid label
df = df[df[TARGET_COL].notna()].copy()

# If probe dataset contains probe errors, remove those rows
if "probe_error" in df.columns:
    df = df[df["probe_error"].isna()].copy()

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].astype(int)

print("Feature mode:", FEATURE_MODE)
print("Dataset path:", DATA_PATH)
print("Dataset shape:", df.shape)
print("Num features:", len(FEATURE_COLS))

print("\nFeature columns:")
print(FEATURE_COLS)

print("\nLabel distribution (counts):")
print(y.value_counts().sort_index())

print("\nLabel distribution (proportion):")
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
# Compute balanced sample weights
# =========================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

train_weight_df = pd.DataFrame({
    "label_bucket": y_train.values,
    "sample_weight": sample_weights
})

print("\nAverage sample weight per class:")
print(train_weight_df.groupby("label_bucket")["sample_weight"].mean().sort_index())

# =========================================
# Model
# =========================================

model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_depth=8,
    min_samples_leaf=20,
    random_state=RANDOM_STATE
)

# =========================================
# Train
# =========================================

model.fit(X_train, y_train, sample_weight=sample_weights)

# =========================================
# Evaluate on original imbalanced test set
# =========================================

y_pred = model.predict(X_test)

print("\n=== Evaluation on original test set (imbalanced) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================================
# Balanced test subset
# =========================================

test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values

print("\nTest-set class counts before balancing:")
print(test_df[TARGET_COL].value_counts().sort_index())

min_size = test_df[TARGET_COL].value_counts().min()

balanced_parts = []
for label, group in test_df.groupby(TARGET_COL):
    balanced_parts.append(group.sample(n=min_size, random_state=RANDOM_STATE))

balanced_test_df = pd.concat(balanced_parts, axis=0).reset_index(drop=True)

print("\nBalanced test-set class counts:")
print(balanced_test_df[TARGET_COL].value_counts().sort_index())

X_test_bal = balanced_test_df[FEATURE_COLS]
y_test_bal = balanced_test_df[TARGET_COL].astype(int)

y_pred_bal = model.predict(X_test_bal)

print("\n=== Balanced test ===")
print("Accuracy:", accuracy_score(y_test_bal, y_pred_bal))
print("\nClassification report:")
print(classification_report(y_test_bal, y_pred_bal, digits=4, zero_division=0))
print("\nConfusion matrix:")
print(confusion_matrix(y_test_bal, y_pred_bal))

# =========================================
# Hard subset
# =========================================

hard_df = test_df[test_df[TARGET_COL] != 25000].copy()

if len(hard_df) > 0:
    X_test_hard = hard_df[FEATURE_COLS]
    y_test_hard = hard_df[TARGET_COL].astype(int)

    y_pred_hard = model.predict(X_test_hard)

    print("\n=== Hard subset ===")
    print("Accuracy:", accuracy_score(y_test_hard, y_pred_hard))
    print("\nClassification report:")
    print(classification_report(y_test_hard, y_pred_hard, digits=4, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test_hard, y_pred_hard))

# =========================================
# Save model
# =========================================

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

joblib.dump({
    "model": model,
    "feature_mode": FEATURE_MODE,
    "feature_cols": FEATURE_COLS,
    "static_feature_cols": STATIC_FEATURE_COLS,
    "probe_feature_cols": PROBE_FEATURE_COLS,
    "target_col": TARGET_COL,
}, MODEL_OUT)

print(f"\nSaved to {MODEL_OUT}")