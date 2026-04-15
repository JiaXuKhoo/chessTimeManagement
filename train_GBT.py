import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# =========================================
# Config
# =========================================

DATA_PATH = "dataset_probe_tol20.csv"   # change to dataset_probe_tol20.csv if needed
MODEL_OUT = "trained_models/gbt_probe_tol20.joblib"

TARGET_COL = "label_bucket"
TEST_SIZE = 0.2
RANDOM_STATE = 42

DROP_COLS = {
    "fen",
    "tau_cp",
    "probe_error",
    "probe_score_type",
}

# =========================================
# Load data
# =========================================

df = pd.read_csv(DATA_PATH)

# Keep only rows with a valid label
df = df[df[TARGET_COL].notna()].copy()

# If probe dataset contains probe errors, remove those rows
if "probe_error" in df.columns:
    df = df[df["probe_error"].isna()].copy()

# Keep only numeric / boolean columns as features
candidate_feature_cols = [c for c in df.columns if c not in DROP_COLS and c != TARGET_COL]
feature_cols = df[candidate_feature_cols].select_dtypes(include=["number", "bool"]).columns.tolist()

X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int)

print("Dataset shape:", df.shape)
print("Num features:", len(feature_cols))

print("\nLabel distribution:")
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

# =========================================
# Compute balanced sample weights
# =========================================

# This gives larger weight to rare classes and smaller weight to the dominant 25k class
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# Optional: inspect average weight per class
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
    max_iter=200,         # number of boosting rounds
    learning_rate=0.05,   # smaller learning rate = more stable learning
    max_depth=8,          # controls tree complexity
    min_samples_leaf=20,  # helps reduce overfitting
    random_state=RANDOM_STATE
)

# =========================================
# Train
# =========================================

# Fit using sample weights so rare classes matter more during optimization
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

min_size = test_df[TARGET_COL].value_counts().min()

balanced_parts = []
for label, group in test_df.groupby(TARGET_COL):
    balanced_parts.append(group.sample(n=min_size, random_state=RANDOM_STATE))

balanced_test_df = pd.concat(balanced_parts, axis=0).reset_index(drop=True)

X_test_bal = balanced_test_df[feature_cols]
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

X_test_hard = hard_df[feature_cols]
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

joblib.dump({
    "model": model,
    "feature_cols": feature_cols
}, MODEL_OUT)

print(f"\nSaved to {MODEL_OUT}")