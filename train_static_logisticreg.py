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

DATA_PATH = "dataset_static_tol20.csv"
MODEL_OUT = "trained_models/logreg_static_tol20.joblib"

TARGET_COL = "label_bucket"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns that are not features, we drop this
DROP_COLS = {
    "fen",          # fen strings
    "tau_cp",       # constant metadata
}

# =========================================
# Load data
# =========================================

df = pd.read_csv(DATA_PATH)

# Keep only rows that actually have a label
df = df[df[TARGET_COL].notna()].copy()

# Build the feature column list by dropping target + non-feature metadata
feature_cols = [c for c in df.columns if c not in DROP_COLS and c != TARGET_COL]

X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int)

print("Dataset shape:", df.shape)
print("Number of features:", len(feature_cols))
print("\nFeature columns:")
print(feature_cols)

print("\nOverall label distribution:")
print(y.value_counts().sort_index())
print("\nOverall label distribution (proportion):")
print(y.value_counts(normalize=True).sort_index())

# =========================================
# Train/test split
# =========================================

# Stratify=y keeps the class proportions similar in train and test
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

#fill missing values using the median of each column
imputer = SimpleImputer(strategy="median")

# Normalize features to zero mean / unit variance
scaler = StandardScaler()

model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

# Put both into one pipeline
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
# Evaluate on normal imbalanced test set
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

# Recombine X_test and y_test so we can sample by class
test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values

print("\nTest-set class counts before balancing:")
print(test_df[TARGET_COL].value_counts().sort_index())

# Find the smallest class count in the TEST split
min_test_class_size = test_df[TARGET_COL].value_counts().min()

# Create a balanced subset by sampling the same number from each class
# Sample each class separately, then concatenate
balanced_parts = []
for label, group in test_df.groupby(TARGET_COL):
    sampled_group = group.sample(n=min_test_class_size, random_state=RANDOM_STATE)
    balanced_parts.append(sampled_group)

balanced_test_df = pd.concat(balanced_parts, axis=0).reset_index(drop=True)

print("\nBalanced test-set class counts:")
print(balanced_test_df[TARGET_COL].value_counts().sort_index())

X_test_balanced = balanced_test_df[feature_cols]
y_test_balanced = balanced_test_df[TARGET_COL].astype(int)

# Predict on balanced evaluation subset
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
    X_test_hard = hard_test_df[feature_cols]
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

joblib.dump({
    "model": clf,
    "feature_cols": feature_cols,
    "target_col": TARGET_COL,
}, MODEL_OUT)

print(f"\nSaved model to {MODEL_OUT}")