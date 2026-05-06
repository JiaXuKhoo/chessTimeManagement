import time
import statistics
import random
import joblib
import pandas as pd

MODEL_PATH = "trained_models/gbt_probe_tol20.joblib"
DATA_PATH = "dataset_probe_tol20.csv"

N_SAMPLES = 1000
WARMUP = 100
REPEATS = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

payload = joblib.load(MODEL_PATH)
model = payload["model"]
feature_cols = payload["feature_cols"]

df = pd.read_csv(DATA_PATH)

# Keep only valid rows
if "probe_error" in df.columns:
    df = df[df["probe_error"].isna()].copy()

df = df[df["label_bucket"].notna()].copy()

# Sample rows and keep only model features
sample_df = df.sample(n=min(N_SAMPLES, len(df)), random_state=RANDOM_SEED)[feature_cols].copy()

# Fill any missing values conservatively
sample_df = sample_df.fillna(0)

# Warm-up
for i in range(min(WARMUP, len(sample_df))):
    _ = model.predict(sample_df.iloc[[i]])

times = []

for _ in range(REPEATS):
    for i in range(len(sample_df)):
        row = sample_df.iloc[[i]]
        t0 = time.perf_counter()
        _ = model.predict(row)
        dt = time.perf_counter() - t0
        times.append(dt)

mean_t = statistics.mean(times)
median_t = statistics.median(times)

print(f"Samples timed: {len(times)}")
print(f"Mean inference time per move:   {mean_t * 1e6:.2f} µs")
print(f"Median inference time per move: {median_t * 1e6:.2f} µs")
print(f"Mean inference time over 56 moves: {mean_t * 56 * 1e3:.4f} ms")