import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

# 1. Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "oracle_dataset.parquet"
OUTPUT_CSV = PROJECT_ROOT / "data" / "pareto_frontier.csv"
MODEL_PATH = PROJECT_ROOT / "backend" / "app" / "router" / "artifacts" / "calibrated_router.joblib"
SCALER_PATH = PROJECT_ROOT / "backend" / "app" / "router" / "artifacts" / "struct_scaler.joblib"

# 2. Pricing constants (USD per 1M tokens, 500 tokens/req average)
PRICE_SMALL_PER_1M = 0.15
PRICE_FRONTIER_PER_1M = 5.00
TOKENS_PER_REQ = 500
COST_SMALL_REQ = (TOKENS_PER_REQ / 1_000_000) * PRICE_SMALL_PER_1M
COST_FRONTIER_REQ = (TOKENS_PER_REQ / 1_000_000) * PRICE_FRONTIER_PER_1M

print("=" * 70)
print("PARETO FRONTIER EVALUATION: MODEL-ROUTER")
print("=" * 70)

# 3. Load Dataset & Model Artifacts
print("Loading dataset and calibrated router artifacts...")
df = pd.read_parquet(DATA_PATH)
router = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Initialize Bi-Encoder for dense embeddings
device = "mps" if torch.backends.mps.is_available() else "cpu"
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 4. Feature Extraction Function
def compute_nesting_depth(text):
    max_d = curr_d = 0
    for char in text:
        if char == "{":
            curr_d += 1
            max_d = max(max_d, curr_d)
        elif char == "}":
            curr_d = max(0, curr_d - 1)
    return max_d

def extract_features(prompts):
    dense = bi_encoder.encode(prompts, convert_to_numpy=True, normalize_embeddings=True)
    raw_struct = []
    for p in prompts:
        p_low = p.lower()
        w = p.split()
        nc = max(len(p), 1)
        raw_struct.append([
            len(w), len(p), compute_nesting_depth(p), p.count("{") + p.count("}"),
            int("json" in p_low or "schema" in p_low or "raw" in p_low or "no markdown" in p_low),
            sum(p_low.count(x) for x in ["def ", "class ", "import ", "return ", "async ", "->", "()", ";"]),
            int(any(x in p_low for x in ["thread", "lock", "async", "backoff", "race"])),
            int(any(x in p_low for x in ["window", "dense_rank", "row_number", "partition"])),
            sum(p.count(x) for x in ["+", "*", "/", "%", ">", "<", "=", "efficiency", "pallets", "req/s"]),
            sum(p_low.count(x) for x in ["must", "cannot", "only", "strictly", "rules:", "ensure", "constraint:"]),
            p.count("?"), sum(c.isdigit() for c in p) / nc
        ])
    scaled_struct = scaler.transform(np.array(raw_struct, dtype=np.float32))
    return np.hstack([dense, scaled_struct])

print("Extracting 396-dim features for all benchmark queries...")
X_features = extract_features(df["prompt"].tolist())

# Predict calibrated probabilities p_frontier (P(requires_frontier | prompt))
p_frontier = router.predict_proba(X_features)[:, 1]

# 5. Evaluate Threshold Spectrum
total_queries = len(df)
always_frontier_cost = total_queries * COST_FRONTIER_REQ
frontier_baseline_acc = df["frontier_success"].mean() * 100

thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []

for tau in thresholds:
    # Routing decision: Escalate to Frontier if p_frontier >= tau, else Small
    route_to_frontier = (p_frontier >= tau).astype(int)
    
    n_frontier = int(np.sum(route_to_frontier == 1))
    n_small = int(np.sum(route_to_frontier == 0))
    
    # Calculate effective accuracy
    successes = (
        np.sum((route_to_frontier == 1) & (df["frontier_success"].values == 1)) +
        np.sum((route_to_frontier == 0) & (df["small_success"].values == 1))
    )
    accuracy_pct = (successes / total_queries) * 100
    retained_acc_pct = (accuracy_pct / frontier_baseline_acc) * 100
    
    # Calculate costs
    total_cost = (n_small * COST_SMALL_REQ) + (n_frontier * COST_FRONTIER_REQ)
    cost_saved_pct = (1.0 - (total_cost / always_frontier_cost)) * 100
    
    results.append({
        "Threshold (tau)": round(tau, 2),
        "Route to Small (%)": round((n_small / total_queries) * 100, 1),
        "Route to Frontier (%)": round((n_frontier / total_queries) * 100, 1),
        "System Accuracy (%)": round(accuracy_pct, 2),
        "Retained Accuracy (%)": round(retained_acc_pct, 2),
        "Total Cost ($)": round(total_cost, 4),
        "Cost Savings (%)": round(cost_saved_pct, 1)
    })

pareto_df = pd.DataFrame(results)

# 6. Save CSV
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
pareto_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nPareto frontier data exported to: {OUTPUT_CSV.resolve()}\n")

# 7. Print Formatted Markdown Table for README
print("PARETO FRONTIER SUMMARY TABLE (Markdown):")
print(pareto_df.to_markdown(index=False))

# 8. Print Executive Findings
optimal = pareto_df[pareto_df["Threshold (tau)"] == 0.50].iloc[0]
print("\n" + "=" * 70)
print("EXECUTIVE FINDINGS (Optimal Operating Point at tau = 0.50):")
print(f"- Retained Accuracy:   {optimal['Retained Accuracy (%)']}% of Frontier GPT-4o")
print(f"- Cost Reduction:      {optimal['Cost Savings (%)']}% reduction in inference bills")
print(f"- Workload Split:      {optimal['Route to Small (%)']}% Small (8B) / {optimal['Route to Frontier (%)']}% Frontier")
print("=" * 70)