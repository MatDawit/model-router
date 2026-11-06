import os
from pathlib import Path
import torch

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "router" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "calibrated_router.joblib"
SCALER_PATH = ARTIFACTS_DIR / "struct_scaler.joblib"

# Hardware Device (Apple Silicon mps or cpu)
DEVICE = os.getenv("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
# Model Names
SMALL_MODEL = os.getenv("SMALL_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "gpt-4o")

# Pricing per 1M tokens (USD)
PRICE_SMALL_PER_1M = float(os.getenv("PRICE_SMALL_PER_1M", "0.15"))
PRICE_FRONTIER_PER_1M = float(os.getenv("PRICE_FRONTIER_PER_1M", "5.00"))

# Router Thresholds
# If calibrated probability of failure >= ROUTING_THRESHOLD_TAU -> Escalate to Frontier
ROUTING_THRESHOLD_TAU = float(os.getenv("ROUTING_THRESHOLD_TAU", "0.50"))

# Speculative Cascade Bounds
# When CASCADE_LOWER <= p_hat < CASCADE_UPPER: speculatively invoke Small model first,
# validate with Pydantic/JSON schema, and fallback to Frontier only on validation error.
CASCADE_LOWER = 0.45
CASCADE_UPPER = 0.75

# Semantic Cache Thresholds (Lab 1)
CACHE_STAGE1_SIM_THRESHOLD = 0.88  # Cosine similarity candidate threshold
CACHE_MAX_ENTRIES = 5000           # In-memory LRU cache capacity