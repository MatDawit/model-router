import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import MODEL_PATH, SCALER_PATH, DEVICE

def compute_nesting_depth(text: str) -> int:
    """Computes the maximum nesting level of JSON/curly braces."""
    max_depth = current_depth = 0
    for char in text:
        if char == "{":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == "}":
            current_depth = max(0, current_depth - 1)
    return max_depth

def extract_structural_features(prompt: str) -> list:
    """Extracts the 12 structural complexity features from the prompt."""
    p_lower = prompt.lower()
    words = prompt.split()
    n_chars = max(len(prompt), 1)

    token_count = len(words)
    char_length = len(prompt)
    nesting_depth = compute_nesting_depth(prompt)
    brace_count = prompt.count("{") + prompt.count("}")
    json_flag = int("json" in p_lower or "schema" in p_lower or "raw" in p_lower or "no markdown" in p_lower)

    code_words = ["def ", "class ", "import ", "return ", "async ", "->", "()", ";"]
    code_count = sum(p_lower.count(w) for w in code_words)

    conc_flag = int(any(w in p_lower for w in ["thread", "lock", "async", "backoff", "race"]))
    sql_flag = int(any(w in p_lower for w in ["window", "dense_rank", "row_number", "partition"]))

    math_signals = ["+", "*", "/", "%", ">", "<", "=", "efficiency", "pallets", "req/s"]
    math_count = sum(prompt.count(s) for s in math_signals)

    constraint_words = ["must", "cannot", "only", "strictly", "rules:", "ensure", "constraint:"]
    constraint_count = sum(p_lower.count(w) for w in constraint_words)

    q_count = prompt.count("?")
    num_density = sum(c.isdigit() for c in prompt) / n_chars

    return [
        token_count, char_length, nesting_depth, brace_count, json_flag,
        code_count, conc_flag, sql_flag, math_count, constraint_count,
        q_count, num_density
    ]

class FeatureExtractor:
    """Production feature extraction engine: 384 dense + 12 structural = 396 dims."""

    def __init__(self):
        print(f"Initializing Bi-Encoder on {DEVICE}...")
        self.bi_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        
        print(f"Loading trained structural scaler from {SCALER_PATH.name}...")
        self.scaler = joblib.load(SCALER_PATH)

    def extract(self, prompt: str) -> np.ndarray:
        """
        Extracts and normalizes features for a single prompt string.
        Returns: 2D numpy array of shape (1, 396)
        """
        # 1. Dense Semantic Embedding (1, 384)
        dense_vec = self.bi_encoder.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)

        # 2. Structural Features (1, 12)
        raw_struct = np.array([extract_structural_features(prompt)], dtype=np.float32)
        scaled_struct = self.scaler.transform(raw_struct)

        # 3. Concatenate into 396-dim hybrid representation
        hybrid_vector = np.hstack([dense_vec, scaled_struct])
        return hybrid_vector