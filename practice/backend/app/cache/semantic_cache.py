import re
import numpy as np
from typing import Optional, Dict, Any, Tuple
from app.config import CACHE_STAGE1_SIM_THRESHOLD, CACHE_MAX_ENTRIES

def check_logical_contradiction(p1: str, p2: str) -> bool:
    """
    Checks if one prompt has a negation/directional flip that the other does not.
    Catches cache poisoning attacks (e.g. 'Drop' vs 'Do NOT drop').
    """
    negation_tokens = {"not", "no", "never", "don't", "dont", "cannot", "cant", "except", "revoke"}
    
    tokens1 = set(re.findall(r"\b\w+\b", p1.lower()))
    tokens2 = set(re.findall(r"\b\w+\b", p2.lower()))
    
    neg1 = tokens1.intersection(negation_tokens)
    neg2 = tokens2.intersection(negation_tokens)
    
    # If one has negation words and the other does not -> Contradiction!
    if neg1 != neg2:
        return True
        
    return False

class TwoStageSemanticCache:
    """
    Production Two-Stage Semantic Cache:
    Stage 1: Vectorized cosine similarity filter (all-MiniLM-L6-v2)
    Stage 2: Anti-poisoning verification layer
    """

    def __init__(self, bi_encoder, threshold: float = CACHE_STAGE1_SIM_THRESHOLD, max_entries: int = CACHE_MAX_ENTRIES):
        self.bi_encoder = bi_encoder
        self.threshold = threshold
        self.max_entries = max_entries
        
        # In-memory storage
        self.prompts: list[str] = []
        self.embeddings: list[np.ndarray] = []  # List of (384,) unit vectors
        self.responses: list[str] = []
        self.metadata: list[Dict[str, Any]] = []

    def get(self, prompt: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Queries the cache.
        Returns: (cached_response or None, telemetry_dict)
        """
        if len(self.prompts) == 0:
            return None, {"status": "miss_empty_cache"}

        # 1. Stage 1: Vectorized Cosine Similarity Search
        query_vec = self.bi_encoder.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Compute dot product against all cached vectors in one vectorized operation
        matrix = np.array(self.embeddings)  # Shape: (N, 384)
        sims = np.dot(matrix, query_vec)    # Shape: (N,)
        
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        
        # Check Stage 1 threshold
        if best_sim < self.threshold:
            return None, {
                "status": "stage_1_miss",
                "best_sim": round(best_sim, 3)
            }

        candidate_prompt = self.prompts[best_idx]

        # 2. Stage 2: Anti-Poisoning Intent Verification
        is_contradiction = check_logical_contradiction(prompt, candidate_prompt)
        if is_contradiction:
            return None, {
                "status": "stage_2_poisoning_blocked",
                "best_sim": round(best_sim, 3),
                "matched_prompt": candidate_prompt
            }

        # Cache Hit Verified!
        return self.responses[best_idx], {
            "status": "hit",
            "similarity": round(best_sim, 3),
            "matched_prompt": candidate_prompt
        }

    def set(self, prompt: str, response: str, meta: Dict[str, Any] = None):
        """Stores a newly generated prompt and response into the cache."""
        if len(self.prompts) >= self.max_entries:
            # Simple FIFO eviction if capacity is reached
            self.prompts.pop(0)
            self.embeddings.pop(0)
            self.responses.pop(0)
            self.metadata.pop(0)

        vec = self.bi_encoder.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        self.prompts.append(prompt)
        self.embeddings.append(vec)
        self.responses.append(response)
        self.metadata.append(meta or {})