import time
import joblib
import numpy as np
from typing import Dict, Any
from app.config import (
    MODEL_PATH, 
    SMALL_MODEL, 
    FRONTIER_MODEL, 
    ROUTING_THRESHOLD_TAU,
    CASCADE_LOWER, 
    CASCADE_UPPER
)
from app.router.features import FeatureExtractor

class RoutingClassifier:
    """Production router that wraps the calibrated logistic regression classifier."""

    def __init__(self, feature_extractor: FeatureExtractor = None):
        print(f"Loading calibrated router model from {MODEL_PATH.name}...")
        self.model = joblib.load(MODEL_PATH)
        self.feature_extractor = feature_extractor or FeatureExtractor()

    def route(self, prompt: str) -> Dict[str, Any]:
        """
        Calculates empirical failure probability and decides routing strategy.
        Returns a dictionary with strategy, selected model, calibrated probability, and latency.
        """
        t0 = time.perf_counter()

        # 1. Extract 396-dim hybrid features
        X_hybrid = self.feature_extractor.extract(prompt)

        # 2. Predict calibrated probability of requiring Frontier (class 1)
        # predict_proba returns [prob_class_0, prob_class_1]
        probs = self.model.predict_proba(X_hybrid)[0]
        p_frontier = float(probs[1])

        # 3. Determine routing strategy
        if CASCADE_LOWER <= p_frontier < CASCADE_UPPER:
            strategy = "speculative_cascade"
            selected_model = SMALL_MODEL
        elif p_frontier >= ROUTING_THRESHOLD_TAU:
            strategy = "direct_frontier"
            selected_model = FRONTIER_MODEL
        else:
            strategy = "direct_small"
            selected_model = SMALL_MODEL

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "strategy": strategy,
            "selected_model": selected_model,
            "p_frontier": round(p_frontier, 4),
            "routing_latency_ms": round(latency_ms, 2)
        }