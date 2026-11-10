import threading
from typing import Dict, Any
from app.config import PRICE_SMALL_PER_1M, PRICE_FRONTIER_PER_1M

class GatewayTelemetry:
    """Thread-safe telemetry tracker for gateway costs, latencies, and routing counters."""

    def __init__(self):
        self._lock = threading.Lock()
        
        # Request Counters
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_poisonings_blocked = 0

        # Routing Counters
        self.routed_to_small = 0
        self.routed_to_frontier = 0
        self.speculative_attempts = 0
        self.speculative_fallbacks = 0

        # Latency & Cost Tracking
        self.total_routing_latency_ms = 0.0
        self.actual_cost_usd = 0.0
        self.baseline_frontier_cost_usd = 0.0

    def record_cache_hit(self):
        with self._lock:
            self.total_requests += 1
            self.cache_hits += 1

    def record_poisoning_blocked(self):
        with self._lock:
            self.cache_poisonings_blocked += 1

    def record_routing_decision(
        self, 
        model_used: str, 
        strategy: str, 
        routing_latency_ms: float, 
        tokens_used: int = 500,
        fallback_triggered: bool = False
    ):
        with self._lock:
            self.total_requests += 1
            self.cache_misses += 1
            self.total_routing_latency_ms += routing_latency_ms

            if strategy == "speculative_cascade":
                self.speculative_attempts += 1
                if fallback_triggered:
                    self.speculative_fallbacks += 1

            if "small" in model_used.lower() or "llama" in model_used.lower():
                self.routed_to_small += 1
                req_cost = (tokens_used / 1_000_000) * PRICE_SMALL_PER_1M
            else:
                self.routed_to_frontier += 1
                req_cost = (tokens_used / 1_000_000) * PRICE_FRONTIER_PER_1M

            # Baseline assumes every request would have gone to Frontier
            baseline_cost = (tokens_used / 1_000_000) * PRICE_FRONTIER_PER_1M

            self.actual_cost_usd += req_cost
            self.baseline_frontier_cost_usd += baseline_cost

    def get_summary(self) -> Dict[str, Any]:
        """Returns a snapshot of system health, routing statistics, and cost savings."""
        with self._lock:
            saved_usd = max(0.0, self.baseline_frontier_cost_usd - self.actual_cost_usd)
            savings_pct = (
                (saved_usd / self.baseline_frontier_cost_usd * 100.0) 
                if self.baseline_frontier_cost_usd > 0 else 0.0
            )
            avg_latency = (
                (self.total_routing_latency_ms / self.total_requests)
                if self.total_requests > 0 else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "cache": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate_pct": round((self.cache_hits / max(1, self.total_requests)) * 100, 2),
                    "poisoning_attacks_blocked": self.cache_poisonings_blocked
                },
                "routing": {
                    "routed_to_small": self.routed_to_small,
                    "routed_to_frontier": self.routed_to_frontier,
                    "speculative_attempts": self.speculative_attempts,
                    "speculative_fallbacks": self.speculative_fallbacks
                },
                "economics": {
                    "actual_cost_usd": round(self.actual_cost_usd, 6),
                    "baseline_always_frontier_usd": round(self.baseline_frontier_cost_usd, 6),
                    "total_saved_usd": round(saved_usd, 6),
                    "savings_percentage": f"{savings_pct:.1f}%"
                },
                "performance": {
                    "avg_routing_overhead_ms": round(avg_latency, 2)
                }
            }

# Global singleton telemetry instance
telemetry = GatewayTelemetry()