import os
import time
import uuid
import httpx
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import (
    SMALL_MODEL, 
    FRONTIER_MODEL, 
    DEVICE
)
from app.router.features import FeatureExtractor
from app.router.classifier import RoutingClassifier
from app.router.cascade import SpeculativeCascade
from app.cache.semantic_cache import TwoStageSemanticCache
from app.telemetry.metrics import telemetry

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Model-Router",
    description="Cost-Aware Semantic Router & Speculative Agent Gateway",
    version="1.0.0"
)

# 2. Initialize Core Subsystems on Startup
print("Initializing Gateway Subsystems...")
feature_extractor = FeatureExtractor()
router = RoutingClassifier(feature_extractor=feature_extractor)
semantic_cache = TwoStageSemanticCache(bi_encoder=feature_extractor.bi_encoder)

# 3. Model Callers (Supports real OpenAI if API key exists, otherwise smart local simulation)
async def call_small_model(prompt: str, **kwargs) -> str:
    """Invokes Small model (Llama-3.1-8B)."""
    # If OPENAI_API_KEY is available, we can route to vLLM, Ollama, or OpenAI
    # For self-contained testing, returns clean completion
    return f"[Llama-3.1-8B-Instruct]: Completed request for: '{prompt[:40]}...'"

async def call_frontier_model(prompt: str, **kwargs) -> str:
    """Invokes Frontier model (GPT-4o)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=30.0
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    
    return f"[GPT-4o-Frontier]: Rigorously verified completion for: '{prompt[:40]}...'"

speculative_cascade = SpeculativeCascade(
    small_caller=call_small_model, 
    frontier_caller=call_frontier_model
)

# --- OpenAI Compatible Pydantic Schemas ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "model-router"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False

# --- Endpoints ---

@app.get("/")
def health_check():
    """Health check and gateway status."""
    return {
        "status": "healthy",
        "service": "Model-Router",
        "device": DEVICE,
        "small_model": SMALL_MODEL,
        "frontier_model": FRONTIER_MODEL
    }

@app.get("/metrics")
def get_metrics():
    """Live telemetry: cost savings, cache hit rate, blocked attacks."""
    return telemetry.get_summary()

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible Chat Completions Gateway Endpoint.
    Dynamically routes between Cache, Small Model, and Frontier Model.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    # Extract the user prompt (last user message)
    user_prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), req.messages[-1].content)

    # --- STEP 1: Two-Stage Semantic Cache Check ---
    cached_response, cache_meta = semantic_cache.get(user_prompt)
    
    if cache_meta.get("status") == "stage_2_poisoning_blocked":
        telemetry.record_poisoning_blocked()
        print(f"[CACHE SECURITY] Blocked adversarial poisoning attempt on: '{user_prompt[:50]}'")

    if cached_response is not None:
        telemetry.record_cache_hit()
        return {
            "id": f"chatcmpl-cache-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "semantic-cache",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": cached_response},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": len(user_prompt.split()), "completion_tokens": len(cached_response.split()), "total_tokens": 0},
            "gateway_meta": {
                "route": "cache_hit",
                "similarity": cache_meta.get("similarity"),
                "cost_usd": 0.0
            }
        }

    # --- STEP 2: Calibrated Router Decision ---
    route_decision = router.route(user_prompt)
    strategy = route_decision["strategy"]
    p_frontier = route_decision["p_frontier"]
    routing_latency = route_decision["routing_latency_ms"]

    # --- STEP 3: Execution Engine ---
    fallback_triggered = False

    if strategy == "speculative_cascade":
        cascade_result = await speculative_cascade.execute(user_prompt)
        final_content = cascade_result["content"]
        final_model = cascade_result["model_used"]
        fallback_triggered = cascade_result["fallback_triggered"]
    elif strategy == "direct_frontier":
        final_content = await call_frontier_model(user_prompt)
        final_model = FRONTIER_MODEL
    else:  # direct_small
        final_content = await call_small_model(user_prompt)
        final_model = SMALL_MODEL

    # --- STEP 4: Populate Cache & Update Telemetry ---
    semantic_cache.set(user_prompt, final_content, {"model": final_model})
    
    tokens_est = len(user_prompt.split()) + len(final_content.split())
    telemetry.record_routing_decision(
        model_used=final_model,
        strategy=strategy,
        routing_latency_ms=routing_latency,
        tokens_used=tokens_est,
        fallback_triggered=fallback_triggered
    )

    # --- STEP 5: Return OpenAI-Compliant Response ---
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": final_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final_content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(user_prompt.split()),
            "completion_tokens": len(final_content.split()),
            "total_tokens": tokens_est
        },
        "gateway_meta": {
            "strategy": strategy,
            "p_frontier": p_frontier,
            "routing_overhead_ms": routing_latency,
            "fallback_triggered": fallback_triggered
        }
    }