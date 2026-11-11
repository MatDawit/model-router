# Model-Router
> **Cost-Aware Semantic Router & Speculative Agent Gateway for Multi-Agent Workflows**

Model-Router is an OpenAI-compatible routing and caching proxy (`/v1/chat/completions`) designed to sit between multi-agent systems and underlying LLMs. It dynamically routes requests across model capability tiers (e.g., Small 8B, Mid 70B, and Frontier models like Claude 3.5 Sonnet / GPT-4o) using calibrated machine learning, a verified two-stage semantic cache, and speculative cascading.

---

## Repository Roadmap & Structure

This repository is organized into distinct phases demonstrating the full ML systems engineering lifecycle:

```text
model-router/
├── practice/                          # Phase 1: Educational Prototype & Foundational Labs
│   ├── notebooks/                     # Labs 1-3: Cache poisoning, oracle data, L2 calibration
│   ├── backend/                       # Prototype FastAPI gateway (/v1/chat/completions)
│   ├── ml_pipeline/                   # Prototype Pareto frontier evaluation script
│   ├── data/                          # Prototype benchmark datasets
│   └── README.md                      # Detailed walkthrough of prototype findings
│
├── core/                              # Phase 2: Production Multi-Tier Gateway (In Development)
│   ├── router/                        # LightGBM + ModernBERT / DeBERTa hybrid routers
│   ├── cache/                         # DeBERTa-v3 NLI Cross-Encoder verifier (MNLI/PAWS)
│   ├── gateway/                       # High-throughput FastAPI proxy with SSE streaming
│   └── conformal/                     # Conformal prediction statistical risk guarantees
│
└── benchmarks/                        # Real-World Evaluation Suite (Milestone A)
    ├── routerbench/                   # LMSYS RouterBench real model outputs (10k+ prompts)
    ├── openrouter/                    # Live OpenRouter evaluation pipeline (8B vs 70B vs Frontier)
    └── metrics/                       # ECE calibration plots & bootstrapped Pareto frontiers
```

---

## Project Phases

### 1. Phase 1: Prototype & Foundational Labs (`practice/`)
Before building the full production system, we designed an end-to-end prototype to validate core hypotheses:
* **Lab 1:** Disproved single-stage vector caching by demonstrating an **83.3% cache poisoning rate** on adversarial negations, establishing the need for two-stage verification.
* **Lab 2:** Modeled deterministic failure triggers across 5 task domains.
* **Lab 3:** Extracted a 396-dim hybrid feature representation (384 dense + 12 structural) and trained a Platt-calibrated $L_2$ Logistic Regression classifier with sub-10ms routing latency.
* **Gateway & Pareto:** Deployed a working `/v1/chat/completions` proxy with speculative fallback and live telemetry, achieving a **43.6% cost reduction with 100% retained accuracy** on the prototype benchmark.

👉 *See [`practice/README.md`](practice/README.md) for the complete walkthrough, code, and laboratory findings.*

---

### 2. Phase 2: Production Upgrade (Current Focus)
We are currently upgrading the system into an enterprise-grade portfolio project:
1. **Real Model Outputs:** Ingesting **LMSYS RouterBench** (10k+ real prompts with Llama-3-8B vs. GPT-4 ground-truth outputs) paired with live **OpenRouter** model evaluation.
2. **Advanced Router Architecture:** Upgrading to LightGBM on hybrid features and fine-tuned DeBERTa encoders, evaluating Expected Calibration Error (ECE) and out-of-domain (OOD) task holdouts.
3. **Statistical Guarantees:** Implementing **Split Conformal Prediction** to mathematically bound accuracy drops under user-defined confidence levels (e.g. 95% confidence).
4. **NLI Semantic Cache:** Upgrading Stage 2 to an NLI Cross-Encoder evaluated on MNLI, PAWS, and QQP contradiction benchmarks.
