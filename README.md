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
├── production/                        # Phase 2: everything real-world (self-contained; run from here)
│   ├── core/                              # Phase 2: Production Multi-Tier Gateway (In Development)
│   │   ├── router/                        # Hybrid features, LightGBM / DeBERTa routers, conformal.py, cascade.py
│   │   ├── cache/                         # Bi-encoder filter + DeBERTa-v3 NLI cross-encoder verifier (MNLI/PAWS/QQP)
│   │   └── gateway/                       # FastAPI proxy with SSE streaming, OpenRouter client, Prometheus telemetry
│   │
│   ├── benchmarks/                        # Real-World Evaluation Suite
│   │   ├── data_loaders/                  # RouterBench (Martian) + OpenRouter eval + PAWS/MNLI loaders
│   │   ├── baselines/                     # Random, length heuristic, kNN, Matrix Factorization (RouteLLM paper)
│   │   └── evaluate_production.py         # Bootstrapped Pareto curves & ECE reliability plots
│   │
│   ├── tests/                             # Pytest suite
│   └── requirements.txt                   # Phase 2 dependencies (run everything from inside production/)
```

---

## Project Phases

### 1. Phase 1: Prototype & Foundational Labs (`practice/`)
Before building the full production system, we designed an end-to-end prototype to validate core hypotheses:
* **Lab 1:** Disproved single-stage vector caching by demonstrating an **83.3% cache poisoning rate** on adversarial negations, establishing the need for two-stage verification.
* **Lab 2:** Modeled deterministic failure triggers across 5 task domains.
* **Lab 3:** Extracted a 396-dim hybrid feature representation (384 dense + 12 structural) and trained a Platt-calibrated $L_2$ Logistic Regression classifier with sub-10ms routing latency.
* **Gateway & Pareto:** Deployed a working `/v1/chat/completions` proxy with speculative fallback and live telemetry, achieving a **43.6% cost reduction with 100% retained accuracy** on the *synthetic* prototype benchmark (a proof of mechanism, not a real-world result; Phase 2 re-measures on real data).

👉 *See [`practice/README.md`](practice/README.md) for the complete walkthrough, code, and laboratory findings.*

---

### 2. Phase 2: Production Upgrade (Current Focus)
We are currently upgrading the system into an enterprise-grade portfolio project:
1. **Real Model Outputs:** Ingesting **RouterBench** (Martian; precomputed real LLM responses and quality scores across multiple models and benchmarks) for training and baseline comparison, and calling real models through **OpenRouter** to validate on the exact models the gateway deploys.
2. **Advanced Router Architecture:** Upgrading to LightGBM on hybrid features and fine-tuned DeBERTa encoders, evaluating Expected Calibration Error (ECE) and out-of-domain (OOD) task holdouts.
3. **Statistical Risk Control:** Using split-conformal / Learn-then-Test style calibration to choose the routing threshold so that the accuracy drop vs. always-frontier is bounded by a target ε with a stated confidence (e.g. 95%). The guarantee holds under exchangeability between calibration and deployment data; out-of-domain behavior is evaluated and reported separately.
4. **NLI Semantic Cache:** Upgrading Stage 2 to an NLI Cross-Encoder evaluated on MNLI, PAWS, and QQP contradiction benchmarks.
