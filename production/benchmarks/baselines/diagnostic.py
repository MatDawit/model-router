"""A DIAGNOSTIC, not a real router: it cheats by reading the benchmark name.

At serving time a prompt arrives with no `eval_name`, so this cannot be deployed.
It answers one question: how much of the failure signal is just "which benchmark
is this prompt from"?
"""


def benchmark_rate_probs(train, other, target, smoothing=10):
    """For each row of `other`, return its benchmark's failure rate measured on `train`.

    Small benchmarks give noisy rates, so each rate is pulled toward the overall
    rate as if `smoothing` extra average prompts had been seen. Benchmarks that
    never appear in `train` get the overall rate.
    """
    overall = train[target].mean()
    stats = train.groupby("eval_name")[target].agg(["sum", "count"])
    smoothed = (stats["sum"] + smoothing * overall) / (stats["count"] + smoothing)
    return other["eval_name"].map(smoothed).fillna(overall).to_numpy()
