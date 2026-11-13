"""Risk control: turn a measured quality drop into a guaranteed upper limit.

We measure the average loss on the calibration set. That is only a sample, so the
true loss on future prompts could be higher. The upper bound adds a "margin of
error" so that, with probability 1 - alpha, the TRUE average loss is below it.
"""

import numpy as np


def empirical_bernstein_upper_bound(losses, alpha=0.05, low=-1.0, high=1.0):
    """Upper limit on the true mean loss, valid with probability at least 1 - alpha.

    Empirical Bernstein bound (Maurer & Pontil, 2009). It assumes the losses are
    independent draws from the same distribution and lie between `low` and `high`.
    It is tighter than Hoeffding because it uses the measured spread of the losses.
    """
    x = np.asarray(losses, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 losses")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if x.min() < low or x.max() > high:
        raise ValueError(f"losses must lie in [{low}, {high}]")

    log_term = np.log(2.0 / alpha)
    spread = high - low
    variance = x.var(ddof=1)  # sample variance: how much the losses differ from each other

    spread_margin = np.sqrt(2.0 * variance * log_term / n)           # shrinks like 1/sqrt(n)
    small_sample_margin = 7.0 * spread * log_term / (3.0 * (n - 1))  # extra caution for small n
    return float(x.mean() + spread_margin + small_sample_margin)


def fixed_sequence_certify(candidate_losses, epsilon=0.02, alpha=0.05):
    """Climb a ladder of candidates, safest first, and certify each until one fails.

    `candidate_losses` is a list of loss arrays (one per candidate cutoff pair),
    ALREADY ORDERED from safest to riskiest, with that order chosen without looking
    at the calibration outcomes. A candidate is certified when its upper bound is
    at most `epsilon`. We stop at the first failure.

    Returns the positions of the certified candidates. With probability at least
    1 - alpha, every one of them has a true average loss of at most `epsilon`.
    """
    certified = []
    for position, losses in enumerate(candidate_losses):
        if empirical_bernstein_upper_bound(losses, alpha=alpha) <= epsilon:
            certified.append(position)
        else:
            break  # first failure: stop, so we never run many tests at full strength at once
    return certified
