"""Compare full-call latencies within matched measurement rounds."""
from dataclasses import dataclass
from math import isfinite
from statistics import median, quantiles


@dataclass(frozen=True)
class LatencyComparison:
    classification: str
    ratio_low: float
    ratio_high: float


def compare_latency(rounds, *, margin=0.05, resolution=0.0, short_limit=0.0):
    """Envelope of within-round middle halves, in candidate/reference order.

    Samples within a round need not be paired. Both the relative margin and
    the timer resolution must be exceeded in every measured round. This is
    an observed spread, not a confidence interval for independent draws.
    """
    if not 0 <= margin < 1 or resolution < 0 or short_limit < 0:
        raise ValueError("invalid comparison margins")
    bounds = []
    votes = []
    for candidate, reference in rounds:
        if min(len(candidate), len(reference)) < 3:
            return None
        if not all(isfinite(x) and x > 0 for group in (candidate, reference) for x in group):
            raise ValueError("comparison requires finite positive latencies")
        clo, _, chi = quantiles(candidate, n=4, method="inclusive")
        rlo, _, rhi = quantiles(reference, n=4, method="inclusive")
        lo, hi = clo / rhi, chi / rlo
        bounds.append((lo, hi))
        floor = resolution if min(median(candidate), median(reference)) < short_limit else 0
        if hi < 1 - margin and rlo - chi > floor:
            votes.append("faster")
        elif lo > 1 + margin and clo - rhi > floor:
            votes.append("slower")
        else:
            votes.append("tie")
    if not bounds:
        return None
    outcome = votes[0] if len(set(votes)) == 1 else "tie"
    return LatencyComparison(outcome, min(x for x, _ in bounds), max(y for _, y in bounds))
