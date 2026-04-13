"""Advanced scoring and evaluation module.

Integrates: scoringrules, netcal, MAPIE for comprehensive
prediction quality assessment beyond basic Brier score.

Provides:
- Multi-metric evaluation (Brier, CRPS, log-loss, ECE, MCE)
- Conformal prediction intervals (MAPIE) for uncertainty quantification
- Reliability diagrams with netcal
- Brier score decomposition (reliability + resolution + uncertainty)
- Per-category performance breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from loguru import logger

try:
    import scoringrules as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    from netcal.metrics import ECE, MCE
    from netcal.presentation import ReliabilityDiagram
    HAS_NETCAL = True
except ImportError:
    HAS_NETCAL = False

# MAPIE removed — conformal intervals implemented directly via split conformal
# residuals, which is simpler and more reliable for our use case.


@dataclass
class BrierDecomposition:
    """Brier score decomposed into reliability, resolution, uncertainty.

    Brier = Reliability - Resolution + Uncertainty

    - Reliability: how well-calibrated (lower = better)
    - Resolution: ability to separate outcomes (higher = better)
    - Uncertainty: base rate entropy (fixed for a dataset)
    """
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    n_bins: int = 10


@dataclass
class ComprehensiveMetrics:
    """Full evaluation of a set of predictions."""
    n_samples: int
    brier: float
    log_loss: float
    ece: float
    mce: float
    crps: float
    accuracy: float

    # Decomposition
    reliability: float
    resolution: float
    uncertainty: float

    # Per-bin calibration
    bin_counts: list[int] = field(default_factory=list)
    bin_accuracies: list[float] = field(default_factory=list)
    bin_confidences: list[float] = field(default_factory=list)

    # Conformal interval (if computed)
    coverage_90: float | None = None
    mean_interval_width: float | None = None


def brier_decomposition(
    predictions: Sequence[float],
    outcomes: Sequence[float],
    n_bins: int = 10,
) -> BrierDecomposition:
    """Decompose Brier score into reliability, resolution, uncertainty.

    Murphy (1973) decomposition. This tells you whether your errors come
    from poor calibration (fixable) or from genuine unpredictability (not fixable).
    """
    preds = np.array(predictions)
    outs = np.array(outcomes)
    n = len(preds)

    if n == 0:
        return BrierDecomposition(0, 0, 0, 0)

    base_rate = np.mean(outs)
    uncertainty = base_rate * (1 - base_rate)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if not np.any(mask):
            continue
        n_k = np.sum(mask)
        p_k = np.mean(preds[mask])
        o_k = np.mean(outs[mask])

        reliability += (n_k / n) * (p_k - o_k) ** 2
        resolution += (n_k / n) * (o_k - base_rate) ** 2

    brier = reliability - resolution + uncertainty

    return BrierDecomposition(
        brier=round(brier, 6),
        reliability=round(reliability, 6),
        resolution=round(resolution, 6),
        uncertainty=round(uncertainty, 6),
        n_bins=n_bins,
    )


def comprehensive_evaluate(
    predictions: Sequence[float],
    outcomes: Sequence[float],
    n_bins: int = 10,
) -> ComprehensiveMetrics:
    """Full multi-metric evaluation of predictions."""
    preds = np.array(predictions, dtype=np.float64)
    outs = np.array(outcomes, dtype=np.float64)
    n = len(preds)

    # Brier
    brier = float(np.mean((preds - outs) ** 2))

    # Log-loss
    eps = 1e-7
    p_clip = np.clip(preds, eps, 1 - eps)
    log_loss = float(-np.mean(outs * np.log(p_clip) + (1 - outs) * np.log(1 - p_clip)))

    # CRPS
    if HAS_SR:
        try:
            crps = float(np.mean(sr.crps_normal(outs, preds, 0.1)))
        except Exception:
            crps = brier
    else:
        crps = brier

    # ECE / MCE
    if HAS_NETCAL:
        ece = float(ECE(n_bins).measure(preds, outs.astype(np.int32)))
        mce = float(MCE(n_bins).measure(preds, outs.astype(np.int32)))
    else:
        ece = _fallback_ece(preds, outs, n_bins)
        mce = _fallback_mce(preds, outs, n_bins)

    # Accuracy
    accuracy = float(np.mean((preds >= 0.5).astype(float) == outs))

    # Brier decomposition
    decomp = brier_decomposition(predictions, outcomes, n_bins)

    # Per-bin data
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = []
    bin_accs = []
    bin_confs = []
    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(round(float(np.mean(outs[mask])), 4))
            bin_confs.append(round(float(np.mean(preds[mask])), 4))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return ComprehensiveMetrics(
        n_samples=n,
        brier=round(brier, 6),
        log_loss=round(log_loss, 6),
        ece=round(ece, 6),
        mce=round(mce, 6),
        crps=round(crps, 6),
        accuracy=round(accuracy, 4),
        reliability=decomp.reliability,
        resolution=decomp.resolution,
        uncertainty=decomp.uncertainty,
        bin_counts=bin_counts,
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
    )


def conformal_prediction_interval(
    train_predictions: Sequence[float],
    train_outcomes: Sequence[float],
    new_prediction: float,
    alpha: float = 0.1,
) -> tuple[float, float] | None:
    """Compute conformal prediction interval for a new prediction.

    Uses split conformal with residuals. Returns (lower, upper) bounds
    with (1 - alpha) coverage guarantee.

    Uses split conformal with absolute residuals. Does not require MAPIE;
    implemented directly for simplicity and reliability.
    """
    preds = np.array(train_predictions)
    outs = np.array(train_outcomes)

    # Compute conformity scores (absolute residuals)
    residuals = np.abs(preds - outs)
    residuals_sorted = np.sort(residuals)

    n = len(residuals_sorted)
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(residuals_sorted, q_level))

    lower = max(0.01, new_prediction - q_hat)
    upper = min(0.99, new_prediction + q_hat)

    return (round(lower, 4), round(upper, 4))


def _fallback_ece(preds, outs, n_bins):
    n = len(preds)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if np.any(mask):
            ece += np.sum(mask) / n * abs(float(np.mean(outs[mask])) - float(np.mean(preds[mask])))
    return float(ece)


def _fallback_mce(preds, outs, n_bins):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if np.any(mask):
            mce = max(mce, abs(float(np.mean(outs[mask])) - float(np.mean(preds[mask]))))
    return float(mce)


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE — Phase 2.3
# ═══════════════════════════════════════════════════════════════════════

def brier_skill_score(our_brier: float, reference_brier: float) -> float:
    """Brier Skill Score: BSS = 1 - (our / reference).

    > 0 means we beat the reference.
    = 0 means equal.
    < 0 means we're worse.

    Common references:
    - 0.25 = uniform random (always predict 0.5)
    - market closing price = crowd wisdom baseline
    """
    if reference_brier <= 0:
        return 0.0
    return 1.0 - our_brier / reference_brier


@dataclass
class PairedBrierResult:
    """Result of a paired bootstrap test comparing two forecasters."""
    bss: float               # Brier Skill Score (>0 = we beat market)
    p_value: float            # probability BSS <= 0 under bootstrap
    ci_lower: float           # 95% CI lower bound on our Brier
    ci_upper: float           # 95% CI upper bound on our Brier
    n_markets: int
    our_brier: float
    market_brier: float
    significant: bool         # p_value < 0.10


def paired_brier_test(
    our_predictions: Sequence[float],
    market_prices: Sequence[float],
    outcomes: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> PairedBrierResult:
    """Bootstrap test: is K-Fish Brier significantly different from market?

    Compares our calibrated predictions against market closing prices
    (the crowd's implicit probability estimate). If BSS > 0 with
    p < 0.10, we have statistically significant edge.

    Args:
        our_predictions: our calibrated probabilities
        market_prices: market YES prices at time of prediction
        outcomes: actual binary outcomes (0 or 1)
        n_bootstrap: number of bootstrap resamples
        seed: random seed

    Returns:
        PairedBrierResult with BSS, p-value, and CI.
    """
    ours = np.array(our_predictions)
    market = np.array(market_prices)
    outs = np.array(outcomes)
    n = len(ours)

    our_brier = float(np.mean((ours - outs) ** 2))
    mkt_brier = float(np.mean((market - outs) ** 2))
    bss = brier_skill_score(our_brier, mkt_brier)

    # Paired bootstrap
    rng = np.random.RandomState(seed)
    boot_bss = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        b_ours = float(np.mean((ours[idx] - outs[idx]) ** 2))
        b_mkt = float(np.mean((market[idx] - outs[idx]) ** 2))
        boot_bss[b] = brier_skill_score(b_ours, b_mkt)

    p_value = float(np.mean(boot_bss <= 0))

    # CI on our Brier — must use PAIRED resampling (same indices for preds and outs)
    boot_briers = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_briers[b] = float(np.mean((ours[idx] - outs[idx]) ** 2))
    ci_lower = float(np.percentile(boot_briers, 2.5))
    ci_upper = float(np.percentile(boot_briers, 97.5))

    return PairedBrierResult(
        bss=round(bss, 4),
        p_value=round(p_value, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        n_markets=n,
        our_brier=round(our_brier, 4),
        market_brier=round(mkt_brier, 4),
        significant=p_value < 0.10,
    )


@dataclass
class CategoryStats:
    """Statistics for a single market category."""
    category: str
    n_markets: int
    brier: float
    accuracy: float
    avg_edge: float
    bss_vs_random: float


def per_category_breakdown(
    predictions: list[dict],
) -> dict[str, CategoryStats]:
    """Which categories does K-Fish have edge in?

    Each dict in predictions should have:
    - category: str
    - prediction: float (our calibrated probability)
    - outcome: float (0 or 1)
    - market_price: float (optional, for edge computation)
    """
    categories: dict[str, list[dict]] = {}
    for p in predictions:
        cat = p.get("category", "general") or "general"
        categories.setdefault(cat, []).append(p)

    stats = {}
    for cat, items in categories.items():
        preds = np.array([i["prediction"] for i in items])
        outs = np.array([i["outcome"] for i in items])
        prices = np.array([i.get("market_price", 0.5) or 0.5 for i in items])

        brier = float(np.mean((preds - outs) ** 2))
        accuracy = float(np.mean((preds >= 0.5).astype(float) == outs))
        avg_edge = float(np.mean(np.abs(preds - prices)))
        bss = brier_skill_score(brier, 0.25)

        stats[cat] = CategoryStats(
            category=cat,
            n_markets=len(items),
            brier=round(brier, 4),
            accuracy=round(accuracy, 4),
            avg_edge=round(avg_edge, 4),
            bss_vs_random=round(bss, 4),
        )

    return dict(sorted(stats.items(), key=lambda x: x[1].brier))
