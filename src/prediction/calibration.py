"""Probability calibration module.

LLMs are systematically overconfident (Geng et al., NAACL 2024).
This module applies post-hoc calibration to raw probability estimates
to make them trustworthy for trading decisions.

Methods:
- Isotonic Regression (≥1000 samples, non-parametric)
- Platt Scaling (sigmoid, <1000 samples)
- Temperature Scaling (single scalar, simplest)

Evaluation:
- Brier Score: mean squared error of probability estimates
- ECE: Expected Calibration Error
- Log-Loss: logarithmic scoring rule
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality."""

    brier_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    log_loss: float
    n_samples: int
    n_bins: int = 10

    def is_well_calibrated(self, brier_target: float = 0.18) -> bool:
        """Check if calibration meets the target (human superforecaster level)."""
        return self.brier_score <= brier_target


class ProbabilityCalibrator:
    """Post-hoc calibration for LLM probability estimates.

    Usage:
        calibrator = ProbabilityCalibrator(method="isotonic")

        # Train on historical data (predicted probabilities + actual outcomes)
        calibrator.fit(predictions=[0.7, 0.3, 0.9, ...], outcomes=[1, 0, 1, ...])

        # Calibrate new predictions
        calibrated = calibrator.calibrate(0.85)  # → e.g., 0.78

        # Evaluate
        metrics = calibrator.evaluate(predictions, outcomes)
    """

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self._fitted = False

        if method == "isotonic":
            self._model = IsotonicRegression(
                y_min=0.01, y_max=0.99, out_of_bounds="clip"
            )
        elif method == "platt":
            self._model = LogisticRegression(C=1.0, solver="lbfgs")
        elif method == "temperature":
            self._temperature: float = 1.0
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # Track calibration history
        self._train_predictions: list[float] = []
        self._train_outcomes: list[float] = []

        logger.info(f"Calibrator initialized: method={method}")

    def fit(self, predictions: list[float], outcomes: list[float]) -> None:
        """Fit the calibration model on historical data.

        Args:
            predictions: Raw probability estimates from the swarm (0 to 1)
            outcomes: Actual binary outcomes (0 or 1)
        """
        if len(predictions) != len(outcomes):
            raise ValueError("predictions and outcomes must have same length")

        self._train_predictions.extend(predictions)
        self._train_outcomes.extend(outcomes)

        preds = np.array(self._train_predictions)
        outs = np.array(self._train_outcomes)

        if self.method == "isotonic":
            self._model.fit(preds, outs)
        elif self.method == "platt":
            # Platt scaling: fit logistic regression on log-odds
            log_odds = np.log(np.clip(preds, 1e-7, 1 - 1e-7) / (1 - np.clip(preds, 1e-7, 1 - 1e-7)))
            self._model.fit(log_odds.reshape(-1, 1), outs)
        elif self.method == "temperature":
            self._temperature = self._optimize_temperature(preds, outs)

        self._fitted = True
        logger.info(
            f"Calibrator fitted: {len(self._train_predictions)} samples, "
            f"method={self.method}"
        )

    def calibrate(self, probability: float) -> float:
        """Calibrate a single probability estimate.

        Returns the raw probability if the calibrator hasn't been fitted yet.
        """
        if not self._fitted:
            return probability

        p = np.clip(probability, 0.01, 0.99)

        if self.method == "isotonic":
            result = self._model.predict([p])[0]
        elif self.method == "platt":
            log_odds = np.log(p / (1 - p))
            result = self._model.predict_proba(np.array([[log_odds]]))[0, 1]
        elif self.method == "temperature":
            # Temperature scaling: adjust log-odds by temperature
            log_odds = np.log(p / (1 - p))
            scaled = log_odds / self._temperature
            result = 1 / (1 + np.exp(-scaled))
        else:
            result = probability

        return float(np.clip(result, 0.01, 0.99))

    def calibrate_batch(self, probabilities: list[float]) -> list[float]:
        """Calibrate a batch of probability estimates."""
        return [self.calibrate(p) for p in probabilities]

    def evaluate(
        self,
        predictions: list[float],
        outcomes: list[float],
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """Evaluate calibration quality using standard metrics."""
        preds = np.array(predictions)
        outs = np.array(outcomes)
        n = len(preds)

        # Brier Score
        brier = float(np.mean((preds - outs) ** 2))

        # Log-Loss
        eps = 1e-7
        preds_clipped = np.clip(preds, eps, 1 - eps)
        log_loss = float(-np.mean(
            outs * np.log(preds_clipped) + (1 - outs) * np.log(1 - preds_clipped)
        ))

        # ECE and MCE (binned)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
            if not np.any(mask):
                continue
            bin_preds = preds[mask]
            bin_outs = outs[mask]
            bin_acc = np.mean(bin_outs)
            bin_conf = np.mean(bin_preds)
            bin_error = abs(bin_acc - bin_conf)

            ece += len(bin_preds) / n * bin_error
            mce = max(mce, bin_error)

        return CalibrationMetrics(
            brier_score=round(brier, 6),
            ece=round(ece, 6),
            mce=round(mce, 6),
            log_loss=round(log_loss, 6),
            n_samples=n,
            n_bins=n_bins,
        )

    def _optimize_temperature(
        self, predictions: np.ndarray, outcomes: np.ndarray
    ) -> float:
        """Find optimal temperature via grid search minimizing log-loss."""
        best_temp = 1.0
        best_loss = float("inf")

        for temp in np.arange(0.1, 5.0, 0.1):
            log_odds = np.log(np.clip(predictions, 1e-7, 1 - 1e-7) / (1 - np.clip(predictions, 1e-7, 1 - 1e-7)))
            scaled = log_odds / temp
            calibrated = 1 / (1 + np.exp(-scaled))
            calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)

            loss = -np.mean(
                outcomes * np.log(calibrated) + (1 - outcomes) * np.log(1 - calibrated)
            )
            if loss < best_loss:
                best_loss = loss
                best_temp = temp

        logger.info(f"Optimal temperature: {best_temp:.2f}")
        return float(best_temp)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def training_size(self) -> int:
        return len(self._train_predictions)
