"""Experiment tracking module using MLflow.

Tracks every retrodiction run, swarm configuration, and calibration
iteration so that performance can be compared across experiments.

Every prediction is logged with its parameters, metrics, and artifacts
so we can answer: "Which Fish configuration produced the best Sharpe?"
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("mlflow not installed. pip install mlflow")


class ExperimentTracker:
    """Track Mirofish experiments with MLflow.

    Logs: swarm configuration, per-market predictions, aggregate metrics,
    calibration parameters, and portfolio performance.
    """

    def __init__(
        self,
        experiment_name: str = "mirofish-predictions",
        tracking_dir: str = "data/mlflow",
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        if HAS_MLFLOW:
            mlflow.set_tracking_uri(f"file:///{self.tracking_dir.resolve()}")
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow tracking: {self.tracking_dir}")
        else:
            logger.warning("MLflow not available, using JSON fallback")

    def log_retrodiction_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        fish_brier_scores: dict[str, float],
        predictions_path: str | Path | None = None,
    ) -> str | None:
        """Log a complete retrodiction run.

        Args:
            run_name: descriptive name (e.g., "retro_v2_30mkts_haiku")
            params: swarm configuration (model, personas, extremize, etc.)
            metrics: aggregate metrics (brier, ece, accuracy, etc.)
            fish_brier_scores: per-persona Brier scores
            predictions_path: path to full predictions JSON file

        Returns:
            MLflow run_id or None if MLflow unavailable.
        """
        if HAS_MLFLOW:
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                for k, v in params.items():
                    if isinstance(v, (list, dict)):
                        mlflow.log_param(k, json.dumps(v)[:250])
                    else:
                        mlflow.log_param(k, v)

                # Log metrics
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

                # Log per-Fish Brier as metrics
                for persona, brier in fish_brier_scores.items():
                    mlflow.log_metric(f"fish_brier_{persona}", brier)

                # Log predictions file as artifact
                if predictions_path and Path(predictions_path).exists():
                    mlflow.log_artifact(str(predictions_path))

                run_id = mlflow.active_run().info.run_id
                logger.info(f"MLflow run logged: {run_id}")
                return run_id
        else:
            return self._json_fallback(run_name, params, metrics, fish_brier_scores)

    def log_live_prediction(
        self,
        market_id: str,
        question: str,
        probability: float,
        confidence: float,
        edge: float,
        position_size: float = 0.0,
    ) -> None:
        """Log a single live prediction for tracking."""
        if HAS_MLFLOW:
            with mlflow.start_run(run_name=f"live_{market_id}", nested=True):
                mlflow.log_params({
                    "market_id": market_id,
                    "question": question[:200],
                })
                mlflow.log_metrics({
                    "probability": probability,
                    "confidence": confidence,
                    "edge": edge,
                    "position_size": position_size,
                })

    def _json_fallback(
        self,
        run_name: str,
        params: dict,
        metrics: dict,
        fish_brier: dict,
    ) -> str:
        """Fallback logging to JSON when MLflow is not available."""
        fallback_dir = self.tracking_dir / "json_runs"
        fallback_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "metrics": metrics,
            "fish_brier_scores": fish_brier,
        }
        path = fallback_dir / f"{run_id}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"JSON fallback logged: {path}")
        return run_id
