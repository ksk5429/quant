"""Configuration loader — merges default.yaml with local.yaml overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class ApiKeys(BaseModel):
    anthropic: str = ""
    openai: str = ""
    polymarket_api_key: str = ""
    polymarket_secret: str = ""
    polymarket_passphrase: str = ""
    news_api: str = ""


class SwarmConfig(BaseModel):
    num_fish: int = 7
    personas: list[str] = Field(default_factory=lambda: [
        "geopolitical_analyst", "financial_quant", "bayesian_statistician",
        "investigative_journalist", "contrarian_thinker", "domain_expert",
        "calibration_specialist",
    ])
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.7
    max_concurrent: int = 5
    timeout_seconds: int = 30


class GodNodeConfig(BaseModel):
    model: str = "claude-opus-4-6"
    event_propagation: str = "broadcast"
    reanalysis_threshold: float = 0.15


class MarketSourceConfig(BaseModel):
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    chain_id: int = 137
    scan_interval_seconds: int = 30
    min_volume_usd: float = 10_000
    min_liquidity_usd: float = 5_000


class MarketsConfig(BaseModel):
    polymarket: MarketSourceConfig = Field(default_factory=MarketSourceConfig)


class SemanticConfig(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    news_sources: list[str] = Field(default_factory=lambda: ["tavily", "serper"])


class NetworkConfig(BaseModel):
    divergence_metric: str = "js"
    edge_threshold: float = 0.3
    update_interval_seconds: int = 300
    centrality_metric: str = "betweenness"


class PredictionConfig(BaseModel):
    aggregation: str = "bayesian_weighted"
    calibration_method: str = "isotonic"
    min_history_for_calibration: int = 100
    brier_score_target: float = 0.18
    confidence_threshold: float = 0.6


class RiskConfig(BaseModel):
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    bankroll_usd: float = 1_000
    paper_trading: bool = True


class VisualizationConfig(BaseModel):
    dashboard_port: int = 8050
    theme: str = "plotly_dark"
    auto_refresh_seconds: int = 60
    heatmap_colorscale: str = "RdYlGn"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/mirofish.log"
    rotation: str = "10 MB"
    retention: str = "30 days"


class MirofishConfig(BaseModel):
    """Top-level configuration for the Mirofish engine."""

    api_keys: ApiKeys = Field(default_factory=ApiKeys)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    god_node: GodNodeConfig = Field(default_factory=GodNodeConfig)
    markets: MarketsConfig = Field(default_factory=MarketsConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_dir: str | Path | None = None) -> MirofishConfig:
    """Load config from default.yaml, override with local.yaml if present."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "config"
    else:
        config_dir = Path(config_dir)

    default_path = config_dir / "default.yaml"
    local_path = config_dir / "local.yaml"

    if not default_path.exists():
        return MirofishConfig()

    with open(default_path, "r", encoding="utf-8") as f:
        base: dict[str, Any] = yaml.safe_load(f) or {}

    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local: dict[str, Any] = yaml.safe_load(f) or {}
        base = _deep_merge(base, local)

    # Environment variable overrides for secrets
    env_overrides = {
        "ANTHROPIC_API_KEY": ("api_keys", "anthropic"),
        "OPENAI_API_KEY": ("api_keys", "openai"),
        "POLYMARKET_API_KEY": ("api_keys", "polymarket_api_key"),
        "POLYMARKET_SECRET": ("api_keys", "polymarket_secret"),
        "POLYMARKET_PASSPHRASE": ("api_keys", "polymarket_passphrase"),
        "NEWS_API_KEY": ("api_keys", "news_api"),
    }
    for env_var, (section, key) in env_overrides.items():
        value = os.environ.get(env_var)
        if value:
            base.setdefault(section, {})[key] = value

    return MirofishConfig(**base)
