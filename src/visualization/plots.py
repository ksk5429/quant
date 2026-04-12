"""Visualization functions for the Mirofish prediction engine.

Provides:
1. Correlation heatmaps (market-market semantic/price similarity)
2. Network graphs (interactive market relationship visualization)
3. Probability distributions (swarm consensus + individual Fish)
4. Calibration plots (reliability diagrams)
5. Performance dashboards (Brier scores over time, PnL curves)
6. Signal processing views (edge detection, mean reversion indicators)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger


def plot_correlation_heatmap(
    similarity_matrix: np.ndarray,
    labels: list[str],
    title: str = "Market Correlation Heatmap",
    colorscale: str = "RdYlGn",
) -> go.Figure:
    """Interactive heatmap of market-market correlations.

    Colors encode strength: red = anti-correlated, green = correlated.
    Hover shows exact correlation value and market pair.
    """
    # Truncate long labels
    short_labels = [l[:40] + "..." if len(l) > 40 else l for l in labels]

    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=short_labels,
        y=short_labels,
        colorscale=colorscale,
        zmin=-1, zmax=1,
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        width=800, height=800,
        xaxis=dict(tickangle=45),
    )
    return fig


def plot_market_network(
    adjacency_matrix: np.ndarray,
    labels: list[str],
    node_sizes: list[float] | None = None,
    node_colors: list[float] | None = None,
    title: str = "Market Correlation Network",
) -> go.Figure:
    """Interactive network graph of market relationships.

    Node size = volume/importance, edge thickness = correlation strength,
    color = category or centrality score.
    """
    import networkx as nx

    # Build NetworkX graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    pos = nx.spring_layout(G, seed=42, k=2 / np.sqrt(len(labels)))

    # Create edge traces
    edge_x, edge_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node traces
    node_x = [pos[i][0] for i in range(len(labels))]
    node_y = [pos[i][1] for i in range(len(labels))]

    if node_sizes is None:
        node_sizes = [15] * len(labels)
    if node_colors is None:
        node_colors = [G.degree(i) for i in range(len(labels))]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[l[:25] for l in labels],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Viridis",
            colorbar=dict(title="Centrality"),
            line=dict(width=1, color="white"),
        ),
    )

    # Hover text
    hover_texts = []
    for i, label in enumerate(labels):
        degree = G.degree(i)
        hover_texts.append(f"{label}<br>Connections: {degree}")
    node_trace.hovertext = hover_texts

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        template="plotly_dark",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=900, height=700,
    )
    return fig


def plot_swarm_prediction(
    fish_probabilities: list[float],
    fish_personas: list[str],
    fish_confidences: list[float],
    swarm_probability: float,
    market_price: float | None = None,
    market_question: str = "",
) -> go.Figure:
    """Visualize individual Fish predictions and swarm consensus.

    Shows:
    - Bar chart of each Fish's probability estimate (colored by persona)
    - Horizontal line for swarm consensus
    - Horizontal line for market price (if available)
    - Confidence as bar opacity
    """
    fig = make_subplots(rows=1, cols=1)

    # Individual Fish predictions
    colors = px.colors.qualitative.Set2[:len(fish_personas)]
    opacities = [0.4 + 0.6 * c for c in fish_confidences]

    fig.add_trace(go.Bar(
        x=fish_personas,
        y=fish_probabilities,
        marker=dict(
            color=colors,
            opacity=opacities,
            line=dict(width=1, color="white"),
        ),
        text=[f"{p:.3f}" for p in fish_probabilities],
        textposition="outside",
        name="Fish Estimates",
        hovertemplate="<b>%{x}</b><br>P=%{y:.3f}<extra></extra>",
    ))

    # Swarm consensus line
    fig.add_hline(
        y=swarm_probability,
        line=dict(color="gold", width=3, dash="dash"),
        annotation_text=f"Swarm: {swarm_probability:.3f}",
        annotation_position="top right",
    )

    # Market price line
    if market_price is not None:
        fig.add_hline(
            y=market_price,
            line=dict(color="red", width=2, dash="dot"),
            annotation_text=f"Market: {market_price:.3f}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=f"Swarm Analysis: {market_question[:60]}...",
        template="plotly_dark",
        yaxis=dict(title="Probability", range=[0, 1.05]),
        xaxis=dict(title="Fish Persona"),
        width=800, height=500,
    )
    return fig


def plot_calibration_diagram(
    predicted_probs: list[float],
    actual_outcomes: list[float],
    n_bins: int = 10,
    title: str = "Calibration Diagram (Reliability Plot)",
) -> go.Figure:
    """Reliability diagram for assessing probability calibration.

    Perfectly calibrated predictions fall on the diagonal.
    Points above = underconfident, below = overconfident.
    """
    preds = np.array(predicted_probs)
    outs = np.array(actual_outcomes)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if np.any(mask):
            bin_centers.append(float(np.mean(preds[mask])))
            bin_accuracies.append(float(np.mean(outs[mask])))
            bin_counts.append(int(np.sum(mask)))

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    # Reliability curve
    fig.add_trace(
        go.Scatter(
            x=bin_centers, y=bin_accuracies,
            mode="lines+markers",
            name="Calibration",
            marker=dict(size=10, color="cyan"),
            line=dict(width=2),
        ),
        row=1, col=1,
    )

    # Perfect calibration diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", dash="dash"),
        ),
        row=1, col=1,
    )

    # Histogram of predictions
    fig.add_trace(
        go.Bar(
            x=bin_centers, y=bin_counts,
            name="Sample Count",
            marker=dict(color="rgba(0, 200, 200, 0.5)"),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        width=700, height=600,
    )
    fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)
    fig.update_yaxes(title_text="Observed Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig


def plot_pnl_curve(
    timestamps: list[float],
    bankroll_history: list[float],
    trade_markers: list[dict[str, Any]] | None = None,
    title: str = "Portfolio Performance",
) -> go.Figure:
    """PnL curve with trade markers and drawdown shading."""
    import pandas as pd

    fig = go.Figure()

    # Bankroll line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=bankroll_history,
        mode="lines",
        name="Bankroll",
        line=dict(color="lime", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 255, 0, 0.05)",
    ))

    # Drawdown shading
    peak = np.maximum.accumulate(bankroll_history)
    drawdown = (peak - np.array(bankroll_history)) / peak * 100

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=-drawdown,
        mode="lines",
        name="Drawdown %",
        line=dict(color="red", width=1),
        yaxis="y2",
    ))

    # Trade markers
    if trade_markers:
        buy_x = [t["time"] for t in trade_markers if t.get("side") == "yes"]
        buy_y = [t["bankroll"] for t in trade_markers if t.get("side") == "yes"]
        sell_x = [t["time"] for t in trade_markers if t.get("side") == "no"]
        sell_y = [t["bankroll"] for t in trade_markers if t.get("side") == "no"]

        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y,
            mode="markers", name="YES Bets",
            marker=dict(symbol="triangle-up", size=10, color="green"),
        ))
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y,
            mode="markers", name="NO Bets",
            marker=dict(symbol="triangle-down", size=10, color="red"),
        ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Bankroll ($)"),
        yaxis2=dict(title="Drawdown (%)", overlaying="y", side="right"),
        width=900, height=500,
    )
    return fig


def plot_edge_distribution(
    edges: list[float],
    title: str = "Edge Distribution (Our Prob - Market Price)",
) -> go.Figure:
    """Histogram of prediction edges across all analyzed markets."""
    fig = go.Figure(data=go.Histogram(
        x=edges,
        nbinsx=50,
        marker=dict(color="rgba(0, 200, 200, 0.7)", line=dict(width=1, color="white")),
    ))

    fig.add_vline(x=0, line=dict(color="yellow", dash="dash"))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(title="Edge (Our P - Market P)"),
        yaxis=dict(title="Count"),
        width=700, height=400,
    )
    return fig
