"""Generate K-Fish pipeline diagram — large, readable, professional."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def generate(bg_color, text_color, dim_color, filename):
    fig, ax = plt.subplots(1, 1, figsize=(20, 38))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 38)
    ax.axis('off')
    fig.patch.set_facecolor(bg_color)

    # Colors
    C_BLUE = '#58a6ff'
    C_GREEN = '#3fb950'
    C_ORANGE = '#d29922'
    C_RED = '#f85149'
    C_PURPLE = '#bc8cff'
    C_FISH = '#1f6feb'

    def box(x, y, w, h, title, subtitle, color, title_size=19, sub_size=14, filled=False):
        alpha = '30' if not filled else '50'
        patch = FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.3", facecolor=color + alpha,
            edgecolor=color, linewidth=3)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h * 0.68, title,
            ha='center', va='center', fontsize=title_size, fontweight='bold',
            color=color, fontfamily='sans-serif')
        if subtitle:
            ax.text(x + w/2, y + h * 0.28, subtitle,
                ha='center', va='center', fontsize=sub_size,
                color=dim_color, fontfamily='sans-serif')

    def arrow(x1, y1, x2, y2, color=dim_color):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                            connectionstyle='arc3,rad=0'))

    cx = 10  # center x for 0-20 canvas

    # ── Title ──
    ax.text(cx, 37.0, 'K - F I S H', ha='center', va='center',
        fontsize=48, fontweight='bold', color=C_BLUE, fontfamily='sans-serif')
    ax.text(cx, 36.0, 'Swarm Intelligence Prediction Pipeline',
        ha='center', va='center', fontsize=18, color=dim_color, fontfamily='sans-serif')

    # ── 1. Market Question ──
    y = 33.8
    box(4, y, 12, 1.8, 'MARKET QUESTION', 'price withheld from all Fish agents', text_color)
    arrow(cx, y, cx, y - 0.7)

    # ── 2. SwarmRouter ──
    y = 31.2
    box(2.5, y, 15, 2.0, 'SWARM ROUTER',
        'classifies category  ·  selects personas  ·  sets rounds  ·  sets extremization', C_ORANGE)
    arrow(cx, y, cx, y - 0.7)

    # ── 3. Researcher ──
    y = 28.4
    box(2.5, y, 15, 2.0, 'RESEARCHER FISH',
        'base rates  ·  key facts  ·  timing analysis  ·  contrarian case', C_PURPLE)
    arrow(cx, y, cx, y - 0.7)

    # ── 4. Delphi ──
    delphi_top = 27.0
    delphi_bot = 19.5
    delphi_h = delphi_top - delphi_bot
    patch = FancyBboxPatch((1.5, delphi_bot), 17, delphi_h,
        boxstyle="round,pad=0.4", facecolor=C_FISH + '12',
        edgecolor=C_FISH, linewidth=3, linestyle='--')
    ax.add_patch(patch)

    ax.text(cx, 26.3, 'MULTI-ROUND DELPHI PROTOCOL',
        ha='center', va='center', fontsize=20, fontweight='bold',
        color=C_FISH, fontfamily='sans-serif')

    # Round labels
    for rx, label, sub in [(5, 'Round 1', 'Independent'), (10, 'Round 2', 'Peer Context'), (15, 'Round N', 'Converge')]:
        ax.text(rx, 25.2, label, ha='center', fontsize=17, fontweight='bold',
            color=C_FISH, fontfamily='sans-serif')
        ax.text(rx, 24.5, sub, ha='center', fontsize=14,
            color=dim_color, fontfamily='sans-serif')

    arrow(7, 24.8, 8, 24.8, C_FISH)
    arrow(12, 24.8, 13, 24.8, C_FISH)

    # Fish circles
    fish_data = [
        ('ANC', 'Anchor'), ('DEC', 'Decomp'), ('INS', 'Inside'),
        ('CON', 'Contra'), ('TMP', 'Temporal'), ('IST', 'Institut'),
        ('PRE', 'Premort'), ('CAL', 'Calibr'), ('BAY', 'Bayes'),
    ]
    n = len(fish_data)
    spacing = 16.0 / n
    x_start = 2.0 + spacing / 2
    fish_y = 22.2

    for i, (short, full) in enumerate(fish_data):
        x = x_start + i * spacing
        circle = plt.Circle((x, fish_y), 0.65,
            facecolor=C_FISH + '30', edgecolor=C_FISH, linewidth=2.5)
        ax.add_patch(circle)
        ax.text(x, fish_y + 0.05, short, ha='center', va='center',
            fontsize=12, fontweight='bold', color=C_FISH, fontfamily='sans-serif')
        ax.text(x, fish_y - 0.95, full, ha='center', va='top',
            fontsize=11, color=dim_color, fontfamily='sans-serif')

    probs = ['0.72', '0.68', '0.71', '0.45', '0.63', '0.67', '0.58', '0.70', '0.69']
    for i, p in enumerate(probs):
        x = x_start + i * spacing
        ax.text(x, fish_y - 1.6, p, ha='center', fontsize=11,
            color=C_FISH, fontfamily='monospace')

    arrow(cx, delphi_bot, cx, delphi_bot - 0.7)

    # ── 5. Aggregation ──
    y = 16.8
    box(2.5, y, 15, 2.0, 'AGGREGATION',
        'trimmed mean  ·  confidence-weighted  ·  asymmetric extremization', C_GREEN)
    arrow(cx, y, cx, y - 0.7)

    # ── 6. Three parallel boxes ──
    py = 13.5
    pw = 5.2
    ph = 2.2
    gap = 0.4
    x1 = 1.0
    x2 = x1 + pw + gap
    x3 = x2 + pw + gap
    box(x1, py, pw, ph, 'CALIBRATE', 'netcal auto-select\nBeta · Histogram · Isotonic', C_BLUE, title_size=17, sub_size=13)
    box(x2, py, pw, ph, 'VOLATILITY', 'GARCH regime detection\nKelly adjustment factor', C_ORANGE, title_size=17, sub_size=13)
    box(x3, py, pw, ph, 'CONFORMAL', '90% coverage interval\nprediction bounds', C_PURPLE, title_size=17, sub_size=13)

    arrow(5.5, y - 0.1, x1 + pw/2, py + ph)
    arrow(cx, y - 0.1, x2 + pw/2, py + ph)
    arrow(14.5, y - 0.1, x3 + pw/2, py + ph)

    arrow(x1 + pw/2, py, 7, py - 1.0)
    arrow(x2 + pw/2, py, cx, py - 1.0)
    arrow(x3 + pw/2, py, 13, py - 1.0)

    # ── 7. Edge Detection ──
    y = 10.3
    box(2.5, y, 15, 2.0, 'EDGE DETECTION',
        '|calibrated - market_price| > 7%  ·  confidence > 40%  ·  spread < 35%', C_ORANGE)
    arrow(cx, y, cx, y - 0.7)

    # ── 8. Kelly Sizing ──
    y = 7.5
    box(2.5, y, 15, 2.0, 'KELLY SIZING',
        'quarter-Kelly  ·  5% max / position  ·  30% max exposure  ·  15% drawdown stop', C_RED)
    arrow(cx, y, cx, y - 0.7)

    # ── 9. Position ──
    y = 4.7
    box(3.5, y, 13, 2.0, 'POSITION',
        'side (YES / NO)  ·  size ($)  ·  expected value  ·  reasoning chain', C_GREEN, filled=True)

    # ── Footer ──
    ax.text(cx, 3.5, '$0.00 per prediction   ·   9 Fish   ·   135 seconds per market',
        ha='center', fontsize=16, color=dim_color, fontfamily='sans-serif')
    ax.text(cx, 2.6, 'K-Fish v4   ·   github.com/ksk5429/quant',
        ha='center', fontsize=15, color=C_BLUE, fontfamily='sans-serif')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
    fig.savefig(filename, dpi=150, bbox_inches='tight',
        facecolor=bg_color, edgecolor='none')
    plt.close(fig)
    print(f'Saved: {filename}')


# Dark theme (GitHub dark mode)
generate(
    bg_color='#0d1117',
    text_color='#c9d1d9',
    dim_color='#8b949e',
    filename='docs/pipeline_diagram.png',
)

# Light theme
generate(
    bg_color='#ffffff',
    text_color='#1f2328',
    dim_color='#656d76',
    filename='docs/pipeline_diagram_light.png',
)
