"""Generate K-Fish pipeline diagram as high-resolution PNG for GitHub README."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 22))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis('off')
fig.patch.set_facecolor('#0d1117')  # GitHub dark mode

# ── Colors ──
C_BG = '#0d1117'
C_BOX = '#161b22'
C_BORDER = '#30363d'
C_ACCENT = '#58a6ff'  # blue
C_GREEN = '#3fb950'
C_ORANGE = '#d29922'
C_RED = '#f85149'
C_PURPLE = '#bc8cff'
C_TEXT = '#c9d1d9'
C_TEXT_DIM = '#8b949e'
C_FISH = '#1f6feb'

def draw_box(x, y, w, h, title, subtitle="", color=C_ACCENT, filled=False):
    bg = color + '20' if not filled else color + '40'
    box = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=bg, edgecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.3, title,
        ha='center', va='top', fontsize=11, fontweight='bold',
        color=color, fontfamily='monospace')
    if subtitle:
        ax.text(x + w/2, y + 0.35, subtitle,
            ha='center', va='bottom', fontsize=8,
            color=C_TEXT_DIM, fontfamily='monospace',
            style='italic')

def draw_arrow(x1, y1, x2, y2, color=C_TEXT_DIM):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

def draw_fish_row(y_center, labels, color=C_FISH):
    n = len(labels)
    spacing = 12.0 / n
    x_start = 1.0 + spacing / 2
    for i, label in enumerate(labels):
        x = x_start + i * spacing
        circle = plt.Circle((x, y_center), 0.35,
            facecolor=color + '30', edgecolor=color, linewidth=1.2)
        ax.add_patch(circle)
        # Split label into short form
        short = label[:3].upper()
        ax.text(x, y_center, short,
            ha='center', va='center', fontsize=7, fontweight='bold',
            color=color, fontfamily='monospace')
        ax.text(x, y_center - 0.55, label,
            ha='center', va='top', fontsize=5.5,
            color=C_TEXT_DIM, fontfamily='monospace')

# ── Title ──
ax.text(7, 21.5, 'K-FISH', ha='center', va='center',
    fontsize=32, fontweight='bold', color=C_ACCENT, fontfamily='monospace')
ax.text(7, 20.9, 'Swarm Intelligence Prediction Pipeline',
    ha='center', va='center', fontsize=12, color=C_TEXT_DIM, fontfamily='monospace')

# ── Step 1: Market Question ──
draw_box(3.5, 19.8, 7, 0.9, 'MARKET QUESTION', 'price withheld from Fish agents', C_TEXT)
draw_arrow(7, 19.8, 7, 19.15, C_TEXT_DIM)

# ── Step 2: SwarmRouter ──
draw_box(2.5, 18.1, 9, 0.95,
    'SWARM ROUTER',
    'classifies category  |  selects personas + rounds + extremize',
    C_ORANGE)
draw_arrow(7, 18.1, 7, 17.45, C_ORANGE)

# ── Step 3: Researcher ──
draw_box(2.5, 16.4, 9, 0.95,
    'RESEARCHER FISH',
    'base rates  |  key facts  |  timing  |  contrarian case',
    C_PURPLE)
draw_arrow(7, 16.4, 7, 15.6, C_PURPLE)

# ── Step 4: Delphi rounds ──
# Round box background
round_bg = FancyBboxPatch((1.0, 12.5), 12, 3.0,
    boxstyle="round,pad=0.2",
    facecolor=C_FISH + '10', edgecolor=C_FISH, linewidth=1.5, linestyle='--')
ax.add_patch(round_bg)
ax.text(7, 15.3, 'MULTI-ROUND DELPHI PROTOCOL',
    ha='center', va='center', fontsize=11, fontweight='bold',
    color=C_FISH, fontfamily='monospace')

# Round 1
ax.text(3, 14.7, 'Round 1', ha='center', fontsize=9, fontweight='bold', color=C_FISH, fontfamily='monospace')
ax.text(3, 14.35, 'Independent', ha='center', fontsize=7, color=C_TEXT_DIM, fontfamily='monospace')

# Round 2
ax.text(7, 14.7, 'Round 2', ha='center', fontsize=9, fontweight='bold', color=C_FISH, fontfamily='monospace')
ax.text(7, 14.35, 'Peer context', ha='center', fontsize=7, color=C_TEXT_DIM, fontfamily='monospace')

# Round N
ax.text(11, 14.7, 'Round N', ha='center', fontsize=9, fontweight='bold', color=C_FISH, fontfamily='monospace')
ax.text(11, 14.35, 'Converge', ha='center', fontsize=7, color=C_TEXT_DIM, fontfamily='monospace')

# Arrows between rounds
draw_arrow(4.5, 14.5, 5.5, 14.5, C_FISH)
draw_arrow(8.5, 14.5, 9.5, 14.5, C_FISH)

# Fish circles
fish_names = ['Anchor', 'Decomp', 'Inside', 'Contra', 'Temporal',
              'Institut', 'Premort', 'Calibr', 'Bayes']
draw_fish_row(13.2, fish_names)

draw_arrow(7, 12.5, 7, 11.85, C_FISH)

# ── Step 5: Aggregation ──
draw_box(2.0, 10.8, 10, 0.95,
    'AGGREGATION',
    'trimmed mean + confidence-weighted + asymmetric extremization',
    C_GREEN)
draw_arrow(7, 10.8, 7, 10.15, C_GREEN)

# ── Step 6: Three parallel boxes ──
draw_box(0.5, 9.0, 3.8, 1.05, 'CALIBRATE', 'netcal auto-select\nBeta / Histogram', C_ACCENT)
draw_box(5.1, 9.0, 3.8, 1.05, 'VOLATILITY', 'GARCH regime\nKelly adjust', C_ORANGE)
draw_box(9.7, 9.0, 3.8, 1.05, 'CONFORMAL', '90% coverage\ninterval bounds', C_PURPLE)

# Arrows from aggregation to parallel
draw_arrow(4, 10.15, 2.4, 10.05, C_TEXT_DIM)
draw_arrow(7, 10.15, 7, 10.05, C_TEXT_DIM)
draw_arrow(10, 10.15, 11.6, 10.05, C_TEXT_DIM)

# Arrows from parallel down
draw_arrow(2.4, 9.0, 5, 8.3, C_TEXT_DIM)
draw_arrow(7, 9.0, 7, 8.3, C_TEXT_DIM)
draw_arrow(11.6, 9.0, 9, 8.3, C_TEXT_DIM)

# ── Step 7: Edge Detection ──
draw_box(2.5, 7.2, 9, 1.0,
    'EDGE DETECTION',
    '|cal_prob - mkt_price| > 7%  |  confidence > 40%  |  spread < 35%',
    C_ORANGE)
draw_arrow(7, 7.2, 7, 6.55, C_ORANGE)

# ── Step 8: Kelly Sizing ──
draw_box(2.5, 5.4, 9, 1.05,
    'KELLY SIZING',
    'quarter-Kelly  |  5% max/position  |  30% max exposure  |  15% drawdown stop',
    C_RED)
draw_arrow(7, 5.4, 7, 4.75, C_RED)

# ── Step 9: Position ──
draw_box(3.0, 3.6, 8, 1.05,
    'POSITION',
    'side (YES/NO)  |  size ($)  |  expected value  |  reasoning chain',
    C_GREEN, filled=True)

# ── Footer ──
ax.text(7, 2.8, '$0.00 per prediction  |  9 Fish  |  135s per market',
    ha='center', fontsize=9, color=C_TEXT_DIM, fontfamily='monospace')

ax.text(7, 2.2, 'K-Fish v4  |  github.com/ksk5429/quant',
    ha='center', fontsize=8, color=C_ACCENT, fontfamily='monospace')

# ── Save ──
plt.tight_layout()
fig.savefig('docs/pipeline_diagram.png', dpi=200, bbox_inches='tight',
    facecolor=C_BG, edgecolor='none')
fig.savefig('docs/pipeline_diagram_light.png', dpi=200, bbox_inches='tight',
    facecolor='white', edgecolor='none')
print('Saved: docs/pipeline_diagram.png (dark) + pipeline_diagram_light.png (light)')
