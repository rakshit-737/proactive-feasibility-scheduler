"""Shared figure style for every chart in this repository.

WHY THIS EXISTS
---------------
The repo had 46 figures in default matplotlib styling, which produced two real
defects, not just inconsistency:

  * the same scheduler was drawn in DIFFERENT colours in adjacent panels of the
    same figure (blue in one, green in the next), so colour tracked the panel
    rather than the entity;
  * panels encoded features by index instead of name, so the reader could not
    tell which feature a mark referred to.

This module fixes the cause: one palette, one semantic colour->role mapping
applied everywhere, and helpers that make the light/dark pair automatic.

COLOUR MODEL (validated, not eyeballed)
---------------------------------------
Emphasis encoding, per the data-viz method: the story of almost every chart here
is "the ML policy lands exactly where the ML-free control lands", so two entities
carry colour and everything else recedes to neutral ink.

    ml        blue    the proposed ML scheduler (PROACTIVE / PROACTIVE_EST)
    control   orange  the ML-free control it is being tested against
    baseline  gray    classical baselines -- present, recessive, never the point

Blue/orange are categorical slots 1 and 2 of the reference palette. Validated
with scripts/validate_palette.js on the all-pairs list in BOTH modes:
light  CVD dE 24.7 (protan), normal-vision dE 33.6, contrast >= 3:1 -- all PASS
dark   CVD dE 26.8 (protan), normal-vision dE 31.8, contrast >= 3:1 -- all PASS

Oracle policies are NOT given a fourth colour; they are marked in the tick label
("(oracle)"), because identity is already carried by the label and a fourth hue
would spend the last free channel on something the text already says.

USAGE
-----
    import sys, os
    sys.path.insert(0, PROJECT_ROOT)
    from vizstyle import figure, finish, save_both, role_of, ROLE_COLORS

    for mode in ('light', 'dark'):
        fig, ax = figure(mode, figsize=(9, 5))
        ...
        finish(fig, mode, title='...', source='05_results/...')
        save_both(fig, path_stem, mode)
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Palette (reference instance; light | dark steps of the same hues)
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    'light': {
        'surface':   '#fcfcfb',
        'page':      '#f9f9f7',
        'ink':       '#0b0b0b',
        'ink_2':     '#52514e',
        'muted':     '#898781',
        'grid':      '#e1e0d9',
        'axis':      '#c3c2b7',
        'series_1':  '#2a78d6',   # blue
        'series_2':  '#eb6834',   # orange
        'critical':  '#d03b3b',
        'good':      '#0ca30c',
    },
    'dark': {
        'surface':   '#1a1a19',
        'page':      '#0d0d0d',
        'ink':       '#ffffff',
        'ink_2':     '#c3c2b7',
        'muted':     '#898781',
        'grid':      '#2c2c2a',
        'axis':      '#383835',
        'series_1':  '#3987e5',   # blue
        'series_2':  '#d95926',   # orange
        'critical':  '#d03b3b',
        'good':      '#0ca30c',
    },
}

# Role -> palette key. Baselines use the muted ink: recessive by construction.
ROLE_KEY = {'ml': 'series_1', 'control': 'series_2', 'baseline': 'muted'}

ROLE_LABEL = {
    'ml': 'Proactive (ML wait-time model)',
    'control': 'Smallest-first (no ML)',
    'baseline': 'Classical baselines',
}

# Every scheduler key used anywhere in the repo -> its role. Colour follows the
# ENTITY, so a policy keeps its colour across panels, figures and filters.
POLICY_ROLE = {
    # the proposed method (both benchmarks' spellings)
    'PROACTIVE': 'ml', 'PROACTIVE_EST': 'ml', 'PROACTIVE_BF': 'ml', 'NN': 'ml',
    # the ML-free control
    'SMALLEST': 'control', 'SMALLEST_FIRST': 'control',
    # classical baselines
    'FIFO': 'baseline', 'FCFS': 'baseline', 'FIFO_STRICT': 'baseline',
    'SJF': 'baseline', 'SJF_EST': 'baseline', 'SJF_MODAL': 'baseline',
    'SJF_ORACLE': 'baseline', 'SJF_USEREST': 'baseline',
    'HRRN': 'baseline', 'HRRN_USEREST': 'baseline', 'PRIORITY': 'baseline',
    'BACKFILL': 'baseline', 'BACKFILL_EST': 'baseline', 'BACKFILL_MODAL': 'baseline',
    'EASY_ORACLE': 'baseline', 'EASY_USEREST': 'baseline',
    'CONS_BF': 'baseline', 'CONS_BF_USEREST': 'baseline',
    'SRPT': 'baseline', 'SRPT_ORACLE': 'baseline',
}

# Human-readable tick labels. Oracle policies are marked in TEXT rather than
# given their own hue -- the label already carries that identity.
POLICY_LABEL = {
    'SRPT_ORACLE': 'SRPT (oracle, preemptive)',
    'SJF_ORACLE': 'SJF (oracle)',
    'SJF_USEREST': 'SJF (real user estimates)',
    'HRRN_USEREST': 'HRRN (real estimates)',
    'PROACTIVE_EST': 'Proactive + estimate feature',
    'SMALLEST_FIRST': 'Smallest-first (no ML)',
    'PROACTIVE': 'Proactive (XGBoost)',
    'FCFS': 'FCFS + first-fit',
    'EASY_ORACLE': 'EASY backfill (oracle)',
    'EASY_USEREST': 'EASY backfill (real estimates)',
    'CONS_BF_USEREST': 'Conservative backfill',
    'FIFO_STRICT': 'Strict FIFO',
    'SJF': 'SJF (oracle)', 'SJF_EST': 'SJF (f-model estimates)',
    'SJF_MODAL': 'SJF (modal estimates)', 'HRRN': 'HRRN',
    'SMALLEST': 'Smallest-first (no ML)', 'NN': 'Neural network (MLP)',
    'PRIORITY': 'Priority + aging', 'FIFO': 'FCFS + first-fit',
    'BACKFILL': 'EASY backfill (oracle)',
    'BACKFILL_EST': 'EASY backfill (estimates)',
    'CONS_BF': 'Conservative backfill', 'SRPT': 'SRPT (oracle, preemptive)',
    'PROACTIVE_BF': 'EASY + predicted-wait scan',
}


def role_of(policy):
    return POLICY_ROLE.get(str(policy).upper(), 'baseline')


def color_of(policy, mode='light'):
    """Colour for a scheduler, by role. Same entity -> same colour, always."""
    return PALETTE[mode][ROLE_KEY[role_of(policy)]]


def label_of(policy):
    return POLICY_LABEL.get(str(policy).upper(), str(policy))


def ROLE_COLORS(mode='light'):
    return {r: PALETTE[mode][k] for r, k in ROLE_KEY.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Figure construction
# ─────────────────────────────────────────────────────────────────────────────

def apply_rc(mode='light'):
    """Recessive chrome: hairline SOLID grid, no top/right spines, system sans."""
    p = PALETTE[mode]
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 10,
        'figure.facecolor': p['surface'],
        'axes.facecolor': p['surface'],
        'savefig.facecolor': p['surface'],
        'savefig.edgecolor': p['surface'],
        'text.color': p['ink'],
        'axes.labelcolor': p['ink_2'],
        'axes.edgecolor': p['axis'],
        'axes.linewidth': 0.8,
        'axes.titlecolor': p['ink'],
        'axes.titlesize': 11,
        'axes.titleweight': 'semibold',
        'axes.labelsize': 9.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': p['grid'],
        'grid.linewidth': 0.7,
        'grid.linestyle': '-',          # never dashed
        'xtick.color': p['muted'],
        'ytick.color': p['muted'],
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.labelcolor': p['ink_2'],
        'ytick.labelcolor': p['ink_2'],
        'legend.frameon': False,
        'legend.fontsize': 9,
        'legend.labelcolor': p['ink_2'],
        'lines.linewidth': 2.0,
        'lines.markersize': 5.5,
        'patch.linewidth': 0,
    })


def figure(mode='light', figsize=(9, 5), nrows=1, ncols=1, **kw):
    """Create a styled figure/axes pair (or grid) for the given mode."""
    apply_rc(mode)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    return fig, axes


def finish(fig, mode='light', title=None, subtitle=None, source=None,
           title_y=0.985, subtitle_y=None, source_y=0.006):
    """Title block + provenance footer. Text wears ink tokens, never series colour.

    The y positions are figure fractions, so a taller figure needs them passed
    explicitly — the defaults suit a ~5in-tall figure. Callers that stack rows
    should raise `subtitle_y` and lower `subplots_adjust(top=...)` together, and
    then LOOK at the render: the title block silently overlapping the first row
    of axes titles is the failure mode these parameters exist to prevent.
    """
    p = PALETTE[mode]
    if title:
        fig.suptitle(title, color=p['ink'], fontsize=13.5, fontweight='semibold',
                     x=0.012, ha='left', y=title_y)
    if subtitle:
        if subtitle_y is None:
            subtitle_y = 0.90 if title else 0.94
        fig.text(0.012, subtitle_y, subtitle, color=p['ink_2'], fontsize=10,
                 ha='left', va='top')
    if source:
        fig.text(0.012, source_y, source, color=p['muted'], fontsize=7.6,
                 ha='left')
    return fig


def save_both(fig, stem, mode, dpi=200):
    """Write `<stem>.png` for light and `<stem>-dark.png` for dark.

    The pair lets a README use <picture> with prefers-color-scheme so the figure
    matches the reader's theme instead of glowing white in a dark viewer.
    """
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    path = f'{stem}.png' if mode == 'light' else f'{stem}-dark.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor=PALETTE[mode]['surface'])
    plt.close(fig)
    return path


def bar_ends(ax, orientation='h'):
    """Hairline baseline at the data origin; the opposite axis carries no rule."""
    if orientation == 'h':
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', visible=True)
        ax.spines['left'].set_visible(False)
    else:
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', visible=True)
        ax.spines['bottom'].set_visible(True)
    ax.set_axisbelow(True)
    return ax


def legend_roles(ax, mode='light', roles=('ml', 'control', 'baseline'), **kw):
    """Role legend. Identity is never colour-alone: the legend is always present
    for >= 2 series and the tick labels name every mark."""
    from matplotlib.patches import Patch
    p = PALETTE[mode]
    handles = [Patch(facecolor=p[ROLE_KEY[r]], label=ROLE_LABEL[r]) for r in roles]
    kw.setdefault('loc', 'lower right')
    return ax.legend(handles=handles, **kw)
