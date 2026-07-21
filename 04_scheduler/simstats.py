"""Shared statistics helpers for the scheduler benchmarks.

Extracted from multi_scheduler_benchmark.py (v3.3) so that the synthetic and
the trace-driven benchmarks apply *identical* fairness and significance
machinery — a reviewer comparing the two studies is then comparing workloads,
not two accidentally-different statistical treatments.
"""

import numpy as np
import pandas as pd
from scipy import stats


def gini(vals):
    """Gini coefficient of a non-negative sample (0 = perfectly equal)."""
    x = np.array([v for v in vals if v >= 0], dtype=float)
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def holm_correction(pvals):
    """Holm step-down adjusted p-values (family-wise error control).
    NaN inputs stay NaN and do not count toward the family size m."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    m = len(valid)
    order = valid[np.argsort(p[valid])]
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adj[idx] = min(running_max, 1.0)
    return adj


def tost_equivalence(a, b, margin_frac=0.10, alpha=0.05):
    """Paired TOST (two one-sided tests) for EQUIVALENCE of a and b.

    A large p-value from a difference test means "we failed to detect a
    difference" -- it is NOT evidence that two schedulers perform the same.
    Claiming equivalence requires rejecting the two composite nulls
        H01: mu_d <= -Delta      H02: mu_d >= +Delta
    where Delta = margin_frac * mean(b) is the largest difference we are
    willing to call practically negligible. p_TOST = max(p1, p2); p < alpha
    means the difference is significantly WITHIN +/- Delta, i.e. equivalent.

    Returns a dict with the margin, both one-sided p-values, p_TOST, the
    (1-2*alpha) CI of the paired difference (the interval TOST inverts), and
    the verdict.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    delta = abs(margin_frac * b.mean())
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n) if n > 1 else np.nan
    if not np.isfinite(se) or se == 0:
        within = abs(d.mean()) < delta
        return {'n': n, 'margin': delta, 'mean_diff': float(d.mean()),
                'p_lower': 0.0 if within else 1.0,
                'p_upper': 0.0 if within else 1.0,
                'p_tost': 0.0 if within else 1.0,
                'ci_low': float(d.mean()), 'ci_high': float(d.mean()),
                'equivalent': bool(within)}
    df = n - 1
    t_lower = (d.mean() + delta) / se      # H01: mu <= -Delta
    t_upper = (d.mean() - delta) / se      # H02: mu >= +Delta
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)
    crit = stats.t.ppf(1 - alpha, df)
    return {
        'n': n,
        'margin': float(delta),
        'mean_diff': float(d.mean()),
        'p_lower': p_lower,
        'p_upper': p_upper,
        'p_tost': float(p_tost),
        'ci_low': float(d.mean() - crit * se),
        'ci_high': float(d.mean() + crit * se),
        'equivalent': bool(p_tost < alpha),
    }


def equivalence_table(runs_df, pairs, metric='mean_wait', unit='run',
                      margin_frac=0.10):
    """Run tost_equivalence over a list of (scheduler_a, scheduler_b) pairs."""
    rows = []
    for a_name, b_name in pairs:
        a = (runs_df[runs_df['scheduler'] == a_name]
             .sort_values(unit)[metric].to_numpy())
        b = (runs_df[runs_df['scheduler'] == b_name]
             .sort_values(unit)[metric].to_numpy())
        if len(a) == 0 or len(b) == 0 or len(a) != len(b):
            continue
        res = tost_equivalence(a, b, margin_frac=margin_frac)
        res.update({'scheduler_a': a_name, 'scheduler_b': b_name,
                    'metric': metric, 'margin_frac': margin_frac,
                    'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
                    'pct_diff': float((a.mean() - b.mean()) / b.mean() * 100.0)})
        rows.append(res)
    cols = ['scheduler_a', 'scheduler_b', 'metric', 'n', 'mean_a', 'mean_b',
            'pct_diff', 'margin_frac', 'margin', 'mean_diff', 'ci_low',
            'ci_high', 'p_lower', 'p_upper', 'p_tost', 'equivalent']
    return pd.DataFrame(rows, columns=cols)


def pairwise_significance(runs_df, references=('PROACTIVE', 'FIFO'),
                          metric='mean_wait', unit='run'):
    """Paired per-unit comparison of every scheduler against each reference:
    paired t-test, Wilcoxon signed-rank, Cohen's dz, and Holm-adjusted p
    within each reference family.

    `unit` is the pairing column ('run' for the synthetic benchmark, 'window'
    for the trace-driven one). diff > 0 means the scheduler scores HIGHER on
    `metric` than the reference (for wait/slowdown metrics: it is worse).
    """
    rows = []
    for ref in references:
        ref_w = (runs_df[runs_df['scheduler'] == ref]
                 .sort_values(unit)[metric].to_numpy())
        fam = []
        for sch in sorted(runs_df['scheduler'].unique()):
            if sch == ref:
                continue
            w = (runs_df[runs_df['scheduler'] == sch]
                 .sort_values(unit)[metric].to_numpy())
            diff = w - ref_w
            tt = stats.ttest_rel(w, ref_w)
            try:
                wp = float(stats.wilcoxon(w, ref_w, zero_method='wilcox').pvalue)
            except ValueError:
                wp = float('nan')
            sd = diff.std(ddof=1)
            fam.append({
                'reference': ref,
                'scheduler': sch,
                'metric': metric,
                f'{metric}_sched': float(w.mean()),
                f'{metric}_ref': float(ref_w.mean()),
                'mean_diff': float(diff.mean()),
                'pct_vs_ref': float(diff.mean() / ref_w.mean() * 100.0),
                'cohens_dz': float(diff.mean() / sd) if sd > 0 else float('nan'),
                'ttest_p': float(tt.pvalue),
                'wilcoxon_p': wp,
            })
        adj_t = holm_correction([r['ttest_p'] for r in fam])
        adj_w = holm_correction([r['wilcoxon_p'] for r in fam])
        for r, at, aw in zip(fam, adj_t, adj_w):
            r['ttest_p_holm'] = float(at)
            r['wilcoxon_p_holm'] = float(aw)
        rows.extend(fam)
    return pd.DataFrame(rows)
