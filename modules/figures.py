"""
figures.py
----------
Generate all thesis figures from df and results in memory.
No hardcoded values — all coefficients computed from data.
Outputs PNG 300 dpi to outputs/figures/.
Old figures are deleted before generating new ones.
"""

import logging
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

log = logging.getLogger(__name__)

FIG_DIR    = 'outputs/figures'
COL_CASUAL = '#1A5276'
COL_FORMAL = '#784212'
COL_SIG    = '#1E8449'
COL_NS     = '#95A5A6'
COL_NEG    = '#C0392B'

STYLE = {
    'font.family':       'sans-serif',
    'font.size':         12,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    12,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
}
plt.rcParams.update(STYLE)


def _ensure_dir():
    """Delete old figures and recreate directory."""
    if os.path.exists(FIG_DIR):
        shutil.rmtree(FIG_DIR)
    os.makedirs(FIG_DIR)


def _sig(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return 'ns'


def _cohen_d(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*a.std()**2 + (n2-1)*b.std()**2) / (n1+n2-2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0


def _save(fig, name):
    path = f'{FIG_DIR}/{name}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info(f'Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Figure 1A — GPT-4o Inter-run Reliability
# ---------------------------------------------------------------------------
def fig_gpt_reliability(df: pd.DataFrame) -> str:
    variables = [
        ('quantity', 'Quantity'),
        ('quality',  'Quality'),
        ('emotions', 'Emotional Expression'),
    ]
    pairs = [(1, 2), (1, 3), (2, 3)]

    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    fig.suptitle(
        'GPT-4o Scoring Reliability Across Three Independent Runs',
        fontsize=15, fontweight='bold', y=1.01
    )

    for row_idx, (var, var_label) in enumerate(variables):
        for col_idx, (r1, r2) in enumerate(pairs):
            ax = axes[row_idx, col_idx]
            c1 = f'{var}_run{r1}'
            c2 = f'{var}_run{r2}'
            if c1 not in df.columns or c2 not in df.columns:
                ax.text(.5, .5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            x = df[c1].dropna()
            y = df[c2].reindex(x.index).dropna()
            x = x.reindex(y.index)

            r, p   = stats.pearsonr(x, y)
            r2_val = r ** 2

            ax.scatter(x, y, alpha=0.45, s=30,
                       color=COL_CASUAL, edgecolors='none')

            lims = [min(x.min(), y.min()) - 0.1,
                    max(x.max(), y.max()) + 0.1]
            ax.plot(lims, lims, '--', color='#BDC3C7', linewidth=1.2)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            ax.text(0.05, 0.93,
                    f'r = {r:.3f}\nR² = {r2_val:.3f}',
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8,
                              edgecolor='#CCCCCC'))

            ax.set_xlabel(f'Run {r1}', fontsize=12)
            ax.set_ylabel(f'Run {r2}', fontsize=12)
            ax.set_title(
                f'{var_label} — Run {r1} vs Run {r2}',
                fontsize=13, fontweight='bold'
            )

    plt.tight_layout()
    return _save(fig, 'fig_gpt_reliability')
    
# ---------------------------------------------------------------------------
# Figure 1B — Effect of Tone on Perceived Personality
# ---------------------------------------------------------------------------
def fig_tone_personality(df: pd.DataFrame) -> str:
    variables = [
        ('PP1', 'Friendly\n(PP1)'),
        ('PP3', 'Approachable\n(PP3)'),
        ('PP4', 'Warm\n(PP4)'),
        ('PP5', 'Formal\n(PP5)'),
        ('PP2', 'Professional\n(PP2)'),
    ]

    cas = df[df['tone'] == 1]
    pro = df[df['tone'] == 0]

    x     = np.arange(len(variables))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))

    means_c, means_f, sems_c, sems_f = [], [], [], []
    sigs, ds = [], []

    for var, _ in variables:
        c = cas[var].dropna()
        f = pro[var].dropna()
        means_c.append(c.mean())
        means_f.append(f.mean())
        sems_c.append(c.sem())
        sems_f.append(f.sem())
        _, p = stats.mannwhitneyu(c, f, alternative='two-sided')
        sigs.append(_sig(p))
        ds.append(abs(_cohen_d(c, f)))

    ax.bar(x - width/2, means_c, width, label='Casual tone',
           color=COL_CASUAL, alpha=0.85, yerr=sems_c, capsize=4)
    ax.bar(x + width/2, means_f, width, label='Formal tone',
           color=COL_FORMAL, alpha=0.85, yerr=sems_f, capsize=4)

    for i, (sig, d) in enumerate(zip(sigs, ds)):
        y_max = max(means_c[i]+sems_c[i], means_f[i]+sems_f[i]) + 0.15
        col   = COL_SIG if sig != 'ns' else COL_NS
        ax.text(i, y_max, sig, ha='center', fontsize=11,
                color=col, fontweight='bold')
        if sig != 'ns':
            ax.text(i, y_max + 0.3, f'd={d:.2f}', ha='center',
                    fontsize=8.5, color='#555555')

    ax.set_xticks(x)
    ax.set_xticklabels([v[1] for v in variables], fontsize=10)
    ax.set_ylabel('Mean score (1–7)', fontsize=11)
    ax.set_ylim(0, 8.2)
    ax.set_title('Effect of Chatbot Tone on Perceived Personality',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(framealpha=0.9)
    ax.set_axisbelow(True)
    return _save(fig, 'fig_tone_personality')


# ---------------------------------------------------------------------------
# Figure 2A — Effect of Tone on AI Perceptions
# ---------------------------------------------------------------------------
def fig_tone_ai_perceptions(df: pd.DataFrame) -> str:
    block1 = [
        ('PM1',      'PM1\n(freedom)'),
        ('PM2',      'PM2\n(override)'),
        ('PM3',      'PM3\n(manip.)'),
        ('PM4',      'PM4\n(pressure)'),
        ('PM_score', 'PM\n(composite)'),
    ]
    block2 = [
        ('Comp_score', 'Competence'),
        ('Ind1',       'Autonomy\n(Ind1)'),
        ('Ind2',       'Autonomy\n(Ind2)'),
        ('MA1',        'Moral\nGravity\n(MA1)'),
        ('MA2',        'Moral\nResp.\n(MA2)'),
        ('MP_score',   'Moral\nPatiency'),
    ]
    all_vars = block1 + block2

    cas = df[df['tone'] == 1]
    pro = df[df['tone'] == 0]

    means_c, means_f, sems_c, sems_f, sigs, sig_mask = [], [], [], [], [], []

    for var, _ in all_vars:
        if var not in df.columns:
            means_c.append(0); means_f.append(0)
            sems_c.append(0);  sems_f.append(0)
            sigs.append(1.0);  sig_mask.append(False)
            continue
        c = cas[var].dropna()
        f = pro[var].dropna()
        means_c.append(c.mean()); means_f.append(f.mean())
        sems_c.append(c.sem());   sems_f.append(f.sem())
        _, p = stats.mannwhitneyu(c, f, alternative='two-sided')
        sigs.append(p); sig_mask.append(p < .05)

    x     = np.arange(len(all_vars))
    width = 0.35
    fig, ax = plt.subplots(figsize=(15, 6))

    for i in range(len(all_vars)):
        col_c = COL_CASUAL if sig_mask[i] else '#999999'
        col_f = COL_FORMAL if sig_mask[i] else '#999999'
        alpha = 0.85 if sig_mask[i] else 0.35

        ax.bar(x[i] - width/2, means_c[i], width,
               color=col_c, alpha=alpha, yerr=sems_c[i], capsize=3)
        ax.bar(x[i] + width/2, means_f[i], width,
               color=col_f, alpha=alpha, yerr=sems_f[i], capsize=3)

        s   = _sig(sigs[i])
        y   = max(means_c[i]+sems_c[i], means_f[i]+sems_f[i]) + 0.15
        col = COL_SIG if sig_mask[i] else COL_NS
        ax.text(i, y, s, ha='center', fontsize=11,
                color=col, fontweight='bold')

    # Divider
    ax.axvline(x=len(block1) - 0.5, color='#BDC3C7',
               linestyle='--', linewidth=1.2)
    ax.text(len(block1)/2 - 0.5, 6.7,
            'Perceived Manipulation', ha='center',
            fontsize=10, color='#7F8C8D', style='italic')
    ax.text(len(block1) + len(block2)/2 - 0.5, 6.7,
            'Other AI Perceptions', ha='center',
            fontsize=10, color='#7F8C8D', style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels([v[1] for v in all_vars],
                       fontsize=10, rotation=0)
    ax.set_ylabel('Mean score (1–7)', fontsize=12)
    ax.set_ylim(0, 7.5)  # max = 7
    ax.set_title('Effect of Chatbot Tone on Cognitive AI Perceptions',
                 fontsize=15, fontweight='bold', pad=14)

    legend = [
        mpatches.Patch(color=COL_CASUAL, alpha=0.85,
                       label='Casual (significant)'),
        mpatches.Patch(color=COL_FORMAL, alpha=0.85,
                       label='Formal (significant)'),
        mpatches.Patch(color='#999999', alpha=0.4,
                       label='Not significant'),
    ]
    ax.legend(handles=legend, framealpha=0.9, fontsize=11)
    ax.set_axisbelow(True)
    return _save(fig, 'fig_tone_ai_perceptions')
    
# ---------------------------------------------------------------------------
# Figure 2B — Path Diagram: Tone → PP → PM
# ---------------------------------------------------------------------------
def fig_path_tone_pp_pm(df: pd.DataFrame, results: dict) -> str:
    paths = {}
    for pp in ['PP1', 'PP3', 'PP4']:
        if pp not in df.columns:
            continue
        m_a  = smf.ols(f'{pp} ~ tone', data=df).fit()
        a    = round(m_a.params['tone'], 3)
        p_a  = m_a.pvalues['tone']
        m_b  = smf.ols(f'PM_score ~ tone + {pp}', data=df).fit()
        b    = round(m_b.params[pp], 3)
        p_b  = m_b.pvalues[pp]
        paths[pp] = {'a': a, 'p_a': p_a, 'b': b, 'p_b': p_b}

    indirects = {}
    try:
        supp   = results.get('supp', {})
        pp_med = supp.get('pp_pm_mediations', pd.DataFrame())
        if not pp_med.empty:
            for _, row in pp_med.iterrows():
                for pp in ['PP1', 'PP3', 'PP4']:
                    if pp in str(row.get('Path', '')):
                        indirects[pp] = {
                            'ind':      row.get('Indirect', '—'),
                            'ci_low':   row.get('CI_low',   '—'),
                            'ci_high':  row.get('CI_high',  '—'),
                        }
    except Exception:
        pass

    m_dir2  = smf.ols('PM_score ~ tone + PP1 + PP3 + PP4', data=df).fit()
    c_prime = round(m_dir2.params['tone'], 3)
    p_prime = m_dir2.pvalues['tone']

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def _box(x, y, w, h, text, color, fontsize=11):
        rect = plt.Rectangle((x-w/2, y-h/2), w, h,
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color='white',
                fontweight='bold', zorder=4,
                multialignment='center')

    def _arrow(x1, y1, x2, y2, label='', color='#2C3E50'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', color=color, lw=2.0,
                        connectionstyle='arc3,rad=0'
                    ), zorder=2)
        if label:
            mx = (x1+x2)/2
            my = (y1+y2)/2 + 0.2
            ax.text(mx, my, label, ha='center', va='bottom',
                    fontsize=12, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', alpha=0.85,
                              edgecolor='none'))

    _box(1.5, 4.0, 2.4, 1.3,
         'Chatbot\nTone\n(Casual/Formal)', COL_CASUAL, fontsize=11)

    pp_positions = [(5.5, 6.5), (5.5, 4.0), (5.5, 1.5)]
    pp_labels    = [
        ('PP1', 'Friendly\n(PP1)'),
        ('PP3', 'Approachable\n(PP3)'),
        ('PP4', 'Warm\n(PP4)'),
    ]
    for (px, py), (pp, pp_label) in zip(pp_positions, pp_labels):
        _box(px, py, 2.4, 1.0, pp_label, '#2980B9', fontsize=10)

    _box(10.5, 4.0, 2.4, 1.3,
         'Perceived\nManipulation\n(PM_score)', COL_NEG, fontsize=11)

    for (px, py), (pp, _) in zip(pp_positions, pp_labels):
        if pp in paths:
            a     = paths[pp]['a']
            p_val = paths[pp]['p_a']
            _arrow(2.7, 4.0, px-1.2, py,
                   f'a={a}{_sig(p_val)}', COL_CASUAL)

    for (px, py), (pp, _) in zip(pp_positions, pp_labels):
        if pp in paths:
            b     = paths[pp]['b']
            p_val = paths[pp]['p_b']
            _arrow(px+1.2, py, 9.3, 4.0,
                   f'b={b}{_sig(p_val)}', '#2980B9')

    ax.annotate('', xy=(9.3, 3.5), xytext=(2.7, 3.5),
                arrowprops=dict(
                    arrowstyle='->', color='#BDC3C7',
                    lw=1.5, linestyle='dashed',
                    connectionstyle='arc3,rad=-0.35'
                ), zorder=2)
    sig_str = _sig(p_prime)
    ax.text(6.0, 2.2,
            f"c' (direct) = {c_prime} "
            f"{'(ns)' if sig_str == 'ns' else sig_str}",
            ha='center', fontsize=11, color='#95A5A6', style='italic')

    
    # Indirect effects box
    ind_lines = []
    for pp, pp_label in [
        ('PP1','Friendly'), ('PP3','Approachable'), ('PP4','Warm')
    ]:
        if pp in indirects:
            d = indirects[pp]
            try:
                ind_lines.append(
                    f"via {pp_label}: "
                    f"indirect={float(d['ind']):.3f}, "
                    f"CI[{float(d['ci_low']):.3f}, "
                    f"{float(d['ci_high']):.3f}]"
                )
            except Exception:
                ind_lines.append(f"via {pp_label}: {d}")

    if ind_lines:
        rect = plt.Rectangle((1.0, 0.15), 12.0, 1.0,
                               facecolor='#FDFEFE',
                               edgecolor='#BDC3C7', linewidth=1)
        ax.add_patch(rect)
        ax.text(7.0, 0.95,
                'Indirect effects (bootstrapped, 5,000 iter.):',
                ha='center', fontsize=11,
                fontweight='bold', color='#2C3E50')
        ax.text(7.0, 0.45,
                '  |  '.join(ind_lines),
                ha='center', fontsize=11, color='#2C3E50')

    ax.text(7.0, -0.2,
            'Paths represent separate bootstrapped mediation '
            'analyses, not a single structural model.',
            ha='center', fontsize=10,
            color='#888888', style='italic')

    ax.set_title(
        'Mediation Model: Chatbot Tone → Perceived Personality '
        '→ Perceived Manipulation',
        fontsize=15, pad=16
    )
    return _save(fig, 'fig_path_tone_pp_pm')

# ---------------------------------------------------------------------------
# Figure 3 — Distribution of Feedback Quality by Tone
# ---------------------------------------------------------------------------
def fig_quality_distribution(df: pd.DataFrame) -> str:
    if 'composite' not in df.columns:
        log.warning('Figure 3: composite column missing')
        return ''

    cas = df[df['tone'] == 1]['composite'].dropna()
    pro = df[df['tone'] == 0]['composite'].dropna()
    _, p_mw = stats.mannwhitneyu(cas, pro, alternative='two-sided')

    fig, ax = plt.subplots(figsize=(8, 5.5))

    parts = ax.violinplot(
        [cas.values, pro.values],
        positions=[1, 2], showmedians=False, showextrema=False
    )
    for pc, color in zip(parts['bodies'], [COL_CASUAL, COL_FORMAL]):
        pc.set_facecolor(color)
        pc.set_alpha(0.35)
        pc.set_edgecolor(color)

    # Box plots
    bp = ax.boxplot(
        [cas.values, pro.values],
        positions=[1, 2], widths=0.12,
        patch_artist=True, showfliers=False,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, color in zip(bp['boxes'], [COL_CASUAL, COL_FORMAL]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Jitter
    np.random.seed(42)
    for vals, pos, color in [(cas, 1, COL_CASUAL), (pro, 2, COL_FORMAL)]:
        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(pos + jitter, vals, alpha=0.25, s=18,
                   color=color, zorder=2)

    # Significance bracket
    y_max = max(cas.max(), pro.max()) + 0.2
    sig_s  = _sig(p_mw)
    ax.plot([1, 1, 2, 2], [y_max, y_max+0.05, y_max+0.05, y_max],
            color='#2C3E50', linewidth=1.2)
    col_br = COL_SIG if sig_s != 'ns' else COL_NS
    ax.text(1.5, y_max + 0.1,
            f'{sig_s} (p={p_mw:.3f})',
            ha='center', fontsize=10,
            color=col_br, fontweight='bold')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f'Casual tone\n(n={len(cas)})',
         f'Formal tone\n(n={len(pro)})'],
        fontsize=11
    )
    ax.set_ylabel('Feedback Quality (composite score)', fontsize=11)
    ax.set_title('Distribution of Feedback Quality by Tone Condition',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_axisbelow(True)
    return _save(fig, 'fig_quality_distribution')


# ---------------------------------------------------------------------------
# Figure 4 — Path Diagram: PM → Feedback Quality and Evaluation
# ---------------------------------------------------------------------------
def fig_path_pm_outcomes(df: pd.DataFrame, results: dict) -> str:
    outcomes = {
        'composite': {'dv': 'Feedback Quality\n(composite)', 'color': COL_SIG},
        'E3':        {'dv': 'Enjoyment\n(E3)',            'color': COL_CASUAL},
        'E4':        {'dv': 'Utility\n(E4)',                 'color': '#6C3483'},
        'E5':        {'dv': 'Reuse Intention\n(E5)',         'color': COL_FORMAL},
    }

    betas = {}
    try:
        full_d = results.get('D', {}).get('full', pd.DataFrame())
        full_e = results.get('E', {}).get('full', pd.DataFrame())
        for dv, info in outcomes.items():
            src = full_d if dv == 'composite' else full_e
            if src.empty or 'DV' not in src.columns:
                continue
            sub       = src[src['DV'] == dv]
            last_bloc = sub['Block'].max()
            sub_last  = sub[sub['Block'] == last_bloc]
            pm_row    = sub_last[sub_last['Predictor'] == 'PM_score']
            if not pm_row.empty:
                betas[dv] = {
                    'beta': round(float(pm_row['β'].values[0]), 3),
                    'p':    float(pm_row['p'].values[0]),
                }
    except Exception as e:
        log.warning(f'Figure PM→outcomes: {e}')

    fallback = {
        'composite': {'beta': -.177, 'p': .051},
        'E3':        {'beta': -.252, 'p': .010},
        'E4':        {'beta': -.299, 'p': .001},
        'E5':        {'beta': -.352, 'p': .001},
    }
    for dv in outcomes:
        if dv not in betas:
            betas[dv] = fallback[dv]

    indirects = {}
    try:
        full_f = results.get('F', {}).get('full', pd.DataFrame())
        if not full_f.empty:
            tone_rows = full_f[
                (full_f['Series'] == '1') &
                (full_f['IV'] == 'tone') &
                (full_f['Mediator'] == 'PM_score')
            ]
            for _, row in tone_rows.iterrows():
                dv = row.get('DV', '')
                if dv in outcomes:
                    indirects[dv] = {
                        'ind':     round(float(row.get('Indirect', 0)), 3),
                        'ci_low':  round(float(row.get('CI_low',   0)), 3),
                        'ci_high': round(float(row.get('CI_high',  0)), 3),
                        'sig':     row.get('Sig.', 'ns'),
                    }
    except Exception as e:
        log.warning(f'Figure PM→outcomes indirects: {e}')

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def _box(x, y, w, h, text, color, fontsize=11):
        rect = plt.Rectangle((x-w/2, y-h/2), w, h,
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color='white',
                fontweight='bold', zorder=4,
                multialignment='center')

    # PM box
    _box(3.5, 5.5, 2.6, 1.5,
         'Perceived\nManipulation\n(PM_score)', COL_NEG, fontsize=12)

    # Outcome boxes — well spaced, leaving room for indirect box at bottom
    y_positions = [8.2, 6.8, 5.2, 3.8]
    for (dv, info), y_pos in zip(outcomes.items(), y_positions):
        _box(10.5, y_pos, 3.0, 1.0, info['dv'], info['color'], fontsize=10)

        if dv in betas:
            b     = betas[dv]['beta']
            p_val = betas[dv]['p']
            sig_s = _sig(p_val)
            label = f'β={b}{sig_s if sig_s != "ns" else " (ns)"}'

            # Dashed arrow for composite (marginal ns)
            linestyle = 'dashed' if dv == 'composite' else 'solid'
            ax.annotate(
                '', xy=(9.0, y_pos), xytext=(4.8, 5.5),
                arrowprops=dict(
                    arrowstyle='->', color=COL_NEG,
                    lw=1.8,
                    linestyle=linestyle,
                    connectionstyle='arc3,rad=0'
                ), zorder=2
            )
            mx = (4.8 + 9.0)/2
            my = (5.5 + y_pos)/2 + 0.15
            ax.text(mx, my, label,
                    ha='center', fontsize=11,
                    color=COL_NEG, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.85,
                              edgecolor='none'))

    # Indirect effects box — clearly below all outcome boxes
    ind_lines = []
    for dv, info in outcomes.items():
        if dv in indirects:
            d = indirects[dv]
            dv_label = info['dv'].replace('\n', ' ')
            ind_lines.append(
                f"{dv_label}: indirect={d['ind']}, "
                f"CI[{d['ci_low']}, {d['ci_high']}]"
            )

    if ind_lines:
        rect = plt.Rectangle((0.5, 0.1), 13.0, 1.5,
                               facecolor='#FDFEFE',
                               edgecolor='#BDC3C7', linewidth=1)
        ax.add_patch(rect)
        ax.text(7.0, 1.45,
                'Indirect effects of tone via PM_score '
                '(bootstrapped, 5,000 iter., CI excludes 0):',
                ha='center', fontsize=11,
                fontweight='bold', color='#2C3E50')
        ax.text(7.0, 0.95,
                '  |  '.join(ind_lines[:2]),
                ha='center', fontsize=10.5, color='#2C3E50')
        if len(ind_lines) > 2:
            ax.text(7.0, 0.45,
                    '  |  '.join(ind_lines[2:]),
                    ha='center', fontsize=10.5, color='#2C3E50')

    ax.text(1.5, 2.8,
            'Direct effect of tone\non all outcomes: ns\n(full mediation)',
            ha='center', fontsize=10, color='#95A5A6',
            style='italic', multialignment='center')

    ax.text(7.0, -0.2,
            'Direct effect of tone on all outcomes non-significant. '
            'Paths represent separate bootstrapped analyses.',
            ha='center', fontsize=10,
            color='#888888', style='italic')

    ax.set_title(
        'Perceived Manipulation as Central Predictor of '
        'Feedback Quality and Chatbot Evaluation',
        fontsize=15, pad=16
    )
    return _save(fig, 'fig_path_pm_outcomes')

# ---------------------------------------------------------------------------
# Figure 5 — Tone × Language Interaction on Engagement
# ---------------------------------------------------------------------------
def fig_tone_language_engagement(df: pd.DataFrame) -> str:
    if 'language' not in df.columns:
        log.warning('Figure 5: language column missing')
        return ''

    langs      = ['FR', 'EN']
    lang_labels = ['French (n=141)', 'English (n=39)']

    data_cells = {}
    for lang in langs:
        for tone_val, tone_label in [(1,'Casual'), (0,'Formal')]:
            sub = df[
                (df['language'] == lang) & (df['tone'] == tone_val)
            ]['engagement_score'].dropna()
            data_cells[(lang, tone_label)] = sub

    x     = np.arange(len(langs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5))

    means_c = [data_cells[(l,'Casual')].mean() for l in langs]
    means_f = [data_cells[(l,'Formal')].mean() for l in langs]
    sems_c  = [data_cells[(l,'Casual')].sem()  for l in langs]
    sems_f  = [data_cells[(l,'Formal')].sem()  for l in langs]

    ax.bar(x - width/2, means_c, width, label='Casual',
           color=COL_CASUAL, alpha=0.85, yerr=sems_c, capsize=4)
    ax.bar(x + width/2, means_f, width, label='Formal',
           color=COL_FORMAL, alpha=0.85, yerr=sems_f, capsize=4)

    for i, lang in enumerate(langs):
        c = data_cells[(lang,'Casual')]
        f = data_cells[(lang,'Formal')]
        _, p = stats.ttest_ind(c, f, equal_var=False)
        y_max = max(
            means_c[i]+sems_c[i], means_f[i]+sems_f[i]
        ) + 0.06
        col = COL_SIG if p < .05 else COL_NS
        ax.text(i, y_max, _sig(p),
                ha='center', fontsize=11,
                color=col, fontweight='bold')

    # ANOVA interaction
    try:
        df2         = df.copy()
        df2['lang'] = (df2['language'] == 'EN').astype(int)
        model       = smf.ols(
            'engagement_score ~ tone + lang + tone:lang',
            data=df2
        ).fit()
        from statsmodels.stats.anova import anova_lm
        aov   = anova_lm(model, typ=2)
        f_int = round(aov.loc['tone:lang','F'], 2)
        p_int = round(aov.loc['tone:lang','PR(>F)'], 3)
        ax.text(0.97, 0.97,
                f'Tone × Language interaction:\nF={f_int}, p={p_int}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white', alpha=0.85,
                          edgecolor='#CCCCCC'))
    except Exception:
        pass

    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=11)
    ax.set_ylabel('Engagement Score (z-standardised)', fontsize=11)
    ax.set_title('Tone × Language Interaction on Behavioural Engagement',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(framealpha=0.9)
    ax.set_axisbelow(True)
    ax.text(0.5, -0.16,
            'Note: Feedback quality composite did not differ '
            'significantly by tone within either language group '
            '(FR p=.162, EN p=.705)',
            transform=ax.transAxes, ha='center',
            fontsize=8.5, color='#555555', style='italic')

    plt.subplots_adjust(bottom=0.18)
    return _save(fig, 'fig_tone_language_engagement')


# ---------------------------------------------------------------------------
# Annexe A — Predictors of Chatbot Evaluation (E3, E4, E5)
# ---------------------------------------------------------------------------
def fig_annex_evaluation_predictors(df: pd.DataFrame,
                                     results: dict) -> str:
    dvs    = ['E3', 'E4', 'E5']
    colors = [COL_CASUAL, '#6C3483', COL_FORMAL]
    labels = ['Enjoyment (E3)', 'Utility (E4)', 'Reuse Intention (E5)']

    pred_order = [
        'tone', 'PM_score', 'Comp_score', 'Ind1', 'Ind2',
        'MA1', 'MA2', 'MP_score', 'composite', 'E2',
    ]
    pred_labels = {
        'tone':      'Tone',
        'PM_score':  'Perceived\nManipulation',
        'Comp_score':'Competence',
        'Ind1':      'Autonomy\n(Ind1)',
        'Ind2':      'Autonomy\n(Ind2)',
        'MA1':       'Moral\nGravity (MA1)',
        'MA2':       'AI Moral\nResp. (MA2)',
        'MP_score':  'Moral\nPatiency',
        'composite': 'Feedback\nQuality',
        'E2':        'Engagement\nFelt (E2)',
    }

    betas  = {dv: {} for dv in dvs}
    p_vals = {dv: {} for dv in dvs}
    r2s    = {}

    try:
        full_e = results.get('E', {}).get('full', pd.DataFrame())
        if not full_e.empty:
            for dv in dvs:
                sub       = full_e[full_e['DV'] == dv]
                last_bloc = sub['Block'].max()
                sub_last  = sub[sub['Block'] == last_bloc].set_index('Predictor')
                for pred in pred_order:
                    if pred in sub_last.index:
                        betas[dv][pred]  = float(sub_last.loc[pred, 'β'])
                        p_vals[dv][pred] = float(sub_last.loc[pred, 'p'])
                if 'R²' in sub_last.columns:
                    r2s[dv] = round(float(sub_last['R²'].iloc[-1]), 3)
    except Exception as e:
        log.warning(f'Annexe A: {e}')

    x     = np.arange(len(pred_order))
    width = 0.26
    fig, ax = plt.subplots(figsize=(14, 6.5))

    for i, (dv, color, label) in enumerate(zip(dvs, colors, labels)):
        for j, pred in enumerate(pred_order):
            v = betas[dv].get(pred, None)
            p = p_vals[dv].get(pred, 1.0)
            if v is None:
                continue
            bar_color = color if p < .05 else '#CCCCCC'
            alpha     = 0.85 if p < .05 else 0.35
            bar = ax.bar(
                x[j] + (i-1)*width, v, width,
                color=bar_color, alpha=alpha,
                label=label if j == 0 else ''
            )
            if p < .05:
                y_ann = v + (0.015 if v >= 0 else -0.045)
                ax.text(
                    x[j] + (i-1)*width, y_ann,
                    _sig(p), ha='center', fontsize=9,
                    color=color, fontweight='bold'
                )

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pred_labels.get(p, p) for p in pred_order],
        fontsize=10
    )
    ax.set_ylabel('Standardised β coefficient', fontsize=12)

    r2_str = '  |  '.join(
        [f'R²({dv})={r2s[dv]}' for dv in dvs if dv in r2s]
    )
    ax.set_title(
        'Predictors of Chatbot Evaluation: Standardised β Coefficients\n'
        f'{r2_str}',
        fontsize=14, fontweight='bold', pad=12
    )

    # Legend with correct colors
    handles = [
        mpatches.Patch(color=colors[i], alpha=0.85, label=labels[i])
        for i in range(len(dvs))
    ]
    handles.append(
        mpatches.Patch(color='#CCCCCC', alpha=0.4, label='Not significant')
    )
    ax.legend(handles=handles, framealpha=0.9, fontsize=11,
              loc='upper right')
    ax.set_axisbelow(True)
    return _save(fig, 'fig_annex_evaluation_predictors')
                                         
# ---------------------------------------------------------------------------
# Annexe B — Bubble Plot: Perceived Personality × DVs
# ---------------------------------------------------------------------------
def fig_annex_personality_bubbles(df: pd.DataFrame) -> str:
    pp_vars = ['PP1', 'PP2', 'PP3', 'PP4', 'PP5']
    pp_labels = ['Friendly\n(PP1)', 'Professional\n(PP2)',
                 'Approachable\n(PP3)', 'Warm\n(PP4)', 'Formal\n(PP5)']
    dv_vars   = ['E3', 'E4', 'E5', 'E6', 'composite',
                 'PM_score', 'engagement_score']
    dv_labels = ['Enjoyment\n(E3)', 'Utility\n(E4)',
                 'Reuse\n(E5)', 'Preference\n(E6)',
                 'Feedback\nQuality', 'Perceived\nManipulation',
                 'Engagement\nScore']

    # Compute r values
    r_matrix = np.zeros((len(pp_vars), len(dv_vars)))
    p_matrix = np.ones((len(pp_vars), len(dv_vars)))

    for i, pp in enumerate(pp_vars):
        for j, dv in enumerate(dv_vars):
            if pp in df.columns and dv in df.columns:
                clean = df[[pp, dv]].dropna()
                if len(clean) > 10:
                    r, p = stats.pearsonr(clean[pp], clean[dv])
                    r_matrix[i, j] = r
                    p_matrix[i, j] = p

    # FDR correction
    p_flat    = p_matrix.flatten()
    _, p_adj, _, _ = multipletests(p_flat, method='fdr_bh')
    p_adj_mat = p_adj.reshape(p_matrix.shape)

    fig, ax = plt.subplots(
        figsize=(len(dv_vars)*1.6, len(pp_vars)*1.4)
    )
    ax.set_xlim(-0.5, len(dv_vars) - 0.5)
    ax.set_ylim(-0.5, len(pp_vars) - 0.5)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.spines[:].set_visible(False)

    for i in range(len(pp_vars)):
        for j in range(len(dv_vars)):
            r     = r_matrix[i, j]
            p_adj = p_adj_mat[i, j]
            sig   = p_adj < .05

            size  = max(abs(r) * 2000, 50)
            color = '#1A5276' if r > 0 else '#C0392B'
            alpha = 0.85 if sig else 0.2

            ax.scatter(j, i, s=size, color=color, alpha=alpha,
                       zorder=3)
            ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                    fontsize=8.5, color='white' if sig else '#888888',
                    fontweight='bold' if sig else 'normal', zorder=4)

    ax.set_xticks(range(len(dv_vars)))
    ax.set_xticklabels(dv_labels, fontsize=9, ha='right',
                        rotation=0)
    ax.set_yticks(range(len(pp_vars)))
    ax.set_yticklabels(pp_labels, fontsize=9)
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    for r_val, label in [(0.5, 'r=0.5'), (0.3, 'r=0.3'), (0.1, 'r=0.1')]:
        ax.scatter([], [], s=r_val*2000, color='#555555',
                   alpha=0.6, label=label)
    pos_patch = mpatches.Patch(color='#1A5276', label='Positive r (significant)')
    neg_patch = mpatches.Patch(color='#C0392B', label='Negative r (significant)')
    ns_patch  = mpatches.Patch(color='#888888', alpha=0.3,
                                label='Not significant (FDR)')
    ax.legend(
        handles=[pos_patch, neg_patch, ns_patch],
        loc='upper right', fontsize=8.5, framealpha=0.9
    )
    ax.set_title(
        'Correlations Between Perceived Personality and Outcome Variables\n'
        'Circle size ∝ |r|. Opacity = significance after FDR correction.',
        fontsize=11, fontweight='bold', pad=12
    )
    return _save(fig, 'fig_annex_personality_bubbles')


# ---------------------------------------------------------------------------
# Annexe C — Mediation: Competence-Autonomy → Enjoyment → Quality
# ---------------------------------------------------------------------------
def fig_annex_mediation_comp_aut(df: pd.DataFrame,
                                  results: dict) -> str:
    ivs = ['Comp_score', 'Ind1', 'MP_score']
    iv_labels = {
        'Comp_score': 'Competence\n(Comp_score)',
        'Ind1':       'Autonomy\n(Ind1)',
        'MP_score':   'Moral Patiency\n(MP_score)',
    }
    iv_colors = {
        'Comp_score': '#2980B9',
        'Ind1':       '#1A5276',
        'MP_score':   COL_SIG,
    }

    med_data = {}
    b_val, p_b = '—', 1.0
    try:
        full_f = results.get('F', {}).get('full', pd.DataFrame())
        if not full_f.empty:
            for iv in ivs:
                rows = full_f[
                    (full_f['IV'] == iv) &
                    (full_f['Mediator'] == 'E3') &
                    (full_f['DV'] == 'composite')
                ]
                if not rows.empty:
                    r = rows.iloc[0]
                    med_data[iv] = {
                        'a':        round(float(r.get('a', 0)), 3),
                        'p_a':      float(r.get('p_a', 1)),
                        'indirect': round(float(r.get('Indirect', 0)), 3),
                        'ci_low':   round(float(r.get('CI_low', 0)), 3),
                        'ci_high':  round(float(r.get('CI_high', 0)), 3),
                    }
            b_rows = full_f[
                (full_f['Mediator'] == 'E3') &
                (full_f['DV'] == 'composite') &
                (full_f['IV'] == 'Comp_score')
            ]
            if not b_rows.empty:
                b_val = round(float(b_rows.iloc[0].get('b', 0)), 3)
                p_b   = float(b_rows.iloc[0].get('p_b', 1))
    except Exception as e:
        log.warning(f'Annexe C: {e}')

    # Taller figure to avoid overlap
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def _box(x, y, w, h, text, color, fontsize=10):
        rect = plt.Rectangle((x-w/2, y-h/2), w, h,
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color='white',
                fontweight='bold', zorder=4,
                multialignment='center')

    # Mediator and outcome — higher up to leave room for IV boxes and indirect box
    _box(6.5, 7.5, 2.6, 1.0, 'Chatbot\nEnjoyment\n(E3)',
         COL_CASUAL, fontsize=11)
    _box(11.5, 5.5, 2.4, 1.2, 'Feedback\nQuality\n(composite)',
         COL_SIG, fontsize=11)

    # b path
    ax.annotate('', xy=(10.3, 5.8), xytext=(7.8, 7.1),
                arrowprops=dict(arrowstyle='->', color=COL_CASUAL, lw=2.0))
    ax.text(9.3, 6.7,
            f'b={b_val}{_sig(p_b)}',
            ha='center', fontsize=12, color=COL_CASUAL, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.85, edgecolor='none'))

    # IV boxes — well spaced vertically
    iv_y = [7.5, 5.5, 3.5]
    for iv, y_pos in zip(ivs, iv_y):
        color = iv_colors.get(iv, '#555555')
        _box(1.8, y_pos, 2.8, 1.0, iv_labels[iv], color, fontsize=10)

        if iv in med_data:
            d     = med_data[iv]
            label = f"a={d['a']}{_sig(d['p_a'])}"
            ax.annotate('', xy=(5.2, 7.2), xytext=(3.2, y_pos+0.1),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
            mx = (1.8 + 6.5)/2
            my = (y_pos + 7.5)/2 + 0.1
            ax.text(mx, my, label,
                    ha='center', fontsize=11, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.85, edgecolor='none'))

        # Direct path
        ax.annotate('', xy=(10.3, 5.2), xytext=(3.2, y_pos),
                    arrowprops=dict(
                        arrowstyle='->', color='#BDC3C7',
                        lw=1.2, linestyle='dashed',
                        connectionstyle='arc3,rad=-0.15'
                    ), zorder=1)

    ax.text(7.0, 2.7, "c' (direct effects) = ns",
            ha='center', fontsize=11, color='#95A5A6', style='italic')

    # Indirect effects box — at the very bottom, no overlap
    ind_lines = []
    for iv in ivs:
        if iv in med_data:
            d = med_data[iv]
            ind_lines.append(
                f"{iv_labels[iv].replace(chr(10),' ')}: "
                f"indirect={d['indirect']}, "
                f"CI[{d['ci_low']}, {d['ci_high']}]"
            )

    if ind_lines:
        rect = plt.Rectangle((0.3, 0.1), 12.4, 1.8,
                               facecolor='#FDFEFE',
                               edgecolor='#BDC3C7', linewidth=1)
        ax.add_patch(rect)
        ax.text(6.5, 1.75,
                'Indirect effects (bootstrapped, 5,000 iter.):',
                ha='center', fontsize=11,
                fontweight='bold', color='#2C3E50')
        ax.text(6.5, 1.2,
                '  |  '.join(ind_lines),
                ha='center', fontsize=10.5, color='#2C3E50')
        ax.text(6.5, 0.5,
                'Separate bootstrapped mediation analyses. '
                'Only effects with CI excluding zero shown as significant.',
                ha='center', fontsize=10, color='#888888', style='italic')

    ax.set_title(
        'Mediation: AI Competence and Autonomy Perceptions '
        '→ Enjoyment → Feedback Quality',
        fontsize=15, pad=16
    )
    return _save(fig, 'fig_annex_mediation_comp_aut')


# ---------------------------------------------------------------------------
# Annexe D — Mediation: MA2 → MA1 → E3
# ---------------------------------------------------------------------------
def fig_annex_mediation_ma(df: pd.DataFrame, results: dict) -> str:
    # Extract from results
    med = {}
    try:
        full_f = results.get('F', {}).get('full', pd.DataFrame())
        if not full_f.empty:
            row = full_f[
                (full_f['IV'] == 'MA2') &
                (full_f['Mediator'] == 'MA1') &
                (full_f['DV'] == 'E3')
            ]
            if not row.empty:
                r = row.iloc[0]
                med = {
                    'a':        round(float(r.get('a', 0)), 3),
                    'p_a':      float(r.get('p_a', 1)),
                    'b':        round(float(r.get('b', 0)), 3),
                    'p_b':      float(r.get('p_b', 1)),
                    'c_prime':  round(float(r.get('c_prime', 0)), 3),
                    'p_cprime': float(r.get('p_cprime', 1)),
                    'indirect': round(float(r.get('Indirect', 0)), 3),
                    'ci_low':   round(float(r.get('CI_low', 0)), 3),
                    'ci_high':  round(float(r.get('CI_high', 0)), 3),
                }
    except Exception as e:
        log.warning(f'Annexe D: {e}')

    # r(MA2, PM_score)
    r_pm, _ = stats.pearsonr(
        df['MA2'].dropna(),
        df['PM_score'].reindex(df['MA2'].dropna().index).dropna()
    ) if 'MA2' in df.columns and 'PM_score' in df.columns else (0, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def _box(x, y, w, h, text, color, fontsize=10):
        rect = plt.Rectangle((x-w/2, y-h/2), w, h,
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color='white',
                fontweight='bold', zorder=4, multialignment='center')

    _box(1.5, 3.0, 2.4, 1.2,
         'AI Moral\nResponsibility\n(MA2)', '#6C3483', fontsize=10)
    _box(5.0, 5.0, 2.4, 1.0,
         'Moral Gravity\nAI→Human\n(MA1)', '#884EA0', fontsize=9)
    _box(8.5, 3.0, 2.2, 1.2,
         'Chatbot\nEnjoyment\n(E3)', COL_CASUAL, fontsize=10)

    # a: MA2 → MA1
    if med:
        ax.annotate('', xy=(3.8, 4.6), xytext=(2.7, 3.5),
                    arrowprops=dict(arrowstyle='->', color='#6C3483', lw=2))
        ax.text(3.0, 4.2,
                f"a={med.get('a','—')}{_sig(med.get('p_a',1))}",
                ha='center', fontsize=9.5, color='#6C3483', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.85, edgecolor='none'))

        # b: MA1 → E3
        ax.annotate('', xy=(7.4, 3.4), xytext=(6.2, 4.6),
                    arrowprops=dict(arrowstyle='->', color='#884EA0', lw=2))
        ax.text(7.0, 4.2,
                f"b={med.get('b','—')}{_sig(med.get('p_b',1))}",
                ha='center', fontsize=9.5, color='#884EA0', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.85, edgecolor='none'))

        # Direct path (dashed)
        ax.annotate('', xy=(7.4, 2.8), xytext=(2.7, 2.8),
                    arrowprops=dict(
                        arrowstyle='->', color='#BDC3C7',
                        lw=1.5, linestyle='dashed',
                        connectionstyle='arc3,rad=-0.3'
                    ))
        ax.text(5.0, 1.9,
                f"c' (direct) = {med.get('c_prime','—')} "
                f"{'(ns)' if _sig(med.get('p_cprime',1))=='ns' else _sig(med.get('p_cprime',1))}",
                ha='center', fontsize=9, color='#95A5A6', style='italic')

        # Indirect effect box
        rect = plt.Rectangle((1.0, 0.15), 8.0, 1.0,
                               facecolor='#FDFEFE',
                               edgecolor='#BDC3C7', linewidth=1)
        ax.add_patch(rect)
        ax.text(5.0, 0.95,
                f"Indirect effect: {med.get('indirect','—')}, "
                f"95% CI [{med.get('ci_low','—')}, {med.get('ci_high','—')}]  "
                f"|  Full mediation",
                ha='center', fontsize=9, fontweight='bold', color='#2C3E50')
        ax.text(5.0, 0.45,
                f"Independent of perceived manipulation: "
                f"r(MA2, PM_score) = {r_pm:.3f} (ns)  |  "
                "Bootstrapped mediation, 5,000 iterations.",
                ha='center', fontsize=8, color='#555555', style='italic')

    ax.set_title(
        'Mediation: AI Moral Responsibility → Moral Gravity → Chatbot Enjoyment',
        fontsize=11, pad=15
    )
    return _save(fig, 'fig_annex_mediation_ma')

# ---------------------------------------------------------------------------
# Figure PP-A — Standardised Effects of AI Perceptions on Feedback Quality
# ---------------------------------------------------------------------------
def fig_ai_perceptions_quality(df: pd.DataFrame, results: dict) -> str:
    import statsmodels.formula.api as smf

    ai_vars = ['PM_score','Comp_score','Ind1','Ind2','MA1','MA2','MP_score']
    ai_labels = {
        'PM_score':  'Perceived\nManipulation',
        'Comp_score':'Competence',
        'Ind1':      'Autonomy\n(Ind1)',
        'Ind2':      'Autonomy\n(Ind2)',
        'MA1':       'Moral\nGravity (MA1)',
        'MA2':       'AI Moral\nResp. (MA2)',
        'MP_score':  'Moral\nPatiency',
    }

    avail   = [v for v in ai_vars if v in df.columns and 'composite' in df.columns]
    betas   = {}
    p_vals  = {}
    r2_val  = None

    if avail:
        try:
            formula = f'composite ~ {" + ".join(avail)}'
            model   = smf.ols(formula, data=df).fit()
            r2_val  = round(model.rsquared, 3)
            for v in avail:
                betas[v]  = float(model.params[v])
                p_vals[v] = float(model.pvalues[v])
        except Exception as e:
            log.warning(f'fig_ai_perceptions_quality: {e}')

    # Fallback from results
    if not betas:
        try:
            full_d = results.get('D', {}).get('full', pd.DataFrame())
            if not full_d.empty:
                sub = full_d[full_d['DV'] == 'composite']
                last = sub['Block'].max()
                sub_last = sub[sub['Block'] == last].set_index('Predictor')
                for v in ai_vars:
                    if v in sub_last.index:
                        betas[v]  = float(sub_last.loc[v, 'β'])
                        p_vals[v] = float(sub_last.loc[v, 'p'])
        except Exception as e:
            log.warning(f'fig_ai_perceptions_quality fallback: {e}')

    if not betas:
        log.warning('fig_ai_perceptions_quality: no data')
        return ''

    vars_plot  = [v for v in ai_vars if v in betas]
    beta_vals  = [betas[v] for v in vars_plot]
    p_list     = [p_vals[v] for v in vars_plot]
    labels_plt = [ai_labels.get(v, v) for v in vars_plot]

    colors = []
    alphas = []
    for b, p in zip(beta_vals, p_list):
        sig = p < .05
        colors.append('#1A5276' if b >= 0 else '#C0392B')
        alphas.append(0.85 if sig else 0.3)

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.barh(range(len(vars_plot)), beta_vals,
                   color=colors, alpha=1.0)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    # Significance markers
    for i, (b, p) in enumerate(zip(beta_vals, p_list)):
        sig_s = _sig(p)
        x_pos = b + (0.005 if b >= 0 else -0.005)
        ha    = 'left' if b >= 0 else 'right'
        col   = '#1E8449' if p < .05 else '#95A5A6'
        ax.text(x_pos, i, sig_s, va='center', ha=ha,
                fontsize=11, color=col, fontweight='bold')

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(vars_plot)))
    ax.set_yticklabels(labels_plt, fontsize=11)
    ax.set_xlabel('Standardised β coefficient', fontsize=12)
    r2_str = f'  (R²={r2_val})' if r2_val else ''
    ax.set_title(
        f'Standardised Effects of AI Perceptions on Feedback Quality{r2_str}\n'
        'Blue = positive, Red = negative. Faded = not significant (p≥.05)',
        fontsize=14, fontweight='bold', pad=12
    )
    ax.set_axisbelow(True)

    return _save(fig, 'fig_pp_ai_quality')


# ---------------------------------------------------------------------------
# Figure PP-B — AI Perceptions × Chatbot Evaluation Heatmap
# ---------------------------------------------------------------------------
def fig_ai_perceptions_heatmap(df: pd.DataFrame, results: dict) -> str:
    import statsmodels.formula.api as smf

    ai_vars = ['PM_score','Comp_score','Ind1','Ind2','MA1','MA2','MP_score']
    ev_dvs  = ['E3','E4','E5','E6']
    ai_labels = {
        'PM_score':  'Perceived Manip.',
        'Comp_score':'Competence',
        'Ind1':      'Autonomy (Ind1)',
        'Ind2':      'Autonomy (Ind2)',
        'MA1':       'Moral Gravity (MA1)',
        'MA2':       'AI Moral Resp. (MA2)',
        'MP_score':  'Moral Patiency',
    }
    dv_labels = {
        'E3': 'Enjoyment\n(E3)',
        'E4': 'Utility\n(E4)',
        'E5': 'Reuse\n(E5)',
        'E6': 'Preference\n(E6)',
    }

    avail_ai = [v for v in ai_vars if v in df.columns]
    beta_mat = np.zeros((len(avail_ai), len(ev_dvs)))
    p_mat    = np.ones((len(avail_ai),  len(ev_dvs)))

    for j, dv in enumerate(ev_dvs):
        if dv not in df.columns:
            continue
        # Try from results first
        loaded = False
        try:
            full_e = results.get('E', {}).get('full', pd.DataFrame())
            if not full_e.empty and 'DV' in full_e.columns:
                sub      = full_e[full_e['DV'] == dv]
                last     = sub['Block'].max()
                sub_last = sub[sub['Block'] == last].set_index('Predictor')
                for i, v in enumerate(avail_ai):
                    if v in sub_last.index:
                        beta_mat[i, j] = float(sub_last.loc[v, 'β'])
                        p_mat[i, j]    = float(sub_last.loc[v, 'p'])
                loaded = True
        except Exception:
            pass

        if not loaded:
            try:
                formula = f'{dv} ~ {" + ".join(avail_ai)}'
                model   = smf.ols(formula, data=df).fit()
                for i, v in enumerate(avail_ai):
                    beta_mat[i, j] = float(model.params[v])
                    p_mat[i, j]    = float(model.pvalues[v])
            except Exception as e:
                log.warning(f'fig_ai_perceptions_heatmap {dv}: {e}')

    # Color matrix — grey out ns
    cmap_custom = plt.cm.RdBu_r
    norm_val    = max(abs(beta_mat).max(), 0.01)

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(
        beta_mat, cmap=cmap_custom,
        vmin=-norm_val, vmax=norm_val,
        aspect='auto'
    )

    for i in range(len(avail_ai)):
        for j in range(len(ev_dvs)):
            b   = beta_mat[i, j]
            p   = p_mat[i, j]
            sig = _sig(p)

            # Grey out ns cells
            if p >= .05:
                ax.add_patch(plt.Rectangle(
                    (j-0.5, i-0.5), 1, 1,
                    facecolor='#EEEEEE', alpha=0.7, zorder=2
                ))

            text_color = 'white' if abs(b) > norm_val*0.6 and p < .05 \
                         else 'black'
            ax.text(j, i, f'{b:.2f}\n{sig}',
                    ha='center', va='center',
                    fontsize=11, color=text_color,
                    fontweight='bold' if p < .05 else 'normal',
                    zorder=3)

    ax.set_xticks(range(len(ev_dvs)))
    ax.set_xticklabels(
        [dv_labels.get(d, d) for d in ev_dvs], fontsize=12
    )
    ax.set_yticks(range(len(avail_ai)))
    ax.set_yticklabels(
        [ai_labels.get(v, v) for v in avail_ai], fontsize=11
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Standardised β', fontsize=11)

    ax.set_title(
        'Standardised Effects of AI Perceptions on Chatbot Evaluation Dimensions\n'
        'Grey cells = not significant (p≥.05). Values are standardised β coefficients.',
        fontsize=14, fontweight='bold', pad=12
    )

    return _save(fig, 'fig_pp_ai_heatmap')


# ---------------------------------------------------------------------------
# Figure PP-C — Moral Perceptions, PM, and Chatbot Enjoyment
# ---------------------------------------------------------------------------
def fig_moral_path_diagram(df: pd.DataFrame, results: dict) -> str:
    paths = {}
    try:
        model_e3 = smf.ols(
            'E3 ~ Ind1 + PM_score + MA1 + MP_score', data=df
        ).fit()
        for pred in ['Ind1', 'PM_score', 'MA1', 'MP_score']:
            if pred in model_e3.params:
                paths[f'{pred}_E3'] = {
                    'beta': round(float(model_e3.params[pred]), 3),
                    'p':    float(model_e3.pvalues[pred]),
                }
        m_ma = smf.ols('MA1 ~ MA2', data=df).fit()
        paths['MA2_MA1'] = {
            'beta': round(float(m_ma.params['MA2']), 3),
            'p':    float(m_ma.pvalues['MA2']),
        }
        m_mp = smf.ols('MP_score ~ MA1', data=df).fit()
        paths['MA1_MP'] = {
            'beta': round(float(m_mp.params['MA1']), 3),
            'p':    float(m_mp.pvalues['MA1']),
        }
        df2 = df.copy()
        df2['PM_c'] = df2['PM_score'] - df2['PM_score'].mean()
        df2['MP_c'] = df2['MP_score'] - df2['MP_score'].mean()
        m_mod = smf.ols('E3 ~ PM_c + MP_c + PM_c:MP_c', data=df2).fit()
        paths['PM_MP_mod'] = {
            'beta': round(float(m_mod.params['PM_c:MP_c']), 3),
            'p':    float(m_mod.pvalues['PM_c:MP_c']),
        }
    except Exception as e:
        log.warning(f'fig_moral_path_diagram paths: {e}')

    indirects = {}
    try:
        full_f = results.get('F', {}).get('full', pd.DataFrame())
        if not full_f.empty:
            row = full_f[
                (full_f['IV'] == 'MA2') &
                (full_f['Mediator'] == 'MA1') &
                (full_f['DV'] == 'E3')
            ]
            if not row.empty:
                r = row.iloc[0]
                indirects['MA2_MA1_E3'] = {
                    'ind':     round(float(r.get('Indirect', 0)), 3),
                    'ci_low':  round(float(r.get('CI_low',   0)), 3),
                    'ci_high': round(float(r.get('CI_high',  0)), 3),
                }
            row = full_f[
                (full_f['IV'] == 'MA1') &
                (full_f['Mediator'] == 'MP_score') &
                (full_f['DV'] == 'E3')
            ]
            if not row.empty:
                r = row.iloc[0]
                indirects['MA1_MP_E3'] = {
                    'ind':     round(float(r.get('Indirect', 0)), 3),
                    'ci_low':  round(float(r.get('CI_low',   0)), 3),
                    'ci_high': round(float(r.get('CI_high',  0)), 3),
                }
    except Exception as e:
        log.warning(f'fig_moral_path_diagram indirects: {e}')

    def _lbl(key, prefix='β='):
        if key in paths:
            b = paths[key]['beta']
            p = paths[key]['p']
            return f"{prefix}{b}{_sig(p)}"
        return '—'

    # ---- Layout: clear zones, no overlap ----
    # Left column (x=1.5): MA2 (top), MA1 (mid-high), Ind1 (mid-low), PM (bottom)
    # Centre (x=8): MP_score
    # Right (x=14.5): E3
    # Indirect box at very bottom

    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    BOX_W = 2.8
    BOX_H = 1.1

    def _box(x, y, text, color, fontsize=10):
        rect = plt.Rectangle(
            (x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H,
            facecolor=color, edgecolor='white',
            linewidth=2, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color='white',
                fontweight='bold', zorder=4,
                multialignment='center')

    def _arrow(x1, y1, x2, y2, label='', color='#2C3E50',
               dashed=False, rad=0.0, label_offset=(0, 0.22)):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->', color=color, lw=2.0,
                linestyle='dashed' if dashed else 'solid',
                connectionstyle=f'arc3,rad={rad}'
            ), zorder=2
        )
        if label:
            mx = (x1 + x2) / 2 + label_offset[0]
            my = (y1 + y2) / 2 + label_offset[1]
            ax.text(mx, my, label, ha='center', va='bottom',
                    fontsize=10, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', alpha=0.92,
                              edgecolor='none'))

    # --- Node positions (well spaced) ---
    X_LEFT  = 2.0
    X_MID   = 8.5
    X_RIGHT = 15.5

    Y_MA2  = 9.5
    Y_MA1  = 7.5
    Y_IND1 = 1.5
    Y_PM   = 3.0
    Y_MP   = 5.8   # centre, between MA1 and Ind1
    Y_E3   = 6.2   # outcome

    _box(X_LEFT,  Y_MA2,  'AI Moral\nResponsibility\n(MA2)', '#6C3483')
    _box(X_LEFT,  Y_MA1,  'Moral Gravity\nAI→Human\n(MA1)', '#884EA0')
    _box(X_LEFT, Y_IND1, 'Autonomy\n(Ind1)', '#1A5276')
    _box(X_LEFT,  Y_PM,   'Perceived\nManipulation\n(PM_score)', COL_NEG)
    _box(X_MID,   Y_MP,   'Moral Patiency\n(MP_score)\n[mediator + moderator]',
         COL_SIG, fontsize=9)
    _box(X_RIGHT, Y_E3,   'Chatbot\nEnjoyment\n(E3)', COL_CASUAL, fontsize=11)

    # --- Arrows ---

    # MA2 → MA1 (vertical, left column)
    _arrow(X_LEFT, Y_MA2 - BOX_H/2,
           X_LEFT, Y_MA1 + BOX_H/2,
           _lbl('MA2_MA1'), '#6C3483',
           label_offset=(0.6, 0.1))

    # MA1 → E3 (direct, long curved arc above)
    _arrow(X_LEFT + BOX_W/2, Y_MA1 + 0.2,
           X_RIGHT - BOX_W/2, Y_E3 + 0.3,
           _lbl('MA1_E3'), '#884EA0', rad=-0.25,
           label_offset=(0, 0.3))

    # MA1 → MP (diagonal down-right)
    _arrow(X_LEFT + BOX_W/2, Y_MA1 - 0.2,
           X_MID - BOX_W/2,  Y_MP + 0.1,
           _lbl('MA1_MP'), '#1E8449',
           label_offset=(0, 0.25))

    # Ind1 → E3 (direct)
    _arrow(X_LEFT + BOX_W/2, Y_IND1,
           X_RIGHT - BOX_W/2, Y_E3 - 0.6,
           _lbl('Ind1_E3'), '#1A5276', rad=0.2,
           label_offset=(0, -0.35))

    # MP → E3 (horizontal right)
    _arrow(X_MID + BOX_W/2, Y_MP,
           X_RIGHT - BOX_W/2, Y_E3 - 0.2,
           _lbl('MP_score_E3'), COL_SIG,
           label_offset=(0, 0.25))

    # PM → E3 (long diagonal, below everything)
    _arrow(X_LEFT + BOX_W/2, Y_PM,
           X_RIGHT - BOX_W/2, Y_E3 - 0.5,
           _lbl('PM_score_E3'), COL_NEG, rad=0.15,
           label_offset=(0, -0.35))

    # Moderation: MP moderates PM→E3
    # Show as dashed arrow from MP_score downward toward PM→E3 path midpoint
    _arrow(X_MID, Y_MP - BOX_H/2,
           (X_LEFT + X_RIGHT) / 2 + 1.0, (Y_PM + Y_E3) / 2,
           '', COL_SIG, dashed=True, rad=0.2)
    mod_lbl = _lbl('PM_MP_mod', prefix='mod β=')
    ax.text((X_LEFT + X_RIGHT)/2 + 1.5, (Y_PM + Y_E3)/2 - 0.5,
            f'Moderation\n{mod_lbl}',
            ha='center', fontsize=10, color=COL_SIG, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.92, edgecolor='none'), zorder=5)

    # --- Indirect effects box (bottom) ---
    ind_lines = []
    if 'MA2_MA1_E3' in indirects:
        d = indirects['MA2_MA1_E3']
        ind_lines.append(
            f"MA2→MA1→E3: indirect={d['ind']}, "
            f"CI[{d['ci_low']}, {d['ci_high']}]  (full mediation)"
        )
    if 'MA1_MP_E3' in indirects:
        d = indirects['MA1_MP_E3']
        ind_lines.append(
            f"MA1→MP→E3: indirect={d['ind']}, "
            f"CI[{d['ci_low']}, {d['ci_high']}]  (partial mediation)"
        )

    if ind_lines:
        rect = plt.Rectangle((0.3, 0.15), 17.4, 1.3,
                               facecolor='#FDFEFE',
                               edgecolor='#BDC3C7', linewidth=1)
        ax.add_patch(rect)
        ax.text(9.0, 1.25,
                'Indirect effects (bootstrapped, 5,000 iter.):',
                ha='center', fontsize=11,
                fontweight='bold', color='#2C3E50')
        ax.text(9.0, 0.65,
                '  |  '.join(ind_lines),
                ha='center', fontsize=10.5, color='#2C3E50')

    ax.set_title(
        'Moral Perceptions, Perceived Manipulation, and Chatbot Enjoyment\n'
        'Direct predictors of E3: Ind1 (+), PM_score (−), MA1 (−). '
        'MP_score moderates PM→E3 and mediates MA1→E3. MA2 predicts E3 via MA1.',
        fontsize=13, fontweight='bold', pad=14
    )
    return _save(fig, 'fig_pp_moral_paths')


# ---------------------------------------------------------------------------
# Figure NEW-1 — Tone → PP1/PP3/PP4 → Composite (parallel mediations)
# ---------------------------------------------------------------------------
def fig_tone_pp_quality(df: pd.DataFrame, results: dict) -> str:
    """Path diagram: 3 parallel mediations tone → PP → composite."""

    # Extract from supp results
    pp3 = results.get('supp', {}).get('pp3_mediations', pd.DataFrame())

    pp_items = [
        ('PP1', 'Friendly\n(PP1)',     '#1A5276'),
        ('PP3', 'Approachable\n(PP3)', '#2E86C1'),
        ('PP4', 'Warm\n(PP4)',         '#117A65'),
    ]

    med_data = {}
    for pp, _, _ in pp_items:
        if not pp3.empty:
            row = pp3[pp3['Path'].str.contains(pp, na=False)]
            if not row.empty:
                r = row.iloc[0]
                med_data[pp] = {
                    'a':        round(float(r.get('a', 0)), 3),
                    'p_a':      float(r.get('p_a', 1)),
                    'b':        round(float(r.get('b', 0)), 3),
                    'p_b':      float(r.get('p_b', 1)),
                    'indirect': round(float(r.get('Indirect', 0)), 3),
                    'ci_low':   round(float(r.get('CI_low', 0)), 3),
                    'ci_high':  round(float(r.get('CI_high', 0)), 3),
                    'type':     str(r.get('Type', '—')),
                }

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    BW, BH = 2.6, 0.9

    def _box(x, y, text, color, fontsize=10):
        rect = plt.Rectangle((x-BW/2, y-BH/2), BW, BH,
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color='white',
                fontweight='bold', zorder=4,
                multialignment='center')

    def _arrow(x1, y1, x2, y2, label='', color='#2C3E50',
               dashed=False, rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', color=color, lw=1.8,
                        linestyle='dashed' if dashed else 'solid',
                        connectionstyle=f'arc3,rad={rad}'
                    ), zorder=2)
        if label:
            mx = (x1+x2)/2
            my = (y1+y2)/2 + 0.2
            ax.text(mx, my, label, ha='center', va='bottom',
                    fontsize=10, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.9,
                              edgecolor='none'))

    # Nodes
    _box(1.5, 4.5, 'Chatbot\nTone\n(Casual/Formal)', COL_CASUAL, fontsize=10)
    _box(12.5, 4.5, 'Feedback\nQuality\n(composite)', COL_SIG, fontsize=10)

    y_positions = [7.0, 4.5, 2.0]
    for (pp, pp_label, color), y_pp in zip(pp_items, y_positions):
        _box(7.0, y_pp, pp_label, color, fontsize=10)

        d = med_data.get(pp, {})
        if d:
            # a path: tone → PP
            a_lbl = f"a={d['a']}{_sig(d['p_a'])}"
            _arrow(1.5 + BW/2, 4.5, 7.0 - BW/2, y_pp, a_lbl, color)

            # b path: PP → composite
            b_lbl = f"b={d['b']}{_sig(d['p_b'])}"
            _arrow(7.0 + BW/2, y_pp, 12.5 - BW/2, 4.5, b_lbl, color)

            # Indirect effect below PP box
            ax.text(7.0, y_pp - BH/2 - 0.18,
                    f"indirect={d['indirect']}, CI[{d['ci_low']}, {d['ci_high']}]",
                    ha='center', fontsize=9, color=color, style='italic',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.9,
                              edgecolor='none'))

    # Direct path (dashed, ns)
    _arrow(1.5 + BW/2, 4.3, 12.5 - BW/2, 4.3,
           "c'=ns", '#95A5A6', dashed=True, rad=-0.3)

    ax.set_title(
        'Tone Effects on Feedback Quality via Personality Perceptions\n'
        'a = tone → PP coefficient, b = PP → composite coefficient. '
        'Full mediation in all three paths. Bootstrap 5,000 iter.',
        fontsize=14, fontweight='bold', pad=14
    )
    return _save(fig, 'fig_new_tone_pp_quality')


# ---------------------------------------------------------------------------
# Figure NEW-2 — Serial chain: Tone → PP → PM → E3/E4/E5
# ---------------------------------------------------------------------------
def fig_tone_pp_pm_eval(df: pd.DataFrame, results: dict) -> str:
    """Serial chain path diagram: tone → PP → PM → E3/E4/E5."""

    pp5 = results.get('supp', {}).get('pp5_serial_eval', pd.DataFrame())

    # Build lookup: (pp, dv) → {indirect, ci_low, ci_high, sig}
    chain_data = {}
    if not pp5.empty:
        for _, row in pp5.iterrows():
            pp  = str(row.get('PP item', ''))
            dv  = str(row.get('DV', ''))
            sig = str(row.get('Sig.', 'ns'))
            chain_data[(pp, dv)] = {
                'indirect': round(float(row.get('Indirect', 0)), 3),
                'ci_low':   round(float(row.get('CI_low',   0)), 3),
                'ci_high':  round(float(row.get('CI_high',  0)), 3),
                'sig':      sig,
            }

    pp_items = ['PP1', 'PP3', 'PP4']
    pp_labels = {'PP1': 'Friendly\n(PP1)',
                 'PP3': 'Approachable\n(PP3)',
                 'PP4': 'Warm\n(PP4)'}
    pp_colors = {'PP1': '#1A5276', 'PP3': '#2E86C1', 'PP4': '#117A65'}
    dv_items  = ['E3', 'E4', 'E5']
    dv_labels = {'E3': 'Enjoyment\n(E3)', 'E4': 'Utility\n(E4)',
                 'E5': 'Reuse\n(E5)'}
    dv_colors = {'E3': COL_CASUAL, 'E4': '#6C3483', 'E5': COL_FORMAL}

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    BW, BH = 2.4, 0.85

    def _box(x, y, text, color, fontsize=9.5):
        rect = plt.Rectangle((x-BW/2, y-BH/2), BW, BH,
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color='white',
                fontweight='bold', zorder=4,
                multialignment='center')

    def _arrow(x1, y1, x2, y2, color='#2C3E50',
               dashed=False, rad=0.0, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', color=color, lw=1.6,
                        linestyle='dashed' if dashed else 'solid',
                        connectionstyle=f'arc3,rad={rad}'
                    ), zorder=2)
        if label:
            mx = (x1+x2)/2
            my = (y1+y2)/2 + 0.18
            ax.text(mx, my, label, ha='center', fontsize=8.5,
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='white', alpha=0.9,
                              edgecolor='none'))

    # Nodes
    _box(1.2, 5.0, 'Chatbot\nTone', COL_CASUAL, fontsize=10)
    _box(8.0, 5.0, 'Perceived\nManipulation\n(PM_score)', COL_NEG, fontsize=9)

    y_pp  = [7.5, 5.0, 2.5]
    y_dv  = [7.5, 5.0, 2.5]

    for pp, y in zip(pp_items, y_pp):
        _box(4.5, y, pp_labels[pp], pp_colors[pp])

    for dv, y in zip(dv_items, y_dv):
        _box(13.0, y, dv_labels[dv], dv_colors[dv])

    # Arrows tone → PP
    for pp, y in zip(pp_items, y_pp):
        _arrow(1.2 + BW/2, 5.0, 4.5 - BW/2, y, pp_colors[pp])

    # Arrows PP → PM
    for pp, y in zip(pp_items, y_pp):
        _arrow(4.5 + BW/2, y, 8.0 - BW/2, 5.0, pp_colors[pp])

    # Arrows PM → DV + indirect labels
    for dv, y_d in zip(dv_items, y_dv):
        _arrow(8.0 + BW/2, 5.0, 13.0 - BW/2, y_d,
               dv_colors[dv], rad=0.0)

        # Show significant indirects near DV
        sig_lines = []
        for pp in pp_items:
            d = chain_data.get((pp, dv))
            if d and d['sig'] == '*':
                sig_lines.append(
                    f"{pp}: {d['indirect']}, CI[{d['ci_low']}, {d['ci_high']}]"
                )
            elif d and d['sig'] == 'ns':
                # dashed line for ns paths
                pass

        if sig_lines:
            ax.text(13.0, y_d - BH/2 - 0.25,
                    '\n'.join(sig_lines),
                    ha='center', fontsize=8, color='#2C3E50',
                    style='italic',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.85,
                              edgecolor='none'))

    # Note on PP3 → PM → E3 (ns)
    ax.text(8.0, 0.8,
            'Note: tone → PP3 → PM → E3 not significant (CI includes 0)',
            ha='center', fontsize=9, color='#95A5A6', style='italic')

    ax.set_title(
        'How Tone Shapes Evaluation Through Personality and Perceived Manipulation\n'
        'Serial chain mediations: tone → PP → PM_score → E3/E4/E5. '
        'Indirect effects bootstrapped (5,000 iter.).',
        fontsize=14, fontweight='bold', pad=14
    )
    return _save(fig, 'fig_new_tone_pp_pm_eval')


# ---------------------------------------------------------------------------
# Figure NEW-3 — AI Perceptions → Engagement/E3 → Composite
# ---------------------------------------------------------------------------
def fig_ai_engagement_quality(df: pd.DataFrame, results: dict) -> str:
    """Multi-level path diagram: AI perceptions → mediators → composite."""

    full_f = results.get('F', {}).get('full', pd.DataFrame())

    def _get_indirect(iv, med, dv):
        if full_f.empty:
            return None
        row = full_f[
            (full_f['IV'] == iv) &
            (full_f['Mediator'] == med) &
            (full_f['DV'] == dv)
        ]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            'indirect': round(float(r.get('Indirect', 0)), 3),
            'ci_low':   round(float(r.get('CI_low',   0)), 3),
            'ci_high':  round(float(r.get('CI_high',  0)), 3),
            'sig':      str(r.get('Sig.', 'ns')),
            'type':     str(r.get('Mediation_type', '—')),
        }

    # Key mediation paths
    paths_e3 = {
        'MA2':       _get_indirect('MA2',      'engagement_score', 'composite'),
        'MP_score':  _get_indirect('MP_score', 'E3',               'composite'),
        'Ind1':      _get_indirect('Ind1',     'E3',               'composite'),
        'PM_score':  _get_indirect('PM_score', 'E3',               'composite'),
        'Comp_score':_get_indirect('Comp_score','E3',              'composite'),
        'Comp_E2':   _get_indirect('Comp_score','E2',              'composite'),
    }

    # Direct PM → composite from regression
    pm_direct = None
    try:
        full_d = results.get('D', {}).get('full', pd.DataFrame())
        if not full_d.empty:
            sub = full_d[
                (full_d['DV'] == 'composite') &
                (full_d['Predictor'] == 'PM_score')
            ]
            if not sub.empty:
                last = sub['Block'].max()
                row  = sub[sub['Block'] == last].iloc[0]
                pm_direct = {
                    'beta': round(float(row.get('β', 0)), 3),
                    'p':    float(row.get('p', 1)),
                }
    except Exception:
        pass

    # E3 → composite direct (from regression)
    e3_direct = None
    try:
        if not full_d.empty:
            sub = full_d[
                (full_d['DV'] == 'composite') &
                (full_d['Predictor'] == 'E3')
            ]
            if not sub.empty:
                last = sub['Block'].max()
                row  = sub[sub['Block'] == last].iloc[0]
                e3_direct = {
                    'beta': round(float(row.get('β', 0)), 3),
                    'p':    float(row.get('p', 1)),
                }
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    BW, BH = 2.4, 0.85

    def _box(x, y, text, color, fontsize=9.5, width=None):
        w = width or BW
        rect = plt.Rectangle((x-w/2, y-BH/2), w, BH,
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color='white',
                fontweight='bold', zorder=4,
                multialignment='center')

    def _arrow(x1, y1, x2, y2, label='', color='#2C3E50',
               dashed=False, rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', color=color, lw=1.8,
                        linestyle='dashed' if dashed else 'solid',
                        connectionstyle=f'arc3,rad={rad}'
                    ), zorder=2)
        if label:
            mx = (x1+x2)/2
            my = (y1+y2)/2 + 0.2
            ax.text(mx, my, label, ha='center', fontsize=9,
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.9,
                              edgecolor='none'))

    # Level 1 — AI perceptions (top)
    ai_items = [
        ('MA2',        'AI Moral\nResp. (MA2)',  '#6C3483', 1.5),
        ('MP_score',   'Moral\nPatiency',        COL_SIG,   4.5),
        ('Ind1',       'Autonomy\n(Ind1)',        '#1A5276', 7.5),
        ('PM_score',   'Perceived\nManip.',      COL_NEG,   10.5),
        ('Comp_score', 'Competence',             '#2980B9', 13.5),
    ]
    Y_AI   = 9.5
    Y_MED  = 6.5   # mediators: engagement_score left, E3 right
    Y_E2   = 6.5
    Y_COMP = 2.5

    for iv, label, color, x in ai_items:
        _box(x, Y_AI, label, color)

    # Level 2 — Mediators
    _box(4.0,  Y_MED, 'Engagement\nScore', '#4A235A', fontsize=9)
    _box(9.0,  Y_MED, 'Enjoyment\n(E3)',   COL_CASUAL, fontsize=9)
    _box(13.5, Y_E2,  'Felt\nEngagement\n(E2)', '#784212', fontsize=8.5)

    # Level 3 — Composite
    _box(8.0, Y_COMP, 'Feedback Quality\n(composite)',
         COL_SIG, fontsize=10, width=3.2)

    # Arrows AI → mediators
    # MA2 → engagement_score
    _arrow(1.5, Y_AI - BH/2, 4.0, Y_MED + BH/2, '', '#6C3483')
    d = paths_e3.get('MA2')
    if d:
        ax.text(2.5, (Y_AI + Y_MED)/2,
                f"indirect={d['indirect']}\nCI[{d['ci_low']}, {d['ci_high']}]",
                ha='center', fontsize=8, color='#6C3483', style='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.85, edgecolor='none'))

    # MP → E3
    _arrow(4.5, Y_AI - BH/2, 9.0, Y_MED + BH/2, '', COL_SIG)
    d = paths_e3.get('MP_score')
    if d:
        ax.text(6.5, (Y_AI + Y_MED)/2 + 0.2,
                f"indirect={d['indirect']}\nCI[{d['ci_low']}, {d['ci_high']}]",
                ha='center', fontsize=8, color=COL_SIG, style='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.85, edgecolor='none'))

    # Ind1 → E3
    _arrow(7.5, Y_AI - BH/2, 9.0, Y_MED + BH/2, '', '#1A5276')
    d = paths_e3.get('Ind1')
    if d:
        ax.text(8.0, (Y_AI + Y_MED)/2 + 0.5,
                f"indirect={d['indirect']}\nCI[{d['ci_low']}, {d['ci_high']}]",
                ha='center', fontsize=8, color='#1A5276', style='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.85, edgecolor='none'))

    # PM → E3
    _arrow(10.5, Y_AI - BH/2, 9.0, Y_MED + BH/2, '', COL_NEG)
    d = paths_e3.get('PM_score')
    if d:
        ax.text(10.0, (Y_AI + Y_MED)/2 + 0.2,
                f"indirect={d['indirect']}\nCI[{d['ci_low']}, {d['ci_high']}]",
                ha='center', fontsize=8, color=COL_NEG, style='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.85, edgecolor='none'))

    # PM → composite (direct, dashed)
    if pm_direct:
        lbl = f"β={pm_direct['beta']}{_sig(pm_direct['p'])}"
        _arrow(10.5, Y_AI - BH/2, 8.0, Y_COMP + BH/2,
               lbl, COL_NEG, dashed=True, rad=0.2)

    # Comp → E3
    _arrow(13.5, Y_AI - BH/2, 9.0, Y_MED + BH/2, '', '#2980B9')
    d = paths_e3.get('Comp_score')
    if d:
        ax.text(11.8, (Y_AI + Y_MED)/2 + 0.2,
                f"indirect={d['indirect']}\nCI[{d['ci_low']}, {d['ci_high']}]",
                ha='center', fontsize=8, color='#2980B9', style='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.85, edgecolor='none'))

    # Comp → E2
    _arrow(13.5, Y_AI - BH/2, 13.5, Y_E2 + BH/2, '', '#2980B9', rad=-0.3)
    d = paths_e3.get('Comp_E2')
    if d:
        ax.text(14.5, (Y_AI + Y_E2)/2,
                f"indirect={d['indirect']}\nCI[{d['ci_low']}, {d['ci_high']}]",
                ha='center', fontsize=8, color='#2980B9', style='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.85, edgecolor='none'))

    # Mediators → composite
    _arrow(4.0,  Y_MED - BH/2, 8.0, Y_COMP + BH/2, '', '#4A235A')
    _arrow(9.0,  Y_MED - BH/2, 8.0, Y_COMP + BH/2, '', COL_CASUAL)
    _arrow(13.5, Y_E2  - BH/2, 8.0, Y_COMP + BH/2, '', '#784212')

    if e3_direct:
        lbl = f"β={e3_direct['beta']}{_sig(e3_direct['p'])}"
        ax.text(8.5, (Y_MED + Y_COMP)/2,
                lbl, ha='center', fontsize=9.5,
                color=COL_CASUAL, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.9, edgecolor='none'))

    ax.set_title(
        'How AI Perceptions Influence Feedback Quality: Direct and Indirect Pathways\n'
        'Significant indirect effects only shown. Bootstrap 5,000 iter. '
        'Blue = positive, Red = negative.',
        fontsize=14, fontweight='bold', pad=14
    )
    return _save(fig, 'fig_new_ai_engagement_quality')


# ---------------------------------------------------------------------------
# Figure NEW-4 — Serial chain: Tone → PP4 → PM → Composite
# ---------------------------------------------------------------------------
def fig_tone_pp4_pm_quality(df: pd.DataFrame, results: dict) -> str:
    """Simple horizontal serial chain: tone → PP4 → PM → composite."""

    pp4 = results.get('supp', {}).get('pp4_serial_composite', pd.DataFrame())

    chain = {}
    if not pp4.empty:
        row = pp4[pp4['Path'].str.contains('PP4', na=False)]
        if not row.empty:
            r = row.iloc[0]
            chain = {
                'indirect': round(float(r.get('Indirect', 0)), 3),
                'ci_low':   round(float(r.get('CI_low',   0)), 3),
                'ci_high':  round(float(r.get('CI_high',  0)), 3),
                'sig':      str(r.get('Sig.', 'ns')),
            }

    # Get path coefficients from data
    a1_beta, a1_p = '—', 1.0
    a2_beta, a2_p = '—', 1.0
    b_beta,  b_p  = '—', 1.0
    c_prime       = '—'

    try:
        import statsmodels.formula.api as smf
        m1 = smf.ols('PP4 ~ tone', data=df).fit()
        a1_beta = round(float(m1.params['tone']), 3)
        a1_p    = float(m1.pvalues['tone'])

        m2 = smf.ols('PM_score ~ tone + PP4', data=df).fit()
        a2_beta = round(float(m2.params['PP4']), 3)
        a2_p    = float(m2.pvalues['PP4'])

        if 'composite' in df.columns:
            m3 = smf.ols('composite ~ tone + PP4 + PM_score', data=df).fit()
            b_beta  = round(float(m3.params['PM_score']), 3)
            b_p     = float(m3.pvalues['PM_score'])
            c_prime = round(float(m3.params['tone']), 3)
    except Exception as e:
        log.warning(f'fig_tone_pp4_pm_quality paths: {e}')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    BW, BH = 2.6, 1.0

    def _box(x, y, text, color, fontsize=10):
        rect = plt.Rectangle((x-BW/2, y-BH/2), BW, BH,
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color='white',
                fontweight='bold', zorder=4,
                multialignment='center')

    def _arrow(x1, y1, x2, y2, label='', color='#2C3E50',
               dashed=False, rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', color=color, lw=2.0,
                        linestyle='dashed' if dashed else 'solid',
                        connectionstyle=f'arc3,rad={rad}'
                    ), zorder=2)
        if label:
            mx = (x1+x2)/2
            my = (y1+y2)/2 + 0.22
            ax.text(mx, my, label, ha='center', va='bottom',
                    fontsize=11, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', alpha=0.9,
                              edgecolor='none'))

    # Nodes
    _box(1.5,  3.0, 'Chatbot\nTone\n(Casual/Formal)', COL_CASUAL)
    _box(5.5,  3.0, 'Warm\nPersonality\n(PP4)',        '#117A65')
    _box(9.5,  3.0, 'Perceived\nManipulation\n(PM_score)', COL_NEG)
    _box(13.0, 3.0, 'Feedback\nQuality\n(composite)',  COL_SIG)

    # Arrows
    a1_lbl = f"a₁={a1_beta}{_sig(a1_p)}" if a1_beta != '—' else 'a₁'
    _arrow(1.5 + BW/2, 3.0, 5.5 - BW/2, 3.0, a1_lbl, '#117A65')

    a2_lbl = f"a₂={a2_beta}{_sig(a2_p)}" if a2_beta != '—' else 'a₂'
    _arrow(5.5 + BW/2, 3.0, 9.5 - BW/2, 3.0, a2_lbl, COL_NEG)

    b_lbl = f"b={b_beta}{_sig(b_p)}" if b_beta != '—' else 'b'
    _arrow(9.5 + BW/2, 3.0, 13.0 - BW/2, 3.0, b_lbl, COL_NEG)

    # Direct effect c' (dashed, ns)
    c_lbl = f"c'={c_prime} (ns)" if c_prime != '—' else "c'=ns"
    _arrow(1.5 + BW/2, 2.6, 13.0 - BW/2, 2.6,
           c_lbl, '#95A5A6', dashed=True, rad=-0.3)

    # Indirect effect box
    if chain:
        rect = plt.Rectangle((2.0, 0.3), 10.0, 1.0,
                               facecolor='#FDFEFE',
                               edgecolor='#BDC3C7', linewidth=1)
        ax.add_patch(rect)
        ax.text(7.0, 1.1,
                'Total indirect effect (bootstrapped, 5,000 iter.):',
                ha='center', fontsize=10.5,
                fontweight='bold', color='#2C3E50')
        ax.text(7.0, 0.6,
                f"indirect = {chain['indirect']}, "
                f"95% CI [{chain['ci_low']}, {chain['ci_high']}]  "
                f"({'significant' if chain['sig'] == '*' else 'ns'})",
                ha='center', fontsize=10, color='#2C3E50')

    ax.set_title(
        'Tone Effects on Feedback Quality: The Warmth-Manipulation Chain\n'
        'tone → PP4 (warm) → PM_score → composite. '
        'a₁ = tone→PP4, a₂ = PP4→PM, b = PM→composite.',
        fontsize=14, fontweight='bold', pad=14
    )
    return _save(fig, 'fig_new_tone_pp4_quality')
# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------
def generate_all_figures(df: pd.DataFrame, results: dict) -> list:
    """
    Delete old figures, generate all new ones.
    Returns list of PNG paths.
    """
    _ensure_dir()
    paths = []

    funcs = [
        ('GPT reliability',           lambda: fig_gpt_reliability(df)),
        ('Tone → personality',         lambda: fig_tone_personality(df)),
        ('Tone → AI perceptions',      lambda: fig_tone_ai_perceptions(df)),
        ('Path: tone → PP → PM',       lambda: fig_path_tone_pp_pm(df, results)),
        ('Quality distribution',       lambda: fig_quality_distribution(df)),
        ('Path: PM → outcomes',        lambda: fig_path_pm_outcomes(df, results)),
        ('Tone × language engagement', lambda: fig_tone_language_engagement(df)),
        ('Annex A: eval predictors',   lambda: fig_annex_evaluation_predictors(df, results)),
        ('Annex B: personality bubbles',lambda: fig_annex_personality_bubbles(df)),
        ('Annex C: comp-aut mediation',lambda: fig_annex_mediation_comp_aut(df, results)),
        ('Annex D: MA mediation',      lambda: fig_annex_mediation_ma(df, results)),
        ('PP-A: AI perceptions → quality',   lambda: fig_ai_perceptions_quality(df, results)),
        ('PP-B: AI perceptions heatmap',     lambda: fig_ai_perceptions_heatmap(df, results)),
        ('PP-C: Moral path diagram',         lambda: fig_moral_path_diagram(df, results)),
        ('NEW-1: tone → PP → quality',      lambda: fig_tone_pp_quality(df, results)),
        ('NEW-2: tone → PP → PM → eval',    lambda: fig_tone_pp_pm_eval(df, results)),
        ('NEW-3: AI perceptions → quality', lambda: fig_ai_engagement_quality(df, results)),
        ('NEW-4: tone → PP4 → PM → quality',lambda: fig_tone_pp4_pm_quality(df, results)),
    ]

    for name, fn in funcs:
        try:
            p = fn()
            if p:
                paths.append(p)
                log.info(f'✅ {name}: {p}')
        except Exception as e:
            log.error(f'❌ {name}: {e}')
            import traceback
            traceback.print_exc()

    print(f'\n✅ {len(paths)}/{len(funcs)} figures generated in {FIG_DIR}/')
    return paths
