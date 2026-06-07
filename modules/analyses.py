"""
analyses.py
-----------
Step 3 of the pipeline: run all statistical analyses.

Produces results for sheets:
    A — Correlation matrix (Pearson + FDR correction)
    B — Effect of tone (t-tests / Mann-Whitney + H3 paired t-test)
    C — AI perception regressions (simple OLS, defined a priori)
    D — Predictors of feedback quality (hierarchical regressions)
    E — Predictors of chatbot evaluation (hierarchical regressions)
    F — Mediation analyses (PyProcessMacro, 5000 bootstrap iterations)
    I — Demographics & robustness checks (ANCOVA + interactions)

All predictors are defined a priori — never selected post-hoc.
FDR correction (Benjamini-Hochberg) applied where specified.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pingouin as pg

import config

log = logging.getLogger(__name__)

# Suppress convergence warnings from statsmodels during mediation
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_all_analyses(df: pd.DataFrame) -> dict:
    """
    Run all analyses and return results as a dict of DataFrames.

    Args:
        df: Cleaned + metrics DataFrame from cleaning.py + metrics.py

    Returns:
        Dict with keys A, B, C, D, E, F, I — each a DataFrame or
        list of DataFrames ready for export.py
    """
    results = {}

    log.info("Sheet A — Correlation matrix")
    results["A"] = run_sheet_a(df)

    log.info("Sheet B — Effect of tone")
    results["B"] = run_sheet_b(df)

    log.info("Sheet C — AI perception regressions")
    results["C"] = run_sheet_c(df)

    log.info("Sheet D — Predictors of feedback quality")
    results["D"] = run_sheet_d(df)

    log.info("Sheet E — Predictors of chatbot evaluation")
    results["E"] = run_sheet_e(df)

    log.info("Sheet F — Mediation analyses")
    results["F"] = run_sheet_f(df)

    log.info("Sheet I — Demographics & robustness checks")
    results["I"] = run_sheet_i(df)

    return results

def _get_predictors_for_scale(scale_name: str) -> list:
    """
    Return the right variable(s) for a scale based on Cronbach's alpha.
    If composite validated (alpha >= 0.70) → [scale_score]
    If items separate (alpha < 0.70)       → individual items
    """
    status = getattr(config, 'SCALE_STATUS', {}).get(scale_name, 'separate')
    if status == 'composite':
        return [f"{scale_name}_score"]
    else:
        return config.SCALE_ITEMS.get(scale_name, [])
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Compute Cohen's d for two independent groups."""
    n1, n2   = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan
    return round((group1.mean() - group2.mean()) / pooled_sd, 3)


def _effect_size_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:  return "negligible"
    if d < 0.5:  return "small"
    if d < 0.8:  return "medium"
    return "large"


def _normality_ok(series: pd.Series) -> bool:
    """
    Shapiro-Wilk normality test.
    Returns True if normality is not rejected (p >= 0.05).
    Only applied when n >= 8 (Shapiro-Wilk unreliable below that).
    """
    clean = series.dropna()
    if len(clean) < 8:
        return True   # assume normal for very small samples
    _, p = stats.shapiro(clean)
    return p >= config.ALPHA


def _fdr_correct(p_values: list) -> list:
    """Apply Benjamini-Hochberg FDR correction. Returns corrected p-values."""
    if len(p_values) == 0:
        return []
    _, p_corrected, _, _ = multipletests(p_values, method=config.FDR_METHOD)
    return list(p_corrected)


def _sig_label(p: float) -> str:
    """Return significance stars."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Sheet A — Correlation matrix
# ---------------------------------------------------------------------------
def run_sheet_a(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlations between all continuous variables.
    FDR correction (Benjamini-Hochberg) applied across all pairs.
    Only significant correlations (p_fdr < 0.05) are kept.

    Columns: Variable X, Label X, Variable Y, Label Y,
             r, p_raw, p_fdr, Sig., Strength, Direction
    """
    continuous_vars = [
        "PM1","PM2","PM3","PM4","PM_score",
        "Comp1","Comp2","Comp_score",
        "Ind1","Ind2","Ind_score",
        "MA1","MA2","MA_score",
        "MP1","MP2","MP_score",
        "PP1","PP2","PP3","PP4","PP5",
        "E1","E2","E3","E4","E5","E6",
        "composite","emotions_mean",
        "engagement_score","avg_words_per_turn","chat_duration_sec",
        "age",
    ]

    # Variable labels for readability
    labels = {
        "PM_score":"Perceived Manipulation","Comp_score":"Competence",
        "Ind_score":"Autonomy","MA_score":"Moral Agency",
        "MP_score":"Moral Patiency","E1":"Effort","E2":"Engagement felt",
        "E3":"Appreciation","E4":"Utility","E5":"Reuse intention",
        "E6":"Preference","composite":"Feedback composite",
        "emotions_mean":"Emotions","engagement_score":"Engagement score",
        "avg_words_per_turn":"Avg words/turn",
        "chat_duration_sec":"Chat duration","age":"Age",
    }

    # Keep only variables present in the DataFrame
    vars_present = [v for v in continuous_vars if v in df.columns]

    rows = []
    for i, x in enumerate(vars_present):
        for y in vars_present[i+1:]:
            clean = df[[x, y]].dropna()
            if len(clean) < 5:
                continue
            r, p = stats.pearsonr(clean[x], clean[y])
            rows.append({
                "Variable X": x, "Label X": labels.get(x, x),
                "Variable Y": y, "Label Y": labels.get(y, y),
                "r": round(r, 3), "p_raw": round(p, 4),
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # FDR correction
    result["p_fdr"] = _fdr_correct(result["p_raw"].tolist())
    result["p_fdr"] = result["p_fdr"].round(4)

    # Keep only significant after FDR
    result = result[result["p_fdr"] < config.ALPHA].copy()

    result["Sig."]      = result["p_fdr"].apply(_sig_label)
    result["Strength"]  = result["r"].abs().apply(
        lambda r: "weak" if r < 0.3 else ("moderate" if r < 0.5 else "strong")
    )
    result["Direction"] = result["r"].apply(
        lambda r: "positive" if r > 0 else "negative"
    )

    result = result.sort_values("p_fdr").reset_index(drop=True)
    log.info(f"Sheet A: {len(result)} significant correlations after FDR correction.")
    return result


# ---------------------------------------------------------------------------
# Sheet B — Effect of tone
# ---------------------------------------------------------------------------
def run_sheet_b(df: pd.DataFrame) -> pd.DataFrame:
    """
    Independent samples t-test (or Mann-Whitney U if non-normal)
    for each variable by tone condition (0=professional, 1=friendly).

    Also includes H3: within-participant paired t-test Comp1 vs Comp2.

    Blocks:
        1 — AI Perceptions (PM, Comp, Ind, MA, MP items + composites)
        2 — Perceived Personality (PP1-PP5)
        3 — Chatbot Evaluation (E1-E6)
        4 — Quality & Engagement

    Columns: Variable, Label, Block,
             Mean_friendly, SD_friendly, N_friendly,
             Mean_pro, SD_pro, N_pro,
             Delta, Test, Statistic, p, Cohen_d,
             Effect_size, Sig.
    """
    blocks = {
        "AI Perceptions": [
            "PM1","PM2","PM3","PM4","PM_score",
            "Comp1","Comp2","Comp_score",
            "Ind1","Ind2","Ind_score",
            "MA1","MA2","MA_score",
            "MP1","MP2","MP_score",
        ],
        "Perceived Personality": ["PP1","PP2","PP3","PP4","PP5"],
        "Chatbot Evaluation":    ["E1","E2","E3","E4","E5","E6"],
        "Quality & Engagement":  [
            "quantity_mean","quality_mean","emotions_mean",
            "composite","engagement_score",
        ],
    }

    labels = {
        "PM1":"Threat to freedom","PM2":"Decision override",
        "PM3":"Manipulation attempt","PM4":"Pressure felt",
        "PM_score":"PM composite",
        "Comp1":"Competence — skills","Comp2":"Competence — morality",
        "Comp_score":"Competence composite",
        "Ind1":"AI plans & goals","Ind2":"AI self-control",
        "Ind_score":"Autonomy composite",
        "MA1":"Moral gravity AI→human","MA2":"AI moral responsibility",
        "MA_score":"Moral Agency composite",
        "MP1":"Moral gravity human→AI","MP2":"AI right to consideration",
        "MP_score":"Moral Patiency composite",
        "PP1":"Friendly","PP2":"Professional","PP3":"Approachable",
        "PP4":"Warm","PP5":"Formal",
        "E1":"Effort","E2":"Engagement felt","E3":"Appreciation",
        "E4":"Utility","E5":"Reuse intention","E6":"Preference",
        "quantity_mean":"Feedback quantity","quality_mean":"Feedback quality",
        "emotions_mean":"Emotions","composite":"Feedback composite",
        "engagement_score":"Engagement score",
    }

    friendly = df[df["tone"] == 1]
    pro      = df[df["tone"] == 0]
    rows     = []

    for block_name, variables in blocks.items():
        for var in variables:
            if var not in df.columns:
                continue

            g1 = friendly[var].dropna()
            g2 = pro[var].dropna()

            if len(g1) < 3 or len(g2) < 3:
                continue

            # Choose test based on normality
            normal = _normality_ok(g1) and _normality_ok(g2)
            if normal:
                stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                test_name = "Welch t-test"
            else:
                stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                test_name = "Mann-Whitney U"

            d = _cohens_d(g1, g2)

            rows.append({
                "Variable":       var,
                "Label":          labels.get(var, var),
                "Block":          block_name,
                "Mean_friendly":  round(g1.mean(), 3),
                "SD_friendly":    round(g1.std(), 3),
                "N_friendly":     len(g1),
                "Mean_pro":       round(g2.mean(), 3),
                "SD_pro":         round(g2.std(), 3),
                "N_pro":          len(g2),
                "Delta":          round(g1.mean() - g2.mean(), 3),
                "Test":           test_name,
                "Statistic":      round(stat, 3),
                "p":              round(p, 4),
                "Cohen_d":        d,
                "Effect_size":    _effect_size_label(d),
                "Sig.":           _sig_label(p),
            })

    # H3 — paired t-test: Comp1 vs Comp2 (regardless of tone)
    if "Comp1" in df.columns and "Comp2" in df.columns:
        clean = df[["Comp1","Comp2"]].dropna()
        t_stat, p_h3 = stats.ttest_rel(clean["Comp1"], clean["Comp2"])
        d_h3 = _cohens_d(clean["Comp1"], clean["Comp2"])
        rows.append({
            "Variable":      "Comp1 vs Comp2",
            "Label":         "H3: Skills > Morality judgment (paired t-test)",
            "Block":         "H3 — Hypothesis test",
            "Mean_friendly": round(clean["Comp1"].mean(), 3),
            "SD_friendly":   round(clean["Comp1"].std(), 3),
            "N_friendly":    len(clean),
            "Mean_pro":      round(clean["Comp2"].mean(), 3),
            "SD_pro":        round(clean["Comp2"].std(), 3),
            "N_pro":         len(clean),
            "Delta":         round(clean["Comp1"].mean() - clean["Comp2"].mean(), 3),
            "Test":          "Paired t-test",
            "Statistic":     round(t_stat, 3),
            "p":             round(p_h3, 4),
            "Cohen_d":       d_h3,
            "Effect_size":   _effect_size_label(d_h3),
            "Sig.":          _sig_label(p_h3),
        })

    result = pd.DataFrame(rows)
    log.info(f"Sheet B: {len(result)} tests run.")
    return result


# ---------------------------------------------------------------------------
# Sheet C — Relations between AI perception variables
# ---------------------------------------------------------------------------
def run_sheet_c(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple OLS regressions between AI perception variables.
    Adaptive: uses composite or separate items based on Cronbach's alpha.
    """
    ind  = _get_predictors_for_scale("Ind")
    ma   = _get_predictors_for_scale("MA")
    mp   = _get_predictors_for_scale("MP")
    comp = _get_predictors_for_scale("Comp")
    pm   = _get_predictors_for_scale("PM")

    regressions = []
    for iv in ind:
        for dv in pm:  regressions.append((iv, dv, ""))
        for dv in ma:  regressions.append((iv, dv, "H4"))
        for dv in mp:  regressions.append((iv, dv, ""))
    for iv in comp:
        for dv in pm:  regressions.append((iv, dv, ""))
        for dv in ma:  regressions.append((iv, dv, ""))
        for dv in mp:  regressions.append((iv, dv, ""))
    for iv in ma:
        for dv in pm:  regressions.append((iv, dv, ""))
    for iv in mp:
        for dv in pm:  regressions.append((iv, dv, ""))

    rows = []
    for reg in regressions:
        iv, dv, note = reg[0], reg[1], reg[2]

        if iv not in df.columns or dv not in df.columns:
            log.warning(f"Sheet C: skipping {iv} → {dv} (column missing)")
            continue

        clean = df[[iv, dv]].dropna()
        if len(clean) < 10:
            log.warning(f"Sheet C: {iv} → {dv} has only {len(clean)} rows")
            continue

        formula = f"{dv} ~ {iv}"
        model   = ols(formula, data=clean).fit()

        label = f"{iv}{' (H4)' if note == 'H4' else ''}"
        rows.append({
            "IV":   label,
            "DV":   dv,
            "β":    round(model.params[iv], 3),
            "SE":   round(model.bse[iv], 3),
            "t":    round(model.tvalues[iv], 3),
            "p":    round(model.pvalues[iv], 4),
            "R²":   round(model.rsquared, 3),
            "F":    round(model.fvalue, 3),
            "Sig.": _sig_label(model.pvalues[iv]),
        })

    result = pd.DataFrame(rows)
    log.info(f"Sheet C: {len(result)} regressions run.")
    return result


# ---------------------------------------------------------------------------
# Sheet D — Predictors of feedback quality
# ---------------------------------------------------------------------------
def run_sheet_d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchical regressions for composite and engagement_score.
    Adaptive predictors based on Cronbach's alpha results.
    """
    ai_preds = (
        _get_predictors_for_scale("PM") +
        _get_predictors_for_scale("Comp") +
        _get_predictors_for_scale("Ind") +
        _get_predictors_for_scale("MA") +
        _get_predictors_for_scale("MP")
    )

    dvs = ["composite", "engagement_score"]

    blocks = [
        ["tone"],
        ["tone"] + ai_preds,
        ["tone"] + ai_preds + ["E1", "E2"],
    ]

    return _hierarchical_regression(df, dvs, blocks, sheet_name="D")


# ---------------------------------------------------------------------------
# Sheet E — Predictors of chatbot evaluation
# ---------------------------------------------------------------------------
def run_sheet_e(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchical regressions for E3, E4, E5, E6.
    Adaptive predictors based on Cronbach's alpha results.
    """
    ai_preds = (
        _get_predictors_for_scale("PM") +
        _get_predictors_for_scale("Comp") +
        _get_predictors_for_scale("Ind") +
        _get_predictors_for_scale("MA") +
        _get_predictors_for_scale("MP")
    )

    dvs = ["E3", "E4", "E5", "E6"]

    blocks = [
        ["tone"],
        ["tone"] + ai_preds,
        ["tone"] + ai_preds + ["composite", "engagement_score"],
    ]

    return _hierarchical_regression(df, dvs, blocks, sheet_name="E")

# ---------------------------------------------------------------------------
# Shared hierarchical regression function
# ---------------------------------------------------------------------------
def _hierarchical_regression(
    df: pd.DataFrame,
    dvs: list,
    blocks: list,
    sheet_name: str,
) -> pd.DataFrame:
    """
    Run hierarchical OLS regressions for multiple DVs and blocks.

    For each DV:
        - Fits each block sequentially
        - Computes ΔR² and F-change between consecutive blocks
        - Returns one row per predictor per block per DV

    Args:
        df:          DataFrame with all variables
        dvs:         List of dependent variable names
        blocks:      List of predictor lists (one per block)
        sheet_name:  For logging only

    Returns:
        DataFrame with all results.
    """
    all_rows = []

    for dv in dvs:
        if dv not in df.columns:
            log.warning(f"Sheet {sheet_name}: DV '{dv}' not found — skipped.")
            continue

        prev_r2  = 0.0
        prev_rss = None
        prev_df_resid = None

        for block_num, predictors in enumerate(blocks, start=1):

            # Keep only predictors present in df
            valid_preds = [p for p in predictors if p in df.columns]
            if not valid_preds:
                log.warning(
                    f"Sheet {sheet_name}: Block {block_num} for {dv} "
                    f"has no valid predictors — skipped."
                )
                continue

            cols_needed = valid_preds + [dv]
            clean = df[cols_needed].dropna()

            if len(clean) < len(valid_preds) + 5:
                log.warning(
                    f"Sheet {sheet_name}: {dv} Block {block_num} — "
                    f"only {len(clean)} complete rows, skipping."
                )
                continue

            formula = f"{dv} ~ {' + '.join(valid_preds)}"
            model   = ols(formula, data=clean).fit()

            # ΔR² and F-change
            delta_r2 = round(model.rsquared - prev_r2, 4)
            curr_rss  = model.ssr
            curr_df_resid = model.df_resid

            if prev_rss is not None and prev_df_resid is not None:
                df_change = prev_df_resid - curr_df_resid
                if df_change > 0 and curr_rss > 0:
                    f_change = (
                        (prev_rss - curr_rss) / df_change
                    ) / (curr_rss / curr_df_resid)
                    p_f_change = 1 - stats.f.cdf(
                        f_change, df_change, curr_df_resid
                    )
                else:
                    f_change   = np.nan
                    p_f_change = np.nan
            else:
                f_change   = model.fvalue
                p_f_change = model.f_pvalue

            # One row per predictor
            for pred in valid_preds:
                if pred not in model.params:
                    continue
                all_rows.append({
                    "DV":       dv,
                    "Block":    block_num,
                    "Predictor": pred,
                    "β":        round(model.params[pred], 3),
                    "SE":       round(model.bse[pred], 3),
                    "t":        round(model.tvalues[pred], 3),
                    "p":        round(model.pvalues[pred], 4),
                    "R²":       round(model.rsquared, 3),
                    "ΔR²":      delta_r2 if pred == valid_preds[0] else "",
                    "F_change": round(f_change, 3) if pred == valid_preds[0] else "",
                    "p_Fchange": round(p_f_change, 4) if pred == valid_preds[0] else "",
                    "Sig.":     _sig_label(model.pvalues[pred]),
                })

            prev_r2       = model.rsquared
            prev_rss      = curr_rss
            prev_df_resid = curr_df_resid

    result = pd.DataFrame(all_rows)
    log.info(
        f"Sheet {sheet_name}: {len(dvs)} DVs x {len(blocks)} blocks completed."
    )
    return result


# ---------------------------------------------------------------------------
# Sheet F — Mediation analyses
# ---------------------------------------------------------------------------
def run_sheet_f(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mediation analyses — adaptive based on Cronbach's alpha results.
    """
    try:
        from pyprocessmacro import Process
    except ImportError:
        log.error("pyprocessmacro not installed.")
        return pd.DataFrame({"error": ["pyprocessmacro not installed"]})

    ind  = _get_predictors_for_scale("Ind")
    ma   = _get_predictors_for_scale("MA")
    mp   = _get_predictors_for_scale("MP")
    comp = _get_predictors_for_scale("Comp")
    pm   = _get_predictors_for_scale("PM")

    models = []

    # Series 1 — Tone as IV
    for p in pm:
        models += [
            ("tone", p, "composite",        "1"),
            ("tone", p, "engagement_score", "1"),
            ("tone", p, "E3",               "1"),
            ("tone", p, "E4",               "1"),
            ("tone", p, "E5",               "1"),
            ("tone", p, "E6",               "1"),
        ]
    for p in comp: models.append(("tone", p, "composite", "1"))
    for p in ma:   models.append(("tone", p, "composite", "1"))
    for p in mp:   models.append(("tone", p, "composite", "1"))
    for pp in ["PP1","PP2","PP3","PP4","PP5"]:
        for p in pm:
            models.append(("tone", pp, p, "1"))

    # Series 2 — AI Perceptions as IV
    for p in pm:
        models += [
            (p, "E2",            "composite", "2"),
            (p, "emotions_mean", "composite", "2"),
        ]
    for p in ind:
        for q in pm: models.append((p, q, "composite", "2"))
        for q in ma: models.append((p, q, "composite", "2"))
        for q in mp: models.append((p, q, "composite", "2"))
    for p in comp:
        for q in ma:
            for dv in ["E3","E4","E5","E6"]:
                models.append((p, q, dv, "2"))
        for q in mp:
            for dv in ["E3","E4","E5","E6"]:
                models.append((p, q, dv, "2"))
    for p in ma:
        for q in pm: models.append((p, q, "composite", "2"))
    for p in mp:
        for q in pm: models.append((p, q, "composite", "2"))

    # Chain mediations
    chain_models = []
    for i in ind:
        for p in pm:
            chain_models.append(("tone", [i, p], "composite", "1"))

    rows = []
    for iv, med, dv, series in models:
        row = _run_simple_mediation(df, iv, med, dv, series, Process)
        if row:
            rows.append(row)

    for iv, mediators, dv, series in chain_models:
        row = _run_chain_mediation(df, iv, mediators, dv, series, Process)
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # FDR correction
    p_vals = []
    for _, r in result.iterrows():
        ci_low  = r.get("CI_low", np.nan)
        ci_high = r.get("CI_high", np.nan)
        if pd.notna(ci_low) and pd.notna(ci_high):
            excludes_zero = (ci_low > 0) or (ci_high < 0)
            p_vals.append(0.01 if excludes_zero else 0.50)
        else:
            p_vals.append(np.nan)

    valid_mask = [not np.isnan(p) for p in p_vals]
    valid_ps   = [p for p, m in zip(p_vals, valid_mask) if m]

    if valid_ps:
        corrected      = _fdr_correct(valid_ps)
        corrected_iter = iter(corrected)
        result["p_fdr"] = [
            round(next(corrected_iter), 4) if m else np.nan
            for m in valid_mask
        ]
    else:
        result["p_fdr"] = np.nan

    result["Sig."] = result["p_fdr"].apply(
        lambda p: _sig_label(p) if pd.notna(p) else "ns"
    )

    log.info(f"Sheet F: {len(result)} mediation models run.")
    return result
    
    # ------------------------------------------------------------------
    # Define all mediation models a priori
    # (iv, mediator, dv, series_label)
    # Chain mediations use a list of mediators [M1, M2]
    # ------------------------------------------------------------------
    models = [
        # Series 1 — Tone as IV
        ("tone", "PM_score",   "composite",        "1"),
        ("tone", "PM_score",   "engagement_score", "1"),
        ("tone", "PM_score",   "E3",               "1"),
        ("tone", "PM_score",   "E4",               "1"),
        ("tone", "PM_score",   "E5",               "1"),
        ("tone", "PM_score",   "E6",               "1"),
        ("tone", "Comp_score", "composite",        "1"),
        ("tone", "MA_score",   "composite",        "1"),
        ("tone", "MP_score",   "composite",        "1"),
        ("tone", "PP1",        "PM_score",         "1"),
        ("tone", "PP2",        "PM_score",         "1"),
        ("tone", "PP3",        "PM_score",         "1"),
        ("tone", "PP4",        "PM_score",         "1"),
        ("tone", "PP5",        "PM_score",         "1"),

        # Series 2 — AI Perceptions as IV
        ("PM_score",   "E2",       "composite",   "2"),
        ("PM_score",   "emotions_mean", "composite", "2"),
        ("Ind_score",  "PM_score", "composite",   "2"),
        ("Ind_score",  "MA_score", "composite",   "2"),
        ("Ind_score",  "MP_score", "composite",   "2"),
        ("Comp_score", "MA_score", "E3",          "2"),
        ("Comp_score", "MA_score", "E4",          "2"),
        ("Comp_score", "MA_score", "E5",          "2"),
        ("Comp_score", "MA_score", "E6",          "2"),
        ("Comp_score", "MP_score", "E3",          "2"),
        ("Comp_score", "MP_score", "E4",          "2"),
        ("Comp_score", "MP_score", "E5",          "2"),
        ("Comp_score", "MP_score", "E6",          "2"),
        ("MA_score",   "PM_score", "composite",   "2"),
        ("MP_score",   "PM_score", "composite",   "2"),
    ]

    # Chain mediation — Tone → Ind_score → PM_score → composite
    chain_models = [
        ("tone", ["Ind_score", "PM_score"], "composite", "1"),
    ]

    rows = []

    # ------------------------------------------------------------------
    # Simple mediations
    # ------------------------------------------------------------------
    for iv, med, dv, series in models:
        row = _run_simple_mediation(df, iv, med, dv, series, Process)
        if row:
            rows.append(row)

    # ------------------------------------------------------------------
    # Chain mediations (serial)
    # ------------------------------------------------------------------
    for iv, mediators, dv, series in chain_models:
        row = _run_chain_mediation(df, iv, mediators, dv, series, Process)
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # FDR correction on indirect effects
    p_vals = []
    for _, r in result.iterrows():
        ci_low  = r.get("CI_low", np.nan)
        ci_high = r.get("CI_high", np.nan)
        # Derive approximate p from CI: significant if CI excludes 0
        if pd.notna(ci_low) and pd.notna(ci_high):
            excludes_zero = (ci_low > 0) or (ci_high < 0)
            p_vals.append(0.01 if excludes_zero else 0.50)
        else:
            p_vals.append(np.nan)

    valid_mask = [not np.isnan(p) for p in p_vals]
    valid_ps   = [p for p, m in zip(p_vals, valid_mask) if m]

    if valid_ps:
        corrected = _fdr_correct(valid_ps)
        corrected_iter = iter(corrected)
        result["p_fdr"] = [
            round(next(corrected_iter), 4) if m else np.nan
            for m in valid_mask
        ]
    else:
        result["p_fdr"] = np.nan

    result["Sig."] = result["p_fdr"].apply(
        lambda p: _sig_label(p) if pd.notna(p) else "ns"
    )

    log.info(f"Sheet F: {len(result)} mediation models run.")
    return result


def _run_simple_mediation(
    df: pd.DataFrame,
    iv: str, med: str, dv: str,
    series: str,
    Process,
) -> dict | None:
    """Run one simple mediation model and return a result row."""
    needed = [iv, med, dv]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(f"Mediation {iv}→{med}→{dv}: missing {missing} — skipped.")
        return None

    clean = df[needed].dropna()
    if len(clean) < 20:
        log.warning(
            f"Mediation {iv}→{med}→{dv}: only {len(clean)} rows — skipped."
        )
        return None

    try:
        p = Process(
            data=clean,
            model=4,           # simple mediation
            x=iv, y=dv, m=med,
            boot=config.BOOTSTRAP_ITERATIONS,
            seed=42,
        )
        results = p.results

        indirect = results["IndirectEffects"]["Indirect"]["Estimate"].iloc[0]
        ci_low   = results["IndirectEffects"]["Indirect"]["LLCI"].iloc[0]
        ci_high  = results["IndirectEffects"]["Indirect"]["ULCI"].iloc[0]
        a_path   = results["AntecedentEffects"]["a"]["Estimate"].iloc[0]
        b_path   = results["AntecedentEffects"]["b"]["Estimate"].iloc[0]
        c_path   = results["TotalEffect"]["c"]["Estimate"].iloc[0]
        c_prime  = results["DirectEffect"]["c_prime"]["Estimate"].iloc[0]

        # Correlations for context
        r_iv_m  = round(clean[iv].corr(clean[med]), 3)
        r_iv_dv = round(clean[iv].corr(clean[dv]),  3)
        r_m_dv  = round(clean[med].corr(clean[dv]), 3)

        return {
            "Series":    series,
            "IV":        iv,
            "Mediator":  med,
            "DV":        dv,
            "n":         len(clean),
            "a":         round(a_path,   3),
            "b":         round(b_path,   3),
            "c":         round(c_path,   3),
            "c_prime":   round(c_prime,  3),
            "Indirect":  round(indirect, 3),
            "CI_low":    round(ci_low,   3),
            "CI_high":   round(ci_high,  3),
            "Type":      "simple",
            "r_IV_M":    r_iv_m,
            "r_IV_DV":   r_iv_dv,
            "r_M_DV":    r_m_dv,
        }

    except Exception as e:
        log.error(f"Mediation {iv}→{med}→{dv}: failed — {e}")
        return None


def _run_chain_mediation(
    df: pd.DataFrame,
    iv: str,
    mediators: list,
    dv: str,
    series: str,
    Process,
) -> dict | None:
    """Run one serial (chain) mediation model and return a result row."""
    needed = [iv] + mediators + [dv]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(
            f"Chain mediation {iv}→{'→'.join(mediators)}→{dv}: "
            f"missing {missing} — skipped."
        )
        return None

    clean = df[needed].dropna()
    if len(clean) < 20:
        log.warning(
            f"Chain mediation: only {len(clean)} rows — skipped."
        )
        return None

    try:
        p = Process(
            data=clean,
            model=6,           # serial mediation
            x=iv, y=dv, m=mediators,
            boot=config.BOOTSTRAP_ITERATIONS,
            seed=42,
        )
        results  = p.results
        indirect = results["IndirectEffects"]["Indirect"]["Estimate"].iloc[0]
        ci_low   = results["IndirectEffects"]["Indirect"]["LLCI"].iloc[0]
        ci_high  = results["IndirectEffects"]["Indirect"]["ULCI"].iloc[0]

        return {
            "Series":   series,
            "IV":       iv,
            "Mediator": " → ".join(mediators),
            "DV":       dv,
            "n":        len(clean),
            "a":        np.nan,
            "b":        np.nan,
            "c":        np.nan,
            "c_prime":  np.nan,
            "Indirect": round(indirect, 3),
            "CI_low":   round(ci_low,   3),
            "CI_high":  round(ci_high,  3),
            "Type":     "serial chain",
            "r_IV_M":   np.nan,
            "r_IV_DV":  round(clean[iv].corr(clean[dv]), 3),
            "r_M_DV":   np.nan,
        }

    except Exception as e:
        log.error(
            f"Chain mediation {iv}→{'→'.join(mediators)}→{dv}: failed — {e}"
        )
        return None


# ---------------------------------------------------------------------------
# Sheet I — Demographics & robustness checks
# ---------------------------------------------------------------------------
def run_sheet_i(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demographics descriptives + robustness checks.

    1. Descriptive statistics: age, gender, language, tone balance
    2. ANCOVA: tone effect on main DVs controlling for age, gender, language
    3. Interaction tests: tone x language, tone x gender

    Returns a single DataFrame with all results labelled by section.
    """
    rows = []

    # ------------------------------------------------------------------
    # 1. Descriptive statistics
    # ------------------------------------------------------------------
    rows.append({"Section": "DEMOGRAPHICS", "Test": "", "Variable": "",
                 "Result": "", "p": "", "Sig.": ""})

    # Age
    if "age" in df.columns:
        age = df["age"].dropna()
        rows.append({
            "Section":  "Descriptives",
            "Test":     "Age",
            "Variable": "age",
            "Result":   f"M={age.mean():.1f}, SD={age.std():.1f}, "
                        f"range={int(age.min())}–{int(age.max())}",
            "p": "", "Sig.": "",
        })

    # Gender
    if "gender" in df.columns:
        gender_counts = df["gender"].value_counts()
        rows.append({
            "Section":  "Descriptives",
            "Test":     "Gender",
            "Variable": "gender",
            "Result":   str(gender_counts.to_dict()),
            "p": "", "Sig.": "",
        })

    # Language
    if "language" in df.columns:
        lang_counts = df["language"].value_counts()
        rows.append({
            "Section":  "Descriptives",
            "Test":     "Language",
            "Variable": "language",
            "Result":   str(lang_counts.to_dict()),
            "p": "", "Sig.": "",
        })

    # Tone balance
    tone_counts = df["tone"].value_counts()
    rows.append({
        "Section":  "Descriptives",
        "Test":     "Tone condition balance",
        "Variable": "tone",
        "Result":   f"Friendly n={tone_counts.get(1,0)}, "
                    f"Professional n={tone_counts.get(0,0)}",
        "p": "", "Sig.": "",
    })

    # ------------------------------------------------------------------
    # 2. ANCOVA — tone effect controlling for demographics
    # ------------------------------------------------------------------
    main_dvs = ["composite", "engagement_score", "E3", "E4", "E5", "E6",
                "PM_score", "Comp_score", "Ind_score", "MA_score", "MP_score"]

    covariates = [c for c in ["age", "gender", "language"] if c in df.columns]

    rows.append({"Section": "ANCOVA", "Test": "", "Variable": "",
                 "Result": f"Covariates: {covariates}", "p": "", "Sig.": ""})

    for dv in main_dvs:
        if dv not in df.columns:
            continue
        cols = [dv, "tone"] + covariates
        clean = df[cols].dropna()
        if len(clean) < 20:
            continue
        try:
            formula = f"{dv} ~ tone + {' + '.join(covariates)}"
            model   = ols(formula, data=clean).fit()
            aov     = anova_lm(model, typ=2)
            tone_p  = aov.loc["tone", "PR(>F)"]
            tone_f  = aov.loc["tone", "F"]
            rows.append({
                "Section":  "ANCOVA",
                "Test":     f"Tone → {dv} (controlling demographics)",
                "Variable": dv,
                "Result":   f"F={tone_f:.3f}",
                "p":        round(tone_p, 4),
                "Sig.":     _sig_label(tone_p),
            })
        except Exception as e:
            log.warning(f"ANCOVA {dv}: {e}")

    # ------------------------------------------------------------------
    # 3. Interaction tests: tone x language, tone x gender
    # ------------------------------------------------------------------
    interactions = [
        ("language", "tone * language"),
        ("gender",   "tone * gender"),
    ]

    for demo_var, interaction_term in interactions:
        if demo_var not in df.columns:
            continue

        rows.append({
            "Section": f"Interaction: tone × {demo_var}",
            "Test": "", "Variable": "", "Result": "", "p": "", "Sig.": ""
        })

        for dv in ["composite", "E3", "PM_score"]:
            if dv not in df.columns:
                continue
            cols  = [dv, "tone", demo_var]
            clean = df[cols].dropna()
            if len(clean) < 20:
                continue
            try:
                formula = f"{dv} ~ {interaction_term}"
                model   = ols(formula, data=clean).fit()
                aov     = anova_lm(model, typ=2)
                int_key = [k for k in aov.index if ":" in k]
                if int_key:
                    int_p = aov.loc[int_key[0], "PR(>F)"]
                    int_f = aov.loc[int_key[0], "F"]
                    rows.append({
                        "Section":  f"Interaction: tone × {demo_var}",
                        "Test":     f"tone × {demo_var} → {dv}",
                        "Variable": dv,
                        "Result":   f"F={int_f:.3f}",
                        "p":        round(int_p, 4),
                        "Sig.":     _sig_label(int_p),
                    })
            except Exception as e:
                log.warning(f"Interaction {demo_var} × tone → {dv}: {e}")

    result = pd.DataFrame(rows)
    log.info(f"Sheet I: {len(result)} rows generated.")
    return result
