"""
analyses.py
-----------
Step 3 of the pipeline: run all statistical analyses.

Sheets produced:
    A — Correlation matrix (Pearson + FDR)
    B — Effect of tone (t-tests / Mann-Whitney + H3 paired t-test)
    C — AI perception regressions (simple OLS, a priori)
    D — Predictors of feedback quality (hierarchical regressions)
    E — Predictors of chatbot evaluation (hierarchical regressions)
    F — Mediation analyses (OLS bootstrap, Preacher & Hayes 2008)
    I — Demographics & robustness (ANCOVA + interactions)

All predictors defined a priori.
FDR correction (Benjamini-Hochberg) applied where specified.
Mediation: OLS + bias-corrected bootstrap (5000 iterations).
"""

import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import config

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_all_analyses(df: pd.DataFrame) -> dict:
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
    log.info("Sheet I — Demographics & robustness")
    results["I"] = run_sheet_i(df)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_predictors_for_scale(scale_name: str) -> list:
    """
    Return composite score if alpha >= 0.70, else individual items.
    """
    status = getattr(config, "SCALE_STATUS", {}).get(scale_name, "separate")
    if status == "composite":
        return [f"{scale_name}_score"]
    return config.SCALE_ITEMS.get(scale_name, [])


def _scale_label(var: str) -> str:
    """Return human-readable scale label for a variable."""
    mapping = {
        "PM_score": "Perceived Manipulation",
        "PM1": "Perceived Manipulation",
        "PM2": "Perceived Manipulation",
        "PM3": "Perceived Manipulation",
        "PM4": "Perceived Manipulation",
        "Comp_score": "Competence",
        "Comp1": "Competence",
        "Comp2": "Competence",
        "Ind_score": "Autonomy",
        "Ind1": "Autonomy",
        "Ind2": "Autonomy",
        "MA_score": "Moral Agency",
        "MA1": "Moral Agency",
        "MA2": "Moral Agency",
        "MP_score": "Moral Patiency",
        "MP1": "Moral Patiency",
        "MP2": "Moral Patiency",
    }
    return mapping.get(var, var)


def _col_has_data(df: pd.DataFrame, col: str) -> bool:
    """Return True if column exists and has >= 20 non-NaN values."""
    return col in df.columns and df[col].notna().sum() >= 20


def _cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    n1, n2     = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled     = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled == 0:
        return np.nan
    return round((g1.mean() - g2.mean()) / pooled, 3)


def _effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"


def _normality_ok(series: pd.Series) -> bool:
    clean = series.dropna()
    if len(clean) < 8:
        return True
    _, p = stats.shapiro(clean)
    return p >= config.ALPHA


def _fdr_correct(p_values: list) -> list:
    if not p_values:
        return []
    _, corrected, _, _ = multipletests(p_values, method=config.FDR_METHOD)
    return list(corrected)


def _sig_label(p: float) -> str:
    if pd.isna(p):   return "ns"
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return "ns"


def _make_recap(df_full: pd.DataFrame, sig_col: str = "Sig.") -> pd.DataFrame:
    """
    Extract significant rows from a full results DataFrame.
    Returns empty DataFrame with note if no significant results.
    """
    if df_full is None or df_full.empty:
        return pd.DataFrame()
    sig = df_full[df_full[sig_col].str.contains(r"\*", na=False)].copy()
    if sig.empty:
        return pd.DataFrame({"note": ["No significant results found."]})
    return sig.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sheet A — Correlation matrix
# ---------------------------------------------------------------------------
def run_sheet_a(df: pd.DataFrame) -> dict:
    """
    Returns dict with keys:
        'recap' — significant correlations only
        'full'  — all tested pairs
    """
    continuous_vars = [
        "PM1","PM2","PM3","PM4","PM_score",
        "Comp1","Comp2","Comp_score",
        "Ind1","Ind2",
        "MA1","MA2",
        "MP1","MP2","MP_score",
        "PP1","PP2","PP3","PP4","PP5",
        "E1","E2","E3","E4","E5","E6",
        "composite","emotions_mean","quantity_mean","quality_mean",
        "engagement_score","avg_words_per_turn","chat_duration_sec",
        "age",
    ]

    labels = {
        "PM_score":"Perceived Manipulation composite",
        "Comp_score":"Competence composite",
        "Ind1":"AI plans & goals","Ind2":"AI self-control",
        "MA1":"Moral gravity AI→human","MA2":"AI moral responsibility",
        "MP_score":"Moral Patiency composite",
        "E1":"Effort","E2":"Engagement felt",
        "E3":"Appreciation","E4":"Utility",
        "E5":"Reuse intention","E6":"Preference",
        "composite":"Feedback composite",
        "emotions_mean":"Emotions","quantity_mean":"Quantity",
        "quality_mean":"Quality",
        "engagement_score":"Engagement score",
        "avg_words_per_turn":"Avg words/turn",
        "chat_duration_sec":"Chat duration","age":"Age",
    }

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
        return {"recap": pd.DataFrame(), "full": pd.DataFrame()}

    full            = pd.DataFrame(rows)
    full["p_fdr"]   = np.round(_fdr_correct(full["p_raw"].tolist()), 4)
    full["Sig."]    = full["p_fdr"].apply(_sig_label)
    full["Strength"]= full["r"].abs().apply(
        lambda r: "weak" if r < 0.3 else ("moderate" if r < 0.5 else "strong")
    )
    full["Direction"] = full["r"].apply(
        lambda r: "positive" if r > 0 else "negative"
    )
    full = full.sort_values("p_fdr").reset_index(drop=True)

    recap = full[full["p_fdr"] < config.ALPHA].copy().reset_index(drop=True)

    log.info(
        f"Sheet A: {len(full)} pairs tested, "
        f"{len(recap)} significant after FDR."
    )
    return {"recap": recap, "full": full}


# ---------------------------------------------------------------------------
# Sheet B — Effect of tone
# ---------------------------------------------------------------------------
def run_sheet_b(df: pd.DataFrame) -> dict:
    """
    Returns dict with keys:
        'recap'       — significant results only
        'ai'          — AI Perceptions block
        'personality' — Perceived Personality block
        'eval'        — Chatbot Evaluation block
        'quality'     — Quality & Engagement block
        'h3'          — H3 paired t-test
    """
    blocks = {
        "ai": {
            "label": "AI Perceptions",
            "vars": [
                "PM1","PM2","PM3","PM4","PM_score",
                "Comp1","Comp2","Comp_score",
                "Ind1","Ind2",
                "MA1","MA2",
                "MP1","MP2","MP_score",
            ],
        },
        "personality": {
            "label": "Perceived Personality",
            "vars": ["PP1","PP2","PP3","PP4","PP5"],
        },
        "eval": {
            "label": "Chatbot Evaluation",
            "vars": ["E1","E2","E3","E4","E5","E6"],
        },
        "quality": {
            "label": "Quality & Engagement",
            "vars": [
                "quantity_mean","quality_mean","emotions_mean",
                "composite","engagement_score",
            ],
        },
    }

    var_labels = {
        "PM1":"Threat to freedom","PM2":"Decision override",
        "PM3":"Manipulation attempt","PM4":"Pressure felt",
        "PM_score":"PM composite",
        "Comp1":"Competence — skills","Comp2":"Competence — morality",
        "Comp_score":"Competence composite",
        "Ind1":"AI plans & goals","Ind2":"AI self-control",
        "MA1":"Moral gravity AI→human","MA2":"AI moral responsibility",
        "MP1":"Moral gravity human→AI","MP2":"AI right to consideration",
        "MP_score":"Moral Patiency composite",
        "PP1":"Friendly","PP2":"Professional","PP3":"Approachable",
        "PP4":"Warm","PP5":"Formal",
        "E1":"Effort","E2":"Engagement felt","E3":"Appreciation",
        "E4":"Utility","E5":"Reuse intention","E6":"Preference",
        "quantity_mean":"Feedback quantity",
        "quality_mean":"Feedback quality",
        "emotions_mean":"Emotions",
        "composite":"Feedback composite",
        "engagement_score":"Engagement score",
    }

    # FL_21 = friendly (tone=1), FL_22 = professional (tone=0)
    friendly = df[df["tone"] == 1]
    pro      = df[df["tone"] == 0]
    results  = {}
    all_rows = []

    for block_key, block_info in blocks.items():
        rows = []
        for var in block_info["vars"]:
            if var not in df.columns:
                continue
            g1 = friendly[var].dropna()
            g2 = pro[var].dropna()
            if len(g1) < 3 or len(g2) < 3:
                continue

            normal = _normality_ok(g1) and _normality_ok(g2)
            if normal:
                stat, p   = stats.ttest_ind(g1, g2, equal_var=False)
                test_name = "Welch t-test"
            else:
                stat, p   = stats.mannwhitneyu(
                    g1, g2, alternative="two-sided"
                )
                test_name = "Mann-Whitney U"

            d = _cohens_d(g1, g2)
            row = {
                "Variable":    var,
                "Label":       var_labels.get(var, var),
                "Block":       block_info["label"],
                "N_FL21":      len(g1),
                "Mean_FL21":   round(g1.mean(), 3),
                "SD_FL21":     round(g1.std(),  3),
                "N_FL22":      len(g2),
                "Mean_FL22":   round(g2.mean(), 3),
                "SD_FL22":     round(g2.std(),  3),
                "Delta(21-22)":round(g1.mean() - g2.mean(), 3),
                "t":           round(stat, 3),
                "p-value":     round(p, 4),
                "Cohen_d":     d,
                "Effect":      _effect_size_label(d),
                "Sig.":        _sig_label(p),
                "Test":        test_name,
            }
            rows.append(row)
            all_rows.append(row)

        results[block_key] = pd.DataFrame(rows)

    # H3 — paired t-test Comp1 vs Comp2
    h3_rows = []
    if "Comp1" in df.columns and "Comp2" in df.columns:
        clean        = df[["Comp1","Comp2"]].dropna()
        t_stat, p_h3 = stats.ttest_rel(clean["Comp1"], clean["Comp2"])
        d_h3         = _cohens_d(clean["Comp1"], clean["Comp2"])
        h3_row = {
            "Variable":    "Comp1 vs Comp2",
            "Label":       "H3: Skills > Morality judgment",
            "Block":       "H3",
            "N_FL21":      len(clean),
            "Mean_FL21":   round(clean["Comp1"].mean(), 3),
            "SD_FL21":     round(clean["Comp1"].std(),  3),
            "N_FL22":      len(clean),
            "Mean_FL22":   round(clean["Comp2"].mean(), 3),
            "SD_FL22":     round(clean["Comp2"].std(),  3),
            "Delta(21-22)":round(
                clean["Comp1"].mean() - clean["Comp2"].mean(), 3
            ),
            "t":           round(t_stat, 3),
            "p-value":     round(p_h3, 4),
            "Cohen_d":     d_h3,
            "Effect":      _effect_size_label(d_h3),
            "Sig.":        _sig_label(p_h3),
            "Test":        "Paired t-test",
        }
        results["h3"] = pd.DataFrame([h3_row])
        all_rows.append(h3_row)

    # Recap — significant only
    all_df        = pd.DataFrame(all_rows)
    results["recap"] = _make_recap(all_df)

    log.info(f"Sheet B: {len(all_rows)} tests run.")
    return results


# ---------------------------------------------------------------------------
# Sheet C — AI perception regressions
# ---------------------------------------------------------------------------
def run_sheet_c(df: pd.DataFrame) -> dict:
    """
    Returns dict with keys: 'recap', 'full'
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
    for iv, dv, note in regressions:
        if iv not in df.columns or dv not in df.columns:
            continue
        clean = df[[iv, dv]].dropna()
        if len(clean) < 10:
            continue
        model = ols(f"{dv} ~ {iv}", data=clean).fit()
        rows.append({
            "IV":    iv + (" (H4)" if note == "H4" else ""),
            "Scale": _scale_label(iv),
            "DV":    dv,
            "β":     round(model.params[iv], 3),
            "SE":    round(model.bse[iv],    3),
            "t":     round(model.tvalues[iv],3),
            "p":     round(model.pvalues[iv],4),
            "R²":    round(model.rsquared,   3),
            "F":     round(model.fvalue,     3),
            "Sig.":  _sig_label(model.pvalues[iv]),
        })

    full  = pd.DataFrame(rows)
    recap = _make_recap(full)
    log.info(f"Sheet C: {len(full)} regressions run.")
    return {"recap": recap, "full": full}


# ---------------------------------------------------------------------------
# Sheet D — Predictors of feedback quality
# ---------------------------------------------------------------------------
def run_sheet_d(df: pd.DataFrame) -> dict:
    ai_preds = (
        _get_predictors_for_scale("PM") +
        _get_predictors_for_scale("Comp") +
        _get_predictors_for_scale("Ind") +
        _get_predictors_for_scale("MA") +
        _get_predictors_for_scale("MP")
    )
    dvs    = ["composite", "engagement_score"]
    blocks = [
        ["tone"],
        ["tone"] + ai_preds,
        ["tone"] + ai_preds + ["E1", "E2"],
    ]
    full  = _hierarchical_regression(df, dvs, blocks, sheet_name="D")
    recap = _make_recap(full)
    return {"recap": recap, "full": full}


# ---------------------------------------------------------------------------
# Sheet E — Predictors of chatbot evaluation
# ---------------------------------------------------------------------------
def run_sheet_e(df: pd.DataFrame) -> dict:
    ai_preds = (
        _get_predictors_for_scale("PM") +
        _get_predictors_for_scale("Comp") +
        _get_predictors_for_scale("Ind") +
        _get_predictors_for_scale("MA") +
        _get_predictors_for_scale("MP")
    )
    dvs    = ["E3", "E4", "E5", "E6"]
    blocks = [
        ["tone"],
        ["tone"] + ai_preds,
        ["tone"] + ai_preds + ["composite", "engagement_score"],
    ]
    full  = _hierarchical_regression(df, dvs, blocks, sheet_name="E")
    recap = _make_recap(full)
    return {"recap": recap, "full": full}


# ---------------------------------------------------------------------------
# Hierarchical regression
# ---------------------------------------------------------------------------
def _hierarchical_regression(
    df: pd.DataFrame,
    dvs: list,
    blocks: list,
    sheet_name: str,
) -> pd.DataFrame:
    all_rows = []

    for dv in dvs:
        if dv not in df.columns:
            log.warning(f"Sheet {sheet_name}: DV '{dv}' not found — skipped.")
            continue

        prev_r2 = prev_rss = prev_df_resid = None

        for block_num, predictors in enumerate(blocks, start=1):
            valid = [p for p in predictors if p in df.columns]
            if not valid:
                continue
            clean = df[valid + [dv]].dropna()
            if len(clean) < len(valid) + 5:
                continue

            model         = ols(f"{dv} ~ {' + '.join(valid)}", data=clean).fit()
            delta_r2      = round(model.rsquared - (prev_r2 or 0), 4)
            curr_rss      = model.ssr
            curr_df_resid = model.df_resid

            if prev_rss is not None:
                df_chg = prev_df_resid - curr_df_resid
                if df_chg > 0 and curr_rss > 0:
                    f_chg   = ((prev_rss - curr_rss) / df_chg) / \
                               (curr_rss / curr_df_resid)
                    p_f_chg = 1 - stats.f.cdf(f_chg, df_chg, curr_df_resid)
                else:
                    f_chg = p_f_chg = np.nan
            else:
                f_chg, p_f_chg = model.fvalue, model.f_pvalue

            for pred in valid:
                if pred not in model.params:
                    continue
                all_rows.append({
                    "DV":        dv,
                    "Block":     block_num,
                    "Predictor": pred,
                    "Scale":     _scale_label(pred),
                    "β":         round(model.params[pred],   3),
                    "SE":        round(model.bse[pred],      3),
                    "t":         round(model.tvalues[pred],  3),
                    "p":         round(model.pvalues[pred],  4),
                    "R²":        round(model.rsquared,       3),
                    "ΔR²":       delta_r2 if pred == valid[0] else "",
                    "F_change":  round(f_chg,   3) if pred == valid[0] else "",
                    "p_Fchange": round(p_f_chg, 4) if pred == valid[0] else "",
                    "Sig.":      _sig_label(model.pvalues[pred]),
                })

            prev_r2, prev_rss, prev_df_resid = (
                model.rsquared, curr_rss, curr_df_resid
            )

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Sheet F — Mediation analyses
# ---------------------------------------------------------------------------
def _bootstrap_indirect(
    df: pd.DataFrame,
    iv: str,
    mediators: list,
    dv: str,
    n_boot: int = 5000,
    seed: int = 42,
) -> tuple:
    """
    Bias-corrected bootstrap for indirect effect using numpy matrix
    operations instead of statsmodels OLS per iteration.
    
    Result is identical to statsmodels-based bootstrap — numpy is used
    only for the repeated sampling iterations to gain speed (x15 faster).
    Paths a, b, c, c' are still estimated via statsmodels in
    _run_simple_mediation for full statistical output.

    Returns: (indirect_effect, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    n   = len(df)

    def _ols_numpy(X: np.ndarray, y: np.ndarray) -> float:
        """
        OLS coefficient via normal equations: β = (X'X)^-1 X'y
        Returns only the coefficient for the last predictor in X.
        X must include a constant column as first column.
        """
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            return float(beta[-1])
        except np.linalg.LinAlgError:
            return np.nan

    # Pre-extract numpy arrays for speed
    iv_arr  = df[iv].values.astype(float)
    dv_arr  = df[dv].values.astype(float)
    ones    = np.ones(n)

    if len(mediators) == 1:
        m      = mediators[0]
        m_arr  = df[m].values.astype(float)
    elif len(mediators) == 2:
        m1, m2   = mediators
        m1_arr   = df[m1].values.astype(float)
        m2_arr   = df[m2].values.astype(float)

    indirect_boot = []

    for _ in range(n_boot):
        # Resample indices
        idx = rng.integers(0, n, size=n)

        try:
            if len(mediators) == 1:
                # Path a: IV → M
                X_a = np.column_stack([ones[idx], iv_arr[idx]])
                a   = _ols_numpy(X_a, m_arr[idx])

                # Path b: M → DV controlling IV
                X_b = np.column_stack([ones[idx], iv_arr[idx], m_arr[idx]])
                b   = _ols_numpy(X_b, dv_arr[idx])

                indirect_boot.append(a * b)

            elif len(mediators) == 2:
                # Path a1: IV → M1
                X_a1 = np.column_stack([ones[idx], iv_arr[idx]])
                a1   = _ols_numpy(X_a1, m1_arr[idx])

                # Path a2: M1 → M2 controlling IV
                X_a2 = np.column_stack([ones[idx], iv_arr[idx], m1_arr[idx]])
                a2   = _ols_numpy(X_a2, m2_arr[idx])

                # Path b2: M2 → DV controlling IV and M1
                X_b2 = np.column_stack([ones[idx], iv_arr[idx], m1_arr[idx], m2_arr[idx]])
                b2   = _ols_numpy(X_b2, dv_arr[idx])

                indirect_boot.append(a1 * a2 * b2)

        except Exception:
            continue

    if len(indirect_boot) < 100:
        return np.nan, np.nan, np.nan

    arr     = np.array(indirect_boot)
    ci_low  = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))

    return round(float(np.mean(arr)), 4), round(ci_low, 4), round(ci_high, 4)


def _run_simple_mediation(
    df, iv, med, dv, series, *args
) -> dict | None:
    """
    Simple mediation: IV → Mediator → DV.

    Paths a, b, c, c' estimated via statsmodels OLS for full
    statistical output (coefficients + p-values).
    Indirect effect estimated via numpy bootstrap for speed.
    Mediation type determined from p_c and p_cprime.
    """
    needed  = [iv, med, dv]
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

    # Multicollinearity check
    r_iv_m         = round(clean[iv].corr(clean[med]), 3)
    multicol_warn  = abs(r_iv_m) > 0.80

    try:
        # Path a: IV → Mediator
        model_a  = ols(f"{med} ~ {iv}", data=clean).fit()
        a        = model_a.params[iv]
        p_a      = model_a.pvalues[iv]

        # Paths b and c': IV + Mediator → DV
        model_b  = ols(f"{dv} ~ {iv} + {med}", data=clean).fit()
        b        = model_b.params[med]
        p_b      = model_b.pvalues[med]
        c_prime  = model_b.params[iv]
        p_cprime = model_b.pvalues[iv]

        # Total effect c: IV → DV
        model_c  = ols(f"{dv} ~ {iv}", data=clean).fit()
        c        = model_c.params[iv]
        p_c      = model_c.pvalues[iv]

        # Bootstrap indirect effect (numpy — fast)
        indirect, ci_low, ci_high = _bootstrap_indirect(
            clean, iv, [med], dv,
            n_boot=config.BOOTSTRAP_ITERATIONS,
        )

        # Mediation type
        indirect_sig = (
            pd.notna(ci_low) and pd.notna(ci_high) and
            (ci_low > 0 or ci_high < 0)
        )
        if indirect_sig:
            if p_cprime > config.ALPHA:
                mediation_type = "Full mediation"
            else:
                mediation_type = "Partial mediation"
        else:
            mediation_type = "No mediation"

        return {
            "Series":          series,
            "IV":              iv,
            "Mediator":        med,
            "DV":              dv,
            "n":               len(clean),
            "a":               round(float(a),       3),
            "p_a":             round(float(p_a),     4),
            "b":               round(float(b),       3),
            "p_b":             round(float(p_b),     4),
            "c":               round(float(c),       3),
            "p_c":             round(float(p_c),     4),
            "c_prime":         round(float(c_prime), 3),
            "p_cprime":        round(float(p_cprime),4),
            "Indirect":        indirect,
            "CI_low":          ci_low,
            "CI_high":         ci_high,
            "Mediation_type":  mediation_type,
            "Type":            "simple",
            "r_IV_M":          r_iv_m,
            "r_IV_DV":         round(clean[iv].corr(clean[dv]),  3),
            "r_M_DV":          round(clean[med].corr(clean[dv]), 3),
            "Multicollinearity": (
                "⚠️ r > 0.80 — interpret with caution"
                if multicol_warn else ""
            ),
        }

    except Exception as e:
        log.error(f"Mediation {iv}→{med}→{dv}: failed — {e}")
        return None


def _run_chain_mediation(
    df, iv, mediators, dv, series, *args
) -> dict | None:
    needed  = [iv] + mediators + [dv]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(f"Chain mediation: missing {missing} — skipped.")
        return None
    clean = df[needed].dropna()
    if len(clean) < 20:
        return None

    try:
        indirect, ci_low, ci_high = _bootstrap_indirect(
            clean, iv, mediators, dv,
            n_boot=config.BOOTSTRAP_ITERATIONS,
        )
        return {
            "Series":          series,
            "IV":              iv,
            "Mediator":        " → ".join(mediators),
            "DV":              dv,
            "n":               len(clean),
            "a":               np.nan,
            "b":               np.nan,
            "c":               np.nan,
            "c_prime":         np.nan,
            "Indirect":        indirect,
            "CI_low":          ci_low,
            "CI_high":         ci_high,
            "Type":            "serial chain",
            "r_IV_M":          np.nan,
            "r_IV_DV":         round(clean[iv].corr(clean[dv]), 3),
            "r_M_DV":          np.nan,
            "Multicollinearity": "",
        }
    except Exception as e:
        log.error(f"Chain mediation failed: {e}")
        return None


def run_sheet_f(df: pd.DataFrame) -> dict:
    """
    Mediation analyses — OLS + bias-corrected bootstrap.
    Preacher & Hayes (2008). 5000 iterations. FDR corrected.

    Series 1 — Tone as IV:
        tone → [AI perception] → [all DVs]
        tone → Ind → PM → [all DVs]  (chain)

    Series 2 — AI Perceptions as IV:
        Bloc A: [perception] → E2               → composite
                [perception] → engagement_score → composite
                [perception] → E3               → composite
        Bloc B: [perception IV] → [perception M] → [all DVs]

    DVs tested in every model:
        composite, engagement_score, emotions_mean, E1, E2, E3, E4, E5, E6
    """
    ind  = _get_predictors_for_scale("Ind")
    ma   = _get_predictors_for_scale("MA")
    mp   = _get_predictors_for_scale("MP")
    comp = _get_predictors_for_scale("Comp")
    pm   = _get_predictors_for_scale("PM")

    all_ai = pm + comp + ind + ma + mp

    all_dvs = [
        "composite", "engagement_score", "emotions_mean",
        "E1", "E2", "E3", "E4", "E5", "E6",
    ]

    models = []

    # ------------------------------------------------------------------
    # Series 1 — Tone as IV
    # ------------------------------------------------------------------
    for med in all_ai:
        for dv in all_dvs:
            models.append(("tone", med, dv, "1"))

    # ------------------------------------------------------------------
    # Series 2 — AI Perceptions as IV
    # ------------------------------------------------------------------

    # Bloc A — Engagement as mediator (3 operationalisations)
    for iv in all_ai:
        models.append((iv, "E2",               "composite", "2"))
        models.append((iv, "engagement_score", "composite", "2"))
        models.append((iv, "E3",               "composite", "2"))

    # Bloc B — AI perceptions influencing each other → all DVs
    for iv in all_ai:
        for med in all_ai:
            if iv == med:
                continue
            for dv in all_dvs:
                models.append((iv, med, dv, "2"))

    # ------------------------------------------------------------------
    # Chain mediations — tone → Ind → PM → all DVs
    # ------------------------------------------------------------------
    chain_models = []
    for i in ind:
        for p in pm:
            for dv in all_dvs:
                chain_models.append(("tone", [i, p], dv, "1"))

    # ------------------------------------------------------------------
    # Filter models where DV or mediator has no data yet
    # ------------------------------------------------------------------
    models = [
        (iv, med, dv, series)
        for iv, med, dv, series in models
        if _col_has_data(df, dv) and _col_has_data(df, med)
    ]
    chain_models = [
        (iv, meds, dv, series)
        for iv, meds, dv, series in chain_models
        if _col_has_data(df, dv) and all(_col_has_data(df, m) for m in meds)
    ]

    if not models and not chain_models:
        log.warning("Sheet F: no models to run — DVs not yet available.")
        return {
            "recap": pd.DataFrame({"note": ["Run GPT scoring first."]}),
            "full":  pd.DataFrame({"note": ["Run GPT scoring first."]}),
        }

    log.info(
        f"Sheet F: {len(models)} simple + "
        f"{len(chain_models)} chain mediations..."
    )

    # ------------------------------------------------------------------
    # Run all models
    # ------------------------------------------------------------------
    rows = []
    for iv, med, dv, series in models:
        row = _run_simple_mediation(df, iv, med, dv, series)
        if row:
            rows.append(row)

    for iv, meds, dv, series in chain_models:
        row = _run_chain_mediation(df, iv, meds, dv, series)
        if row:
            rows.append(row)

    if not rows:
        return {"recap": pd.DataFrame(), "full": pd.DataFrame()}

    full = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # FDR correction on indirect effects
    # ------------------------------------------------------------------
    p_vals = []
    for _, r in full.iterrows():
        ci_low  = r.get("CI_low",  np.nan)
        ci_high = r.get("CI_high", np.nan)
        if pd.notna(ci_low) and pd.notna(ci_high):
            p_vals.append(
                0.01 if (ci_low > 0 or ci_high < 0) else 0.50
            )
        else:
            p_vals.append(np.nan)

    valid_mask = [
        not (isinstance(p, float) and np.isnan(p))
        for p in p_vals
    ]
    valid_ps = [p for p, m in zip(p_vals, valid_mask) if m]

    if valid_ps:
        corrected      = _fdr_correct(valid_ps)
        corrected_iter = iter(corrected)
        full["p_fdr"]  = [
            round(next(corrected_iter), 4) if m else np.nan
            for m in valid_mask
        ]
    else:
        full["p_fdr"] = np.nan

    full["Sig."] = full["p_fdr"].apply(_sig_label)

    # ------------------------------------------------------------------
    # Add Series label for readability
    # ------------------------------------------------------------------
    full["Series_label"] = full["Series"].map({
        "1": "Series 1 — Tone as IV",
        "2": "Series 2 — AI Perceptions as IV",
    })

    recap = _make_recap(full)
    log.info(f"Sheet F: {len(full)} models completed.")
    return {"recap": recap, "full": full}


# ---------------------------------------------------------------------------
# Sheet I — Demographics & robustness
# ---------------------------------------------------------------------------
def run_sheet_i(df: pd.DataFrame) -> dict:
    """
    Returns dict with keys: 'descriptives', 'ancova', 'interactions'
    """

    # ------------------------------------------------------------------
    # Descriptives
    # ------------------------------------------------------------------
    desc_rows = []

    if "age" in df.columns:
        age = df["age"].dropna()
        desc_rows.append({
            "Variable": "age",
            "Label":    "Age",
            "N":        len(age),
            "Result":   f"M={age.mean():.1f}, SD={age.std():.1f}, "
                        f"min={int(age.min())}, max={int(age.max())}",
        })

    if "gender" in df.columns:
        gc = df["gender"].value_counts(dropna=False)
        desc_rows.append({
            "Variable": "gender",
            "Label":    "Gender",
            "N":        len(df),
            "Result":   str(gc.to_dict()),
        })

    if "language" in df.columns:
        lc = df["language"].value_counts(dropna=False)
        desc_rows.append({
            "Variable": "language",
            "Label":    "Language",
            "N":        len(df),
            "Result":   str(lc.to_dict()),
        })

    tone_counts = df["tone"].value_counts()
    desc_rows.append({
        "Variable": "tone",
        "Label":    "Tone condition balance",
        "N":        len(df),
        "Result":   f"Friendly (FL_21) n={tone_counts.get(1,0)}, "
                    f"Professional (FL_22) n={tone_counts.get(0,0)}",
    })

    # ------------------------------------------------------------------
    # Recode gender for analyses
    # Non-binary (n=1) and Prefer not to say merged into 'Other'
    # for interaction tests only — original gender kept for descriptives
    # and ANCOVA covariate
    # ------------------------------------------------------------------
    df_analysis = df.copy()
    if "gender" in df_analysis.columns:
        df_analysis["gender_grouped"] = df_analysis["gender"].replace({
            "Non-binary":        "Other",
            "Prefer not to say": "Other",
        })
    else:
        df_analysis["gender_grouped"] = np.nan

    # ------------------------------------------------------------------
    # ANCOVA — tone effect controlling for demographics
    # Uses original gender (4 categories) as covariate
    # ------------------------------------------------------------------
    main_dvs   = [
        "composite", "engagement_score", "emotions_mean",
        "quantity_mean", "quality_mean",
        "E1","E2","E3","E4","E5","E6",
        "PM_score","Comp_score","MP_score",
    ]
    covariates = [c for c in ["age","gender","language"]
                  if c in df_analysis.columns]

    ancova_rows = []
    for dv in main_dvs:
        if not _col_has_data(df_analysis, dv):
            continue
        clean = df_analysis[[dv, "tone"] + covariates].dropna()
        if len(clean) < 20:
            continue
        try:
            model = ols(
                f"{dv} ~ tone + {' + '.join(covariates)}",
                data=clean
            ).fit()
            aov   = anova_lm(model, typ=2)
            f_val = aov.loc["tone", "F"]
            p_val = aov.loc["tone", "PR(>F)"]
            ancova_rows.append({
                "DV":         dv,
                "Covariates": ", ".join(covariates),
                "N":          len(clean),
                "F":          round(f_val, 3),
                "p":          round(p_val, 4),
                "Sig.":       _sig_label(p_val),
            })
        except Exception as e:
            log.warning(f"ANCOVA {dv}: {e}")

    # ------------------------------------------------------------------
    # Interactions — tone × language and tone × gender_grouped
    # Uses gender_grouped (Male/Female/Other) for interactions
    # Note: 'Other' category has small n — interpret with caution
    # ------------------------------------------------------------------
    inter_rows = []
    interaction_vars = []

    if "language" in df_analysis.columns:
        interaction_vars.append(("language", "tone * language"))
    if "gender_grouped" in df_analysis.columns:
        interaction_vars.append(("gender_grouped", "tone * gender_grouped"))

    for demo_var, interaction_term in interaction_vars:
        for dv in ["composite","engagement_score","E3","E4","E5","E6",
                   "PM_score","Comp_score","MP_score","emotions_mean"]:
            if not _col_has_data(df_analysis, dv):
                continue
            clean = df_analysis[[dv, "tone", demo_var]].dropna()
            if len(clean) < 20:
                continue
            try:
                model   = ols(
                    f"{dv} ~ {interaction_term}",
                    data=clean
                ).fit()
                aov     = anova_lm(model, typ=2)
                int_key = [k for k in aov.index if ":" in k]
                if not int_key:
                    continue
                f_val = aov.loc[int_key[0], "F"]
                p_val = aov.loc[int_key[0], "PR(>F)"]
                note  = (
                    " (Other n=5 — interpret with caution)"
                    if demo_var == "gender_grouped" else ""
                )
                inter_rows.append({
                    "Interaction":  f"tone × {demo_var}{note}",
                    "DV":           dv,
                    "N":            len(clean),
                    "F":            round(f_val, 3),
                    "p":            round(p_val, 4),
                    "Sig.":         _sig_label(p_val),
                })
            except Exception as e:
                log.warning(
                    f"Interaction {demo_var} × tone → {dv}: {e}"
                )

    descriptives = pd.DataFrame(desc_rows)
    ancova       = pd.DataFrame(ancova_rows)
    interactions = pd.DataFrame(inter_rows)

    # Recaps
    ancova_recap = _make_recap(ancova)
    inter_recap  = _make_recap(interactions)

    log.info(
        f"Sheet I: {len(ancova_rows)} ANCOVA tests, "
        f"{len(inter_rows)} interaction tests."
    )

    return {
        "descriptives":   descriptives,
        "ancova":         ancova,
        "ancova_recap":   ancova_recap,
        "interactions":   interactions,
        "inter_recap":    inter_recap,
    }
