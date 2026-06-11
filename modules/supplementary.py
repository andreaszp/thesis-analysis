"""
supplementary.py
----------------
All supplementary analyses referenced in the Results section.
Called from export pipeline after main analyses.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

log = logging.getLogger(__name__)


def run_supplementary_analyses(df: pd.DataFrame) -> dict:
    """
    Run all supplementary analyses and return dict of DataFrames.
    """
    results = {}

    # Centre variables for moderation analyses
    df = df.copy()
    for var in ['PM_score', 'Comp_score', 'MP_score',
                'Ind1', 'Ind2', 'MA1', 'MA2', 'tone']:
        if var in df.columns:
            df[f'{var}_c'] = df[var] - df[var].mean()

    mp_sd = df['MP_score'].std() if 'MP_score' in df.columns else 1.0

    # ------------------------------------------------------------------
    # 1. Supplementary correlations
    # ------------------------------------------------------------------
    pairs = [
        ('E3',            'E5',            'Appreciation × Reuse intention'),
        ('E2',            'E5',            'Engagement felt × Reuse intention'),
        ('E2',            'E3',            'Engagement felt × Appreciation'),
        ('E2',            'E4',            'Engagement felt × Utility'),
        ('PP2',           'E3',            'Professional × Appreciation'),
        ('PP2',           'E4',            'Professional × Utility'),
        ('PP2',           'E5',            'Professional × Reuse intention'),
        ('PP2',           'PM_score',      'Professional × PM score'),
        ('PP2',           'composite',     'Professional × Feedback composite'),
        ('PP5',           'E3',            'Formal × Appreciation'),
        ('PP5',           'E4',            'Formal × Utility'),
        ('PP5',           'E5',            'Formal × Reuse intention'),
        ('E1',            'E3',            'Effort × Appreciation'),
        ('E1',            'E4',            'Effort × Utility'),
        ('E1',            'E5',            'Effort × Reuse intention'),
        ('PM_score',      'E1',            'PM score × Effort'),
        ('MA1',           'PM_score',      'MA1 × PM score'),
        ('MA2',           'PM_score',      'MA2 × PM score'),
        ('MA1',           'Comp_score',    'MA1 × Competence'),
        ('MA2',           'Comp_score',    'MA2 × Competence'),
        ('MA1',           'Ind1',          'MA1 × Autonomy Ind1'),
        ('MA2',           'Ind1',          'MA2 × Autonomy Ind1'),
        ('emotions_mean', 'engagement_score', 'Emotional expression × Engagement score'),
        ('emotions_mean', 'composite',     'Emotional expression × Composite'),
        ('emotions_mean', 'E2',            'Emotional expression × Engagement felt'),
        ('PP1',           'PP2',           'Friendly × Professional'),
        ('PP3',           'PP4',           'Approachable × Warm'),
    ]

    corr_rows = []
    for v1, v2, label in pairs:
        if v1 not in df.columns or v2 not in df.columns:
            continue
        clean = df[[v1, v2]].dropna()
        if len(clean) < 10:
            continue
        r, p = stats.pearsonr(clean[v1], clean[v2])

        def _sig(p):
            if p < .001: return '***'
            if p < .01:  return '**'
            if p < .05:  return '*'
            return 'ns'

        corr_rows.append({
            'Pair':        label,
            'Variable 1':  v1,
            'Variable 2':  v2,
            'N':           len(clean),
            'r':           round(r, 3),
            'p':           round(p, 4),
            'Sig.':        _sig(p),
            'Strength':    'strong' if abs(r) >= .5
                           else 'moderate' if abs(r) >= .3
                           else 'weak',
            'Direction':   'positive' if r > 0 else 'negative',
        })

    results['correlations'] = pd.DataFrame(corr_rows)

    # ------------------------------------------------------------------
    # 2. Tone × Language × Composite / Engagement
    # ------------------------------------------------------------------
    tone_lang_rows = []
    for tone_val, tone_label in {1: 'Casual', 0: 'Formal'}.items():
        for lang in ['FR', 'EN']:
            sub = df[(df['tone'] == tone_val) & (df['language'] == lang)]
            comp = sub['composite'].dropna() if 'composite' in df.columns else pd.Series()
            eng  = sub['engagement_score'].dropna()
            tone_lang_rows.append({
                'Condition':             tone_label,
                'Language':              lang,
                'N':                     len(sub),
                'M composite':           round(comp.mean(), 3) if len(comp) else None,
                'SD composite':          round(comp.std(),  3) if len(comp) else None,
                'M engagement':          round(eng.mean(),  3) if len(eng)  else None,
                'SD engagement':         round(eng.std(),   3) if len(eng)  else None,
            })

    for lang in ['FR', 'EN']:
        cas_c = df[(df['tone'] == 1) & (df['language'] == lang)]['composite'].dropna() \
                if 'composite' in df.columns else pd.Series()
        pro_c = df[(df['tone'] == 0) & (df['language'] == lang)]['composite'].dropna() \
                if 'composite' in df.columns else pd.Series()
        cas_e = df[(df['tone'] == 1) & (df['language'] == lang)]['engagement_score'].dropna()
        pro_e = df[(df['tone'] == 0) & (df['language'] == lang)]['engagement_score'].dropna()

        def _ttest(a, b):
            if len(a) < 2 or len(b) < 2:
                return None, None
            t, p = stats.ttest_ind(a, b, equal_var=False)
            return round(t, 3), round(p, 4)

        t_c, p_c = _ttest(cas_c, pro_c)
        t_e, p_e = _ttest(cas_e, pro_e)

        tone_lang_rows.append({
            'Condition':     f'Test Casual vs Formal — {lang}',
            'Language':      lang,
            'N':             len(cas_c) + len(pro_c) if len(cas_c) else len(cas_e) + len(pro_e),
            'M composite':   f't={t_c}' if t_c else '—',
            'SD composite':  f'p={p_c}' if p_c else '—',
            'M engagement':  f't={t_e}' if t_e else '—',
            'SD engagement': f'p={p_e}' if p_e else '—',
        })

    results['tone_language'] = pd.DataFrame(tone_lang_rows)

    # ------------------------------------------------------------------
    # 3. Emotions by tone
    # ------------------------------------------------------------------
    if 'emotions_mean' in df.columns:
        em_rows = []
        for tone_val, tone_label in {1: 'Casual', 0: 'Formal'}.items():
            sub = df[df['tone'] == tone_val]['emotions_mean'].dropna()
            em_rows.append({
                'Condition': tone_label,
                'N': len(sub),
                'M': round(sub.mean(), 3),
                'SD': round(sub.std(), 3),
                'Min': round(sub.min(), 3),
                'Max': round(sub.max(), 3),
            })
        cas = df[df['tone'] == 1]['emotions_mean'].dropna()
        pro = df[df['tone'] == 0]['emotions_mean'].dropna()
        t, p = stats.ttest_ind(cas, pro, equal_var=False)
        em_rows.append({
            'Condition': 'Test Casual vs Formal',
            'N': len(cas) + len(pro),
            'M': f't={round(t,3)}',
            'SD': f'p={round(p,4)} — ns',
            'Min': '', 'Max': '',
        })
        results['emotions_tone'] = pd.DataFrame(em_rows)

    # ------------------------------------------------------------------
    # 4. H3 by tone (Comp1 - Comp2)
    # ------------------------------------------------------------------
    if 'Comp1' in df.columns and 'Comp2' in df.columns:
        df['comp_diff'] = df['Comp1'] - df['Comp2']
        h3_rows = []
        for tone_val, tone_label in {1: 'Casual', 0: 'Formal'}.items():
            sub = df[df['tone'] == tone_val]['comp_diff'].dropna()
            h3_rows.append({
                'Condition': tone_label,
                'N': len(sub),
                'M (Comp1-Comp2)': round(sub.mean(), 3),
                'SD': round(sub.std(), 3),
            })
        cas = df[df['tone'] == 1]['comp_diff'].dropna()
        pro = df[df['tone'] == 0]['comp_diff'].dropna()
        t, p = stats.ttest_ind(cas, pro, equal_var=False)
        h3_rows.append({
            'Condition': 'Test Casual vs Formal',
            'N': len(cas) + len(pro),
            'M (Comp1-Comp2)': f't={round(t,3)}',
            'SD': f'p={round(p,4)} — ns',
        })
        results['h3_by_tone'] = pd.DataFrame(h3_rows)

    # ------------------------------------------------------------------
    # 5. PP2 vs PP5 correlations with outcomes
    # ------------------------------------------------------------------
    pp_rows = []
    for pp, label in [('PP2', 'Professional'), ('PP5', 'Formal')]:
        if pp not in df.columns:
            continue
        for ev, ev_label in [
            ('E3', 'Appreciation'), ('E4', 'Utility'),
            ('E5', 'Reuse intention'), ('E6', 'Preference'),
            ('PM_score', 'PM score'), ('composite', 'Feedback composite'),
        ]:
            if ev not in df.columns:
                continue
            clean = df[[pp, ev]].dropna()
            r, p  = stats.pearsonr(clean[pp], clean[ev])
            sig   = '***' if p < .001 else '**' if p < .01 \
                    else '*' if p < .05 else 'ns'
            pp_rows.append({
                'Personality dim.': label,
                'Variable':         pp,
                'Outcome':          ev_label,
                'r':                round(r, 3),
                'p':                round(p, 4),
                'Sig.':             sig,
            })
    results['pp2_vs_pp5'] = pd.DataFrame(pp_rows)

    # ------------------------------------------------------------------
    # 6. Interaction PM × AI perceptions → E3/E4/E5
    # ------------------------------------------------------------------
    interaction_rows = []
    perceptions_c = [
        ('Comp_c',  'Competence'),
        ('Ind1_c',  'Autonomy Ind1'),
        ('Ind2_c',  'Autonomy Ind2'),
        ('MP_c',    'Moral Patiency'),
        ('MA1_c',   'Moral gravity MA1'),
        ('MA2_c',   'AI moral resp. MA2'),
    ]
    for dv in ['E3', 'E4', 'E5']:
        if dv not in df.columns:
            continue
        for var_c, var_label in perceptions_c:
            if var_c not in df.columns:
                continue
            try:
                model   = smf.ols(
                    f'{dv} ~ PM_score_c + {var_c} + PM_score_c:{var_c} + tone',
                    data=df
                ).fit()
                int_key = f'PM_score_c:{var_c}'
                beta    = model.params[int_key]
                p_val   = model.pvalues[int_key]
                sig     = '***' if p_val < .001 else '**' if p_val < .01 \
                          else '*' if p_val < .05 else 'ns'
                interaction_rows.append({
                    'DV':              dv,
                    'Interaction':     f'PM × {var_label}',
                    'β interaction':   round(float(beta), 3),
                    'p':               round(float(p_val), 4),
                    'Sig.':            sig,
                    'Interpretation':  (
                        f'PM effect on {dv} depends on {var_label}'
                        if p_val < .05 else
                        f'No moderation of PM effect on {dv}'
                    ),
                })
            except Exception as e:
                log.warning(f'Interaction PM × {var_label} → {dv}: {e}')

    results['pm_interactions'] = pd.DataFrame(interaction_rows)

    # ------------------------------------------------------------------
    # 7. Simple slopes: PM × MP → E3
    # ------------------------------------------------------------------
    slopes_rows = []
    if all(c in df.columns for c in ['PM_score_c', 'MP_c', 'E3']):
        try:
            model_int = smf.ols(
                'E3 ~ PM_score_c + MP_c + PM_score_c:MP_c + tone',
                data=df
            ).fit()
            for label, mp_val in [
                ('Low MP (M−1SD)', -mp_sd),
                ('Mean MP',         0),
                ('High MP (M+1SD)', mp_sd),
            ]:
                slope = (model_int.params['PM_score_c']
                         + model_int.params['PM_score_c:MP_c'] * mp_val)
                slopes_rows.append({
                    'Level of Moral Patiency': label,
                    'Effect of PM on E3 (β)':  round(float(slope), 3),
                    'Interpretation': (
                        'PM strongly reduces appreciation'  if mp_val < 0 else
                        'PM moderately reduces appreciation' if mp_val == 0 else
                        'PM effect weaker at high MP'
                    ),
                })
            slopes_rows.append({
                'Level of Moral Patiency': 'Interaction β',
                'Effect of PM on E3 (β)':  round(
                    float(model_int.params['PM_score_c:MP_c']), 3
                ),
                'Interpretation': (
                    f"p={round(float(model_int.pvalues['PM_score_c:MP_c']),4)}"
                ),
            })
        except Exception as e:
            log.warning(f'Simple slopes PM × MP: {e}')

    results['simple_slopes'] = pd.DataFrame(slopes_rows)

    # ------------------------------------------------------------------
    # 8. Moderation: tone × PM → outcomes
    # ------------------------------------------------------------------
    mod_rows = []
    for dv, dv_label in [
        ('composite', 'Feedback Quality'),
        ('E3', 'Appreciation'),
        ('E4', 'Utility'),
        ('E5', 'Reuse intention'),
    ]:
        if dv not in df.columns or df[dv].dropna().empty:
            continue
        try:
            model = smf.ols(
                f'{dv} ~ PM_score_c + tone_c + PM_score_c:tone_c',
                data=df
            ).fit()
            beta    = model.params['PM_score_c:tone_c']
            p_val   = model.pvalues['PM_score_c:tone_c']
            sig     = '***' if p_val < .001 else '**' if p_val < .01 \
                      else '*' if p_val < .05 else 'ns'
            s_cas   = (model.params['PM_score_c']
                       + model.params['PM_score_c:tone_c'] * 0.5)
            s_for   = (model.params['PM_score_c']
                       + model.params['PM_score_c:tone_c'] * (-0.5))
            mod_rows.append({
                'DV':               dv_label,
                'β PM (main)':      round(float(model.params['PM_score_c']), 3),
                'β tone (main)':    round(float(model.params['tone_c']), 3),
                'β interaction':    round(float(beta), 3),
                'p (interaction)':  round(float(p_val), 4),
                'Sig.':             sig,
                'β PM in Casual':   round(float(s_cas), 3),
                'β PM in Formal':   round(float(s_for), 3),
                'Interpretation':   (
                    f'PM effect on {dv_label} differs by tone'
                    if p_val < .05 else
                    f'PM effect consistent across tone conditions'
                ),
            })
        except Exception as e:
            log.warning(f'Moderation tone × PM → {dv}: {e}')

    results['tone_pm_moderation'] = pd.DataFrame(mod_rows)

    # ------------------------------------------------------------------
    # 9. Mediation: tone → PP1/PP3/PP4 → PM_score
    # ------------------------------------------------------------------
    try:
        from modules.analyses import _run_simple_mediation
        pp_med_rows = []
        for pp, pp_label in [
            ('PP1', 'Friendly'), ('PP3', 'Approachable'), ('PP4', 'Warm')
        ]:
            if pp not in df.columns:
                continue
            res = _run_simple_mediation(df, 'tone', pp, 'PM_score', 'supp')
            if res:
                pp_med_rows.append({
                    'Path':        f'tone → {pp_label} ({pp}) → PM_score',
                    'a':           res['a'],
                    'p_a':         res['p_a'],
                    'b':           res['b'],
                    'p_b':         res['p_b'],
                    'c (total)':   res['c'],
                    "c' (direct)": res['c_prime'],
                    'Indirect':    res['Indirect'],
                    'CI_low':      res['CI_low'],
                    'CI_high':     res['CI_high'],
                    'Type':        res['Mediation_type'],
                    'Sig.':        '*' if (
                        res['CI_low'] > 0 or res['CI_high'] < 0
                    ) else 'ns',
                })
        results['pp_pm_mediations'] = pd.DataFrame(pp_med_rows)
    except Exception as e:
        log.warning(f'PP → PM mediations: {e}')
        results['pp_pm_mediations'] = pd.DataFrame()

    # ------------------------------------------------------------------
    # 10. Serial mediation: tone → PM → E1 → E3/E4/E5
    # ------------------------------------------------------------------
    try:
        from modules.analyses import _run_chain_mediation
        serial_rows = []
        for dv, dv_label in [
            ('E3', 'Appreciation'), ('E4', 'Utility'), ('E5', 'Reuse intention')
        ]:
            if dv not in df.columns:
                continue
            res = _run_chain_mediation(
                df, 'tone', ['PM_score', 'E1'], dv, 'supp'
            )
            if res:
                serial_rows.append({
                    'Path':           f'tone → PM → E1 → {dv_label}',
                    'Indirect':       res['Indirect'],
                    'CI_low':         res['CI_low'],
                    'CI_high':        res['CI_high'],
                    'Type':           res.get('Mediation_type', '—'),
                    'Sig.':           '*' if (
                        res['CI_low'] > 0 or res['CI_high'] < 0
                    ) else 'ns',
                    'Interpretation': (
                        'Serial mediation significant'
                        if (res['CI_low'] > 0 or res['CI_high'] < 0)
                        else 'Not significant — PM effect is direct'
                    ),
                })
        results['serial_mediations'] = pd.DataFrame(serial_rows)
    except Exception as e:
        log.warning(f'Serial mediations: {e}')
        results['serial_mediations'] = pd.DataFrame()

    # ------------------------------------------------------------------
    # 11. E6 regression with PP2
    # ------------------------------------------------------------------
    avail = [
        v for v in ['tone', 'PM_score', 'Comp_score', 'Ind1', 'Ind2',
                    'MA1', 'MA2', 'MP_score', 'PP2']
        if v in df.columns
    ]
    if 'E6' in df.columns and avail:
        try:
            model      = smf.ols(f'E6 ~ {" + ".join(avail)}', data=df).fit()
            model_base = smf.ols(
                f'E6 ~ {" + ".join([v for v in avail if v != "PP2"])}',
                data=df
            ).fit()

            def _sig(p):
                if p < .001: return '***'
                if p < .01:  return '**'
                if p < .05:  return '*'
                return 'ns'

            pred_labels = {
                'tone': 'Chatbot Tone',
                'PM_score': 'Perceived Manipulation',
                'Comp_score': 'Competence',
                'Ind1': 'Autonomy Ind1',
                'Ind2': 'Autonomy Ind2',
                'MA1': 'Moral gravity MA1',
                'MA2': 'AI moral resp. MA2',
                'MP_score': 'Moral Patiency',
                'PP2': 'Perceived Professionalism',
            }
            e6_rows = []
            for pred in avail:
                e6_rows.append({
                    'Predictor': pred,
                    'Label':     pred_labels.get(pred, pred),
                    'β':         round(float(model.params.get(pred, 0)), 3),
                    'SE':        round(float(model.bse.get(pred, 0)), 3),
                    't':         round(float(model.tvalues.get(pred, 0)), 3),
                    'p':         round(float(model.pvalues.get(pred, 0)), 4),
                    'Sig.':      _sig(float(model.pvalues.get(pred, 1))),
                })
            results['e6_regression'] = pd.DataFrame(e6_rows)
            results['e6_model_comparison'] = pd.DataFrame([{
                'Model':   'Base (without PP2)',
                'R²':      round(model_base.rsquared, 3),
                'ΔR²':     '—',
                'PP2 β':   '—',
                'PP2 p':   '—',
            }, {
                'Model':   'Extended (with PP2)',
                'R²':      round(model.rsquared, 3),
                'ΔR²':     round(model.rsquared - model_base.rsquared, 3),
                'PP2 β':   round(float(model.params.get('PP2', 0)), 3),
                'PP2 p':   round(float(model.pvalues.get('PP2', 1)), 4),
            }])
        except Exception as e:
            log.warning(f'E6 regression: {e}')

    # ------------------------------------------------------------------
    # 12. Friendly+Pro vs Formal+Pro group comparison
    # ------------------------------------------------------------------
    if all(c in df.columns for c in ['PP1', 'PP2', 'PP5']):
        med_pp1 = df['PP1'].median()
        med_pp2 = df['PP2'].median()
        med_pp5 = df['PP5'].median()
        grp_fp  = df[(df['PP1'] > med_pp1) & (df['PP2'] > med_pp2)]
        grp_fop = df[(df['PP5'] > med_pp5) & (df['PP2'] > med_pp2)]

        fp_rows = []
        for dv, dv_label in [
            ('composite', 'Feedback Quality'),
            ('E3', 'Appreciation'), ('E4', 'Utility'),
            ('E5', 'Reuse intention'), ('E6', 'Preference'),
        ]:
            if dv not in df.columns:
                continue
            fp  = grp_fp[dv].dropna()
            fop = grp_fop[dv].dropna()
            if len(fp) < 5 or len(fop) < 5:
                continue
            u, p = stats.mannwhitneyu(fp, fop, alternative='two-sided')
            fp_rows.append({
                'Outcome':         dv_label,
                'N (F+Pro)':       len(fp),
                'M (F+Pro)':       round(float(fp.mean()), 3),
                'SD (F+Pro)':      round(float(fp.std()),  3),
                'N (Formal+Pro)':  len(fop),
                'M (Formal+Pro)':  round(float(fop.mean()), 3),
                'SD (Formal+Pro)': round(float(fop.std()),  3),
                'U':               round(float(u), 1),
                'p':               round(float(p), 4),
                'Sig.':            '***' if p < .001 else '**' if p < .01
                                   else '*' if p < .05 else 'ns',
            })
        results['friendly_pro_comparison'] = pd.DataFrame(fp_rows)

    log.info(f'Supplementary analyses complete: {list(results.keys())}')
    return results

def run_pp_analyses(df: pd.DataFrame) -> dict:
    """
    Section PP — Personality Perceptions: Additional Analyses
    PP-1: simple regressions each PP → composite
    PP-2: multiple regression all PP → composite
    PP-3: tone → each PP → composite (simple mediation)
    PP-4: tone → each PP → PM_score → composite (serial chain)
    PP-5: tone → each PP → PM_score → E3/E4/E5 (serial chain)
    PP-6: PP4 → MP_score → E3 (simple mediation)
    """
    import statsmodels.formula.api as smf
    results = {}

    # ------------------------------------
    # PP-1 Simple regressions PP → composite
    # ------------------------------------
    pp1_rows = []
    for pp in ['PP1', 'PP2', 'PP3', 'PP4', 'PP5']:
        if pp not in df.columns or 'composite' not in df.columns:
            continue
        try:
            model = smf.ols(f'composite ~ {pp}', data=df).fit()
            beta  = model.params[pp]
            se    = model.bse[pp]
            t     = model.tvalues[pp]
            p     = model.pvalues[pp]
            r2    = model.rsquared
            pp1_rows.append({
                'PP item':  pp,
                'β':        round(float(beta), 3),
                'SE':       round(float(se),   3),
                't':        round(float(t),    3),
                'p':        round(float(p),    4),
                'R²':       round(float(r2),   3),
                'Sig.':     _sig_label(p),
            })
        except Exception as e:
            log.warning(f'PP-1 {pp}: {e}')
    results['pp1_simple_regressions'] = pd.DataFrame(pp1_rows)

    # ------------------------------------
    # PP-2 Multiple regression all PP → composite
    # ------------------------------------
    pp_avail = [p for p in ['PP1','PP2','PP3','PP4','PP5']
                if p in df.columns]
    pp2_rows = []
    if pp_avail and 'composite' in df.columns:
        try:
            formula = f'composite ~ {" + ".join(pp_avail)}'
            model   = smf.ols(formula, data=df).fit()
            r2      = model.rsquared
            for pp in pp_avail:
                beta = model.params[pp]
                se   = model.bse[pp]
                t    = model.tvalues[pp]
                p    = model.pvalues[pp]
                pp2_rows.append({
                    'PP item':  pp,
                    'β':        round(float(beta), 3),
                    'SE':       round(float(se),   3),
                    't':        round(float(t),    3),
                    'p':        round(float(p),    4),
                    'R²':       round(float(r2),   3),
                    'Sig.':     _sig_label(p),
                })
        except Exception as e:
            log.warning(f'PP-2: {e}')
    results['pp2_multiple_regression'] = pd.DataFrame(pp2_rows)

    # ------------------------------------
    # PP-3 Mediation: tone → each PP → composite
    # ------------------------------------
    try:
        from modules.analyses import _run_simple_mediation
        pp3_rows = []
        for pp in ['PP1', 'PP2', 'PP3', 'PP4']:
            if pp not in df.columns:
                continue
            res = _run_simple_mediation(df, 'tone', pp, 'composite', 'pp')
            if res:
                pp3_rows.append({
                    'Path':        f'tone → {pp} → composite',
                    'a':           round(float(res['a']),        3),
                    'p_a':         round(float(res['p_a']),      4),
                    'b':           round(float(res['b']),        3),
                    'p_b':         round(float(res['p_b']),      4),
                    'c (total)':   round(float(res['c']),        3),
                    "c' (direct)": round(float(res['c_prime']),  3),
                    'Indirect':    round(float(res['Indirect']), 3),
                    'CI_low':      round(float(res['CI_low']),   3),
                    'CI_high':     round(float(res['CI_high']),  3),
                    'Type':        res['Mediation_type'],
                    'Sig.':        '*' if (res['CI_low'] > 0 or
                                           res['CI_high'] < 0) else 'ns',
                })
        results['pp3_mediations'] = pd.DataFrame(pp3_rows)
    except Exception as e:
        log.warning(f'PP-3: {e}')
        results['pp3_mediations'] = pd.DataFrame()

    # ------------------------------------
    # PP-4 Serial chain: tone → PP → PM_score → composite
    # ------------------------------------
    try:
        from modules.analyses import _run_chain_mediation
        pp4_rows = []
        for pp in ['PP1', 'PP3', 'PP4']:
            if pp not in df.columns:
                continue
            res = _run_chain_mediation(
                df, 'tone', [pp, 'PM_score'], 'composite', 'pp'
            )
            if res:
                pp4_rows.append({
                    'Path':     f'tone → {pp} → PM_score → composite',
                    'Indirect': round(float(res['Indirect']), 3),
                    'CI_low':   round(float(res['CI_low']),   3),
                    'CI_high':  round(float(res['CI_high']),  3),
                    'Type':     res.get('Mediation_type', '—'),
                    'Sig.':     '*' if (res['CI_low'] > 0 or
                                        res['CI_high'] < 0) else 'ns',
                })
        results['pp4_serial_composite'] = pd.DataFrame(pp4_rows)
    except Exception as e:
        log.warning(f'PP-4: {e}')
        results['pp4_serial_composite'] = pd.DataFrame()

    # ------------------------------------
    # PP-5 Serial chain: tone → PP → PM_score → E3/E4/E5
    # ------------------------------------
    try:
        from modules.analyses import _run_chain_mediation
        pp5_rows = []
        for pp in ['PP1', 'PP3', 'PP4']:
            if pp not in df.columns:
                continue
            for dv in ['E3', 'E4', 'E5']:
                if dv not in df.columns:
                    continue
                res = _run_chain_mediation(
                    df, 'tone', [pp, 'PM_score'], dv, 'pp'
                )
                if res:
                    pp5_rows.append({
                        'Path':     f'tone → {pp} → PM_score → {dv}',
                        'PP item':  pp,
                        'DV':       dv,
                        'Indirect': round(float(res['Indirect']), 3),
                        'CI_low':   round(float(res['CI_low']),   3),
                        'CI_high':  round(float(res['CI_high']),  3),
                        'Type':     res.get('Mediation_type', '—'),
                        'Sig.':     '*' if (res['CI_low'] > 0 or
                                            res['CI_high'] < 0) else 'ns',
                    })
        results['pp5_serial_eval'] = pd.DataFrame(pp5_rows)
    except Exception as e:
        log.warning(f'PP-5: {e}')
        results['pp5_serial_eval'] = pd.DataFrame()

    # ------------------------------------
    # PP-6 Mediation: PP4 → MP_score → E3
    # ------------------------------------
    try:
        from modules.analyses import _run_simple_mediation
        pp6_rows = []
        if all(c in df.columns for c in ['PP4', 'MP_score', 'E3']):
            res = _run_simple_mediation(df, 'PP4', 'MP_score', 'E3', 'pp')
            if res:
                pp6_rows.append({
                    'Path':        'PP4 → MP_score → E3',
                    'a':           round(float(res['a']),        3),
                    'p_a':         round(float(res['p_a']),      4),
                    'b':           round(float(res['b']),        3),
                    'p_b':         round(float(res['p_b']),      4),
                    'c (total)':   round(float(res['c']),        3),
                    "c' (direct)": round(float(res['c_prime']),  3),
                    'Indirect':    round(float(res['Indirect']), 3),
                    'CI_low':      round(float(res['CI_low']),   3),
                    'CI_high':     round(float(res['CI_high']),  3),
                    'Type':        res['Mediation_type'],
                    'Sig.':        '*' if (res['CI_low'] > 0 or
                                           res['CI_high'] < 0) else 'ns',
                })
        results['pp6_pp4_mp_e3'] = pd.DataFrame(pp6_rows)
    except Exception as e:
        log.warning(f'PP-6: {e}')
        results['pp6_pp4_mp_e3'] = pd.DataFrame()

    log.info(f'PP analyses complete: {list(results.keys())}')
    return results


def _sig_label(p):
    try:
        p = float(p)
        if p < .001: return '***'
        if p < .01:  return '**'
        if p < .05:  return '*'
        return 'ns'
    except:
        return '—'
