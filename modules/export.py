"""
export.py
---------
Final step: write all results to a single formatted Excel file.

Color scheme:
    Headers:    deep navy
    Blocks:     teal / burnt orange / sage green / violet / dark grey
    p-values:   grey (ns) / orange (p<0.05) / green (p<0.01) /
                dark green (p<0.001)
    Rows:       white / very light blue alternating
"""

import logging
import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-built fills
# ---------------------------------------------------------------------------
def _fill(hex_color: str) -> PatternFill:
    return PatternFill(
        start_color=hex_color,
        end_color=hex_color,
        fill_type="solid"
    )

BLOCK_FILLS = {
    "ai":          _fill("1A5276"),
    "eval":        _fill("784212"),
    "quality":     _fill("1D6A39"),
    "engage":      _fill("4A235A"),
    "demo":        _fill("424949"),
    "personality": _fill("7B241C"),
    "tone":        _fill("1F618D"),
    "default":     _fill("1B4F72"),
}

SECTION_PRIMARY   = _fill("1B4F72")
SECTION_SECONDARY = _fill("2E86C1")
ROW_WHITE         = _fill("FFFFFF")
ROW_ALT           = _fill("EBF5FB")

P_FILLS = {
    "ns":   _fill("E8E8E8"),
    "p05":  _fill("F39C12"),
    "p01":  _fill("27AE60"),
    "p001": _fill("1E8449"),
}

THIN_BORDER = Border(bottom=Side(style="thin", color="CCCCCC"))


# ---------------------------------------------------------------------------
# p-value color
# ---------------------------------------------------------------------------
def _p_fill(p_val) -> PatternFill | None:
    try:
        p = float(p_val)
        if p < 0.001: return P_FILLS["p001"]
        if p < 0.01:  return P_FILLS["p01"]
        if p < 0.05:  return P_FILLS["p05"]
        return P_FILLS["ns"]
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
PREDICTOR_LABELS: dict = {
    "tone":             "Chatbot Tone (0=Pro / 1=Friendly)",
    "PM_score":         "Perceived Manipulation (composite)",
    "PM1":              "Perceived Manipulation — item 1 (threat to freedom)",
    "PM2":              "Perceived Manipulation — item 2 (decision override)",
    "PM3":              "Perceived Manipulation — item 3 (manipulation attempt)",
    "PM4":              "Perceived Manipulation — item 4 (pressure felt)",
    "Comp_score":       "Competence (composite)",
    "Comp1":            "Competence — item 1 (skills judgment)",
    "Comp2":            "Competence — item 2 (morality judgment)",
    "Ind_score":        "Autonomy (composite)",
    "Ind1":             "Autonomy — item 1 (AI plans & goals)",
    "Ind2":             "Autonomy — item 2 (AI self-control)",
    "MA_score":         "Moral Agency (composite)",
    "MA1":              "Moral Agency — item 1 (moral gravity AI→human)",
    "MA2":              "Moral Agency — item 2 (AI moral responsibility)",
    "MP_score":         "Moral Patiency (composite)",
    "MP1":              "Moral Patiency — item 1 (moral gravity human→AI)",
    "MP2":              "Moral Patiency — item 2 (AI right to consideration)",
    "E1":               "Required effort",
    "E2":               "Engagement felt",
    "E3":               "Chatbot appreciation",
    "E4":               "Conversation utility",
    "E5":               "Reuse intention",
    "E6":               "Chatbot preference",
    "composite":        "Feedback composite score",
    "engagement_score": "Engagement score (behavioural)",
    "emotions_mean":    "Emotional expression (mean)",
    "quantity_mean":    "Feedback quantity (mean)",
    "quality_mean":     "Feedback quality (mean)",
}

DV_LABELS: dict = {
    "composite":        "Feedback Composite Score",
    "engagement_score": "Engagement Score (behavioural)",
    "emotions_mean":    "Emotional Expression",
    "E3":               "Chatbot Appreciation (E3)",
    "E4":               "Conversation Utility (E4)",
    "E5":               "Reuse Intention (E5)",
    "E6":               "Chatbot Preference (E6)",
}

IV_LABELS             = PREDICTOR_LABELS
DV_LABELS_REGRESSION  = {
    "PM_score":  "Perceived Manipulation (composite)",
    "PM1":       "PM — threat to freedom",
    "PM2":       "PM — decision override",
    "PM3":       "PM — manipulation attempt",
    "PM4":       "PM — pressure felt",
    "Comp_score":"Competence (composite)",
    "Comp1":     "Competence — skills",
    "Comp2":     "Competence — morality",
    "Ind_score": "Autonomy (composite)",
    "Ind1":      "Autonomy — AI plans & goals",
    "Ind2":      "Autonomy — AI self-control",
    "MA_score":  "Moral Agency (composite)",
    "MA1":       "Moral Agency — moral gravity AI→human",
    "MA2":       "Moral Agency — AI moral responsibility",
    "MP_score":  "Moral Patiency (composite)",
    "MP1":       "Moral Patiency — moral gravity human→AI",
    "MP2":       "Moral Patiency — AI right to consideration",
}


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
def _write_value(cell, value) -> None:
    if isinstance(value, (np.integer,)):
        cell.value = int(value)
    elif isinstance(value, (np.floating,)):
        cell.value = float(value) if not np.isnan(value) else None
    elif isinstance(value, float) and np.isnan(value):
        cell.value = None
    elif isinstance(value, bool):
        cell.value = str(value)
    else:
        cell.value = value if value is not None else None


def _style_header(ws, row_num: int, n_cols: int,
                  bg: str = "1B4F72") -> None:
    for col in range(1, n_cols + 1):
        cell           = ws.cell(row=row_num, column=col)
        cell.fill      = _fill(bg)
        cell.font      = Font(bold=True, color="FFFFFF", size=10)
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )
        cell.border = THIN_BORDER
    ws.row_dimensions[row_num].height = 30


def _style_data_row(ws, row_num: int, n_cols: int,
                    alt: bool = False) -> None:
    fill = ROW_ALT if alt else ROW_WHITE
    for col in range(1, n_cols + 1):
        cell           = ws.cell(row=row_num, column=col)
        cell.fill      = fill
        cell.font      = Font(size=10)
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def _color_p_values(ws, row_num: int, headers: list) -> None:
    p_cols = [
        i+1 for i, h in enumerate(headers)
        if str(h).lower() in (
            "p", "p-value", "p_fdr", "p_a",
            "p_b", "p_c", "p_cprime", "p_fchange"
        )
    ]
    for col in p_cols:
        cell  = ws.cell(row=row_num, column=col)
        p_val = cell.value
        fill  = _p_fill(p_val)
        if fill:
            cell.fill = fill
            cell.font = Font(size=10, bold=True)


def _autofit(ws, min_w: int = 10, max_w: int = 45) -> None:
    for col in ws.columns:
        max_len    = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                max_len = max(max_len, len(str(cell.value or "")))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(
            max(max_len + 2, min_w), max_w
        )


def _freeze(ws) -> None:
    ws.freeze_panes = "A2"


def _write_section_title(ws, row_num: int, title: str,
                          n_cols: int,
                          primary: bool = True) -> int:
    fill  = SECTION_PRIMARY if primary else SECTION_SECONDARY
    cell  = ws.cell(row=row_num, column=1)
    cell.value     = title
    cell.font      = Font(bold=True, size=11, color="FFFFFF")
    cell.fill      = fill
    cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[row_num].height = 22
    for col in range(2, n_cols + 1):
        ws.cell(row=row_num, column=col).fill = fill
    return row_num + 1


def _write_df(ws, df: pd.DataFrame, start_row: int,
              color_p: bool = True) -> int:
    if df is None or df.empty:
        cell       = ws.cell(row=start_row, column=1)
        cell.value = "No results."
        cell.font  = Font(italic=True, color="888888")
        return start_row + 2

    headers = list(df.columns)
    n_cols  = len(headers)

    for col_idx, h in enumerate(headers, start=1):
        ws.cell(row=start_row, column=col_idx).value = h
    _style_header(ws, start_row, n_cols)
    current_row = start_row + 1

    for row_idx, (_, row) in enumerate(df.iterrows()):
        alt = (row_idx % 2 == 0)
        _style_data_row(ws, current_row, n_cols, alt=alt)
        for col_idx, value in enumerate(row, start=1):
            _write_value(ws.cell(row=current_row, column=col_idx), value)
        if color_p:
            _color_p_values(ws, current_row, headers)
        current_row += 1

    return current_row + 1


# ---------------------------------------------------------------------------
# Sheet writers
# ---------------------------------------------------------------------------
def _write_sheet_toc(ws, df: pd.DataFrame) -> None:
    ws.sheet_view.showGridLines = False
    if df is None or df.empty:
        return
    headers = list(df.columns)
    n_cols  = len(headers)
    for col_idx, h in enumerate(headers, start=1):
        ws.cell(row=1, column=col_idx).value = h
    _style_header(ws, 1, n_cols)
    for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
        alt = (row_idx % 2 == 0)
        _style_data_row(ws, row_idx, n_cols, alt=alt)
        for col_idx, value in enumerate(row, start=1):
            cell       = ws.cell(row=row_idx, column=col_idx)
            cell.value = str(value) if value else ""
            cell.font  = Font(size=10)
    _autofit(ws, min_w=20, max_w=80)


def _write_sheet_l(ws, df: pd.DataFrame) -> None:
    """Write Variable Definitions sheet — grouped by category."""
    categories = {
        "Experimental Design":      ["tone"],
        "Chatbot Evaluation":       ["E1","E2","E3","E4","E5","E6"],
        "Perceived Manipulation":   ["PM1","PM2","PM3","PM4","PM_score"],
        "Competence to Judge":      ["Comp1","Comp2","Comp_score"],
        "Moral Agency":             ["MA1","MA2","MA_score"],
        "Moral Patiency":           ["MP1","MP2","MP_score"],
        "Perceived Autonomy":       ["Ind1","Ind2","Ind_score"],
        "Perceived Personality":    ["PP1","PP2","PP3","PP4","PP5"],
        "Feedback Quality (GPT-4o)": [
            "quantity_run1/2/3","quantity_mean",
            "quality_run1/2/3","quality_mean",
            "emotions_run1/2/3","emotions_mean","composite",
        ],
        "Engagement": [
            "avg_words_per_turn","chat_duration_sec",
            "z_avg_words","z_duration","engagement_score",
            "conversation_completed","total_user_words","num_turns",
        ],
        "Demographics": ["age","gender","language"],
    }
    cat_colors = {
        "Experimental Design":       "1F618D",
        "Chatbot Evaluation":        "784212",
        "Perceived Manipulation":    "1A5276",
        "Competence to Judge":       "1A5276",
        "Moral Agency":              "1A5276",
        "Moral Patiency":            "1A5276",
        "Perceived Autonomy":        "1A5276",
        "Perceived Personality":     "7B241C",
        "Feedback Quality (GPT-4o)": "1D6A39",
        "Engagement":                "4A235A",
        "Demographics":              "424949",
    }
    headers  = list(df.columns)
    n_cols   = len(headers)
    curr_row = 1

    for cat_name, var_list in categories.items():
        curr_row = _write_section_title(
            ws, curr_row, cat_name, n_cols, primary=True
        )
        for col in range(1, n_cols + 1):
            ws.cell(row=curr_row-1, column=col).fill = _fill(
                cat_colors.get(cat_name, "1B4F72")
            )
        for col_idx, h in enumerate(headers, start=1):
            ws.cell(row=curr_row, column=col_idx).value = h
        _style_header(ws, curr_row, n_cols)
        curr_row += 1

        cat_df = df[df["Variable"].isin(var_list)].copy()
        for row_idx, (_, row) in enumerate(cat_df.iterrows()):
            alt = (row_idx % 2 == 0)
            _style_data_row(ws, curr_row, n_cols, alt=alt)
            for col_idx, value in enumerate(row, start=1):
                _write_value(ws.cell(row=curr_row, column=col_idx), value)
            curr_row += 1
        curr_row += 1

    _autofit(ws, min_w=15, max_w=60)
    _freeze(ws)


def _write_sheet_k(ws, df: pd.DataFrame) -> None:
    """Write Cleaned Data sheet with vivid color-coded headers."""
    def _header_color(col: str) -> str:
        if col.startswith(("PM","Comp","Ind","MA","MP")):
            return "1A5276"
        if col.startswith("PP"):
            return "7B241C"
        if col.startswith("E") and col[1:].isdigit():
            return "784212"
        if col in ["composite","quantity_mean","quality_mean",
                   "emotions_mean","quantity_run1","quantity_run2",
                   "quantity_run3","quality_run1","quality_run2",
                   "quality_run3","emotions_run1","emotions_run2",
                   "emotions_run3","quantity_sd","quality_sd","emotions_sd"]:
            return "1D6A39"
        if col in ["engagement_score","z_avg_words","z_duration",
                   "avg_words_per_turn","chat_duration_sec",
                   "num_turns","total_user_words","conversation_completed"]:
            return "4A235A"
        if col in ["age","gender","language"]:
            return "424949"
        if col in ["tone","tone_raw"]:
            return "1F618D"
        return "1B4F72"

    if df is None or df.empty:
        ws.cell(row=1, column=1).value = "No data."
        return

    headers = list(df.columns)
    n_cols  = len(headers)

    for col_idx, h in enumerate(headers, start=1):
        cell           = ws.cell(row=1, column=col_idx)
        cell.value     = h
        cell.fill      = _fill(_header_color(h))
        cell.font      = Font(bold=True, color="FFFFFF", size=9)
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )
    ws.row_dimensions[1].height = 35

    for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
        alt = (row_idx % 2 == 0)
        _style_data_row(ws, row_idx, n_cols, alt=alt)
        for col_idx, value in enumerate(row, start=1):
            _write_value(ws.cell(row=row_idx, column=col_idx), value)

    _autofit(ws, min_w=8, max_w=30)
    _freeze(ws)


def _write_sheet_b(ws, data: dict) -> None:
    """Write Effect of Tone sheet — recap + 5 blocks."""
    block_configs = [
        ("recap",       "RECAP — Significant results only",        True),
        ("ai",          "AI Perceptions",                          False),
        ("personality", "Perceived Personality",                   False),
        ("eval",        "Chatbot Evaluation",                      False),
        ("quality",     "Quality & Engagement",                    False),
        ("h3",          "H3 — Paired t-test: Comp1 vs Comp2",     False),
    ]
    n_cols = max(
        len(v.columns) if isinstance(v, pd.DataFrame) and not v.empty else 1
        for v in data.values()
    ) if data else 10
    current_row = 1
    for key, title, primary in block_configs:
        df_section  = data.get(key, pd.DataFrame())
        current_row = _write_section_title(
            ws, current_row, title, n_cols, primary=primary
        )
        current_row = _write_df(ws, df_section, current_row)
    _autofit(ws)
    _freeze(ws)


def _write_sheet_c(ws, data: dict) -> None:
    """Write AI Perception Regressions — with IV Label + DV Label."""
    recap = data.get("recap", pd.DataFrame())
    full  = data.get("full",  pd.DataFrame())

    def _add_labels(df):
        if df.empty or "note" in df.columns:
            return df
        df = df.copy()
        if "IV" in df.columns:
            df["IV_clean"] = df["IV"].str.replace(
                r"\s*\(H4\)", "", regex=True
            ).str.strip()
            idx = df.columns.get_loc("IV") + 1
            df.insert(idx, "IV Label",
                      df["IV_clean"].map(IV_LABELS).fillna(df["IV_clean"]))
        if "DV" in df.columns:
            idx = df.columns.get_loc("DV") + 1
            df.insert(idx, "DV Label",
                      df["DV"].map(DV_LABELS_REGRESSION).fillna(df["DV"]))
        df = df.drop(
            columns=[c for c in ["Scale","IV_clean"] if c in df.columns]
        )
        return df

    recap = _add_labels(recap)
    full  = _add_labels(full)

    n_cols      = max(
        len(full.columns)  if not full.empty  else 1,
        len(recap.columns) if not recap.empty else 1,
    )
    current_row = 1

    current_row = _write_section_title(
        ws, current_row,
        "RECAP — Significant results only",
        n_cols, primary=True
    )
    current_row = _write_df(ws, recap, current_row)

    current_row = _write_section_title(
        ws, current_row,
        "FULL RESULTS — All a priori regressions",
        n_cols, primary=False
    )
    current_row = _write_df(ws, full, current_row)

    _autofit(ws)
    _freeze(ws)


def _write_hierarchical_sheet(
    ws, data: dict, dvs: list, sheet_title: str
) -> None:
    """Hierarchical regression sheet with subtables per DV."""
    full  = data.get("full",  pd.DataFrame())
    recap = data.get("recap", pd.DataFrame())

    def _add_labels(df):
        if df.empty or "note" in df.columns:
            return df
        df = df.copy()
        if "Predictor" in df.columns:
            idx = df.columns.get_loc("Predictor") + 1
            df.insert(idx, "Label",
                      df["Predictor"].map(PREDICTOR_LABELS).fillna(
                          df["Predictor"]
                      ))
        df = df.drop(
            columns=[c for c in ["Block","Scale"] if c in df.columns]
        )
        return df

    full  = _add_labels(full)
    recap = _add_labels(recap)

    n_cols      = max(
        len(full.columns)  if not full.empty  else 1,
        len(recap.columns) if not recap.empty else 1,
    )
    current_row = 1

    current_row = _write_section_title(
        ws, current_row,
        "RECAP — Significant results only",
        n_cols, primary=True
    )
    current_row = _write_df(ws, recap, current_row)

    current_row = _write_section_title(
        ws, current_row, "FULL RESULTS",
        n_cols, primary=True
    )

    for dv in dvs:
        if full.empty or "DV" not in full.columns:
            break
        dv_label = DV_LABELS.get(dv, dv)
        df_dv    = full[full["DV"] == dv].copy()
        if df_dv.empty:
            continue
        current_row = _write_section_title(
            ws, current_row,
            f"DV = {dv_label}",
            n_cols, primary=False
        )
        df_dv       = df_dv.drop(columns=["DV"], errors="ignore")
        current_row = _write_df(ws, df_dv, current_row)

    _autofit(ws)
    _freeze(ws)


def _write_sheet_d(ws, data: dict) -> None:
    _write_hierarchical_sheet(
        ws, data,
        dvs=["composite", "engagement_score", "emotions_mean"],
        sheet_title="Predictors of Feedback Quality",
    )


def _write_sheet_e(ws, data: dict) -> None:
    _write_hierarchical_sheet(
        ws, data,
        dvs=["E3", "E4", "E5", "E6"],
        sheet_title="Predictors of Chatbot Evaluation",
    )


def _write_sheet_f(ws, data: dict) -> None:
    """Mediation Analyses — subtables per mediator."""
    full  = data.get("full",  pd.DataFrame())
    recap = data.get("recap", pd.DataFrame())

    if full.empty or "note" in full.columns:
        ws.cell(row=1, column=1).value = (
            "Run GPT scoring first to enable full mediation analyses."
        )
        return

    n_cols      = len(full.columns)
    current_row = 1

    current_row = _write_section_title(
        ws, current_row,
        "RECAP — Significant results only",
        n_cols, primary=True
    )
    current_row = _write_df(ws, recap, current_row)

    for series_val, series_label in [
        ("1", "SERIES 1 — Tone as IV"),
        ("2", "SERIES 2 — AI Perceptions as IV"),
    ]:
        df_series = full[full["Series"] == series_val].copy()
        if df_series.empty:
            continue
        current_row = _write_section_title(
            ws, current_row, series_label,
            n_cols, primary=True
        )
        for med in df_series["Mediator"].unique():
            df_med = df_series[df_series["Mediator"] == med].copy()
            if df_med.empty:
                continue
            med_label   = PREDICTOR_LABELS.get(med, med)
            current_row = _write_section_title(
                ws, current_row,
                f"Mediator = {med_label}",
                n_cols, primary=False
            )
            current_row = _write_df(ws, df_med, current_row)

    _autofit(ws)
    _freeze(ws)


def _write_sheet_g(ws, data) -> None:
    """Write GPT Scoring sheet."""
    if isinstance(data, dict):
        sheet_g   = data.get("scores",  pd.DataFrame())
        summary_g = data.get("summary", pd.DataFrame())
    else:
        sheet_g   = data
        summary_g = pd.DataFrame()

    current_row = 1
    n_cols      = max(
        len(sheet_g.columns)   if not sheet_g.empty   else 1,
        len(summary_g.columns) if not summary_g.empty else 1,
    )

    if not summary_g.empty:
        current_row = _write_section_title(
            ws, current_row,
            "SUMMARY BY TONE CONDITION",
            n_cols, primary=True
        )
        current_row = _write_df(ws, summary_g, current_row, color_p=False)

    current_row = _write_section_title(
        ws, current_row,
        "FULL SCORES — All participants",
        n_cols, primary=False
    )
    current_row = _write_df(ws, sheet_g, current_row, color_p=False)

    _autofit(ws)
    _freeze(ws)


def _write_sheet_h(ws, data: dict) -> None:
    """Word Frequencies — two tables side by side + wordcloud note."""
    if not data:
        ws.cell(row=1, column=1).value = "No word frequency results."
        return

    freq_table  = data.get("freq_table",  pd.DataFrame())
    tfidf_table = data.get("tfidf_table", pd.DataFrame())
    wc_paths    = data.get("wordcloud_paths", [])

    n_cols      = max(
        len(freq_table.columns)  if not freq_table.empty  else 1,
        len(tfidf_table.columns) if not tfidf_table.empty else 1,
    )
    current_row = 1

    current_row = _write_section_title(
        ws, current_row,
        "TABLE 1 — Word Frequencies (top 50 per condition)",
        n_cols, primary=True
    )
    current_row = _write_df(ws, freq_table, current_row, color_p=False)

    current_row = _write_section_title(
        ws, current_row,
        "TABLE 2 — Delta TF-IDF "
        "(positive = more distinctive in Friendly / "
        "negative = more distinctive in Professional)",
        n_cols, primary=True
    )
    current_row = _write_df(ws, tfidf_table, current_row, color_p=False)

    if wc_paths:
        note        = ws.cell(row=current_row, column=1)
        note.value  = (
            f"Wordcloud PNG files: "
            f"{', '.join(os.path.basename(p) for p in wc_paths)} "
            f"— saved in outputs/wordclouds/"
        )
        note.font   = Font(italic=True, color="555555", size=10)

    _autofit(ws)


def _write_sheet_i(ws, data: dict) -> None:
    """Write Demographics & Robustness sheet."""
    current_row = 1
    n_cols      = 8

    sections = [
        ("DESCRIPTIVE STATISTICS",            "descriptives", False),
        ("ANCOVA — Recap (significant only)",  "ancova_recap", True),
        ("ANCOVA — Full results",              "ancova",       True),
        ("INTERACTIONS — Recap",               "inter_recap",  True),
        ("INTERACTIONS — Full results",        "interactions", True),
    ]

    for title, key, primary in sections:
        current_row = _write_section_title(
            ws, current_row, title, n_cols, primary=primary
        )
        df_section  = data.get(key, pd.DataFrame())
        current_row = _write_df(ws, df_section, current_row)

    _autofit(ws)
    _freeze(ws)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def write_excel(
    results: dict,
    sheet_order: list,
    output_path: str,
) -> None:
    wb = Workbook()
    wb.remove(wb.active)

    sheet_names = {
        "J": "Table of Contents",
        "L": "Variable Definitions",
        "K": "Cleaned Data",
        "A": "Correlations",
        "B": "Effect of Tone",
        "C": "AI Perception Regressions",
        "D": "Predictors of Feedback Quality",
        "E": "Predictors of Chatbot Evaluation",
        "F": "Mediation Analyses",
        "G": "GPT Scoring",
        "H": "Word Frequencies",
        "I": "Demographics & Robustness",
    }

    for key in sheet_order:
        if key not in results:
            log.warning(f"Sheet {key}: no data — skipped.")
            continue

        name = sheet_names.get(key, key)
        ws   = wb.create_sheet(title=name)
        data = results[key]
        log.info(f"Writing: {name}")

        if key == "J":
            _write_sheet_toc(ws, data)
        elif key == "L":
            _write_sheet_l(ws, data)
        elif key == "K":
            _write_sheet_k(ws, data)
        elif key == "A":
            current_row = 1
            n_cols      = 12
            current_row = _write_section_title(
                ws, current_row,
                "RECAP — Significant correlations only (FDR corrected)",
                n_cols, primary=True
            )
            current_row = _write_df(
                ws, data.get("recap", pd.DataFrame()), current_row
            )
            current_row = _write_section_title(
                ws, current_row,
                "FULL RESULTS — All tested pairs",
                n_cols, primary=False
            )
            _write_df(ws, data.get("full", pd.DataFrame()), current_row)
            _autofit(ws)
            _freeze(ws)
        elif key == "B":
            _write_sheet_b(ws, data)
        elif key == "C":
            _write_sheet_c(ws, data if isinstance(data, dict) else {})
        elif key == "D":
            _write_sheet_d(ws, data if isinstance(data, dict) else {})
        elif key == "E":
            _write_sheet_e(ws, data if isinstance(data, dict) else {})
        elif key == "F":
            _write_sheet_f(ws, data if isinstance(data, dict) else {})
        elif key == "G":
            _write_sheet_g(ws, data)
        elif key == "H":
            _write_sheet_h(ws, data if isinstance(data, dict) else {})
        elif key == "I":
            _write_sheet_i(ws, data if isinstance(data, dict) else {})
        else:
            if isinstance(data, pd.DataFrame):
                _write_df(ws, data, start_row=1)
            else:
                ws.cell(row=1, column=1).value = f"No handler for {key}."

        _autofit(ws)

    try:
        wb.save(output_path)
        log.info(f"Saved: {output_path}")
    except Exception as e:
        log.error(f"Failed to save: {e}")
        raise
