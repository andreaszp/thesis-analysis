"""
export.py
---------
Final step: write all results to a single formatted Excel file.

Sheet order:
    Table of Contents → Variable Definitions → Cleaned Data →
    Correlations → Effect of Tone → AI Perception Regressions →
    Predictors of Feedback Quality → Predictors of Chatbot Evaluation →
    Mediation Analyses → GPT Scoring → Word Frequencies →
    Demographics & Robustness

Each analysis sheet has:
    - A RECAP table at the top (significant results only)
    - The full results table below
"""

import logging
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fills and styles
# ---------------------------------------------------------------------------
def _fill(hex_color: str) -> PatternFill:
    return PatternFill(
        start_color=hex_color,
        end_color=hex_color,
        fill_type="solid",
    )

FILLS = {k: _fill(v) for k, v in config.COLORS.items()}

THIN_BORDER = Border(
    bottom=Side(style="thin", color="CCCCCC")
)

BLOCK_FILLS = {
    "AI Perceptions":        FILLS["block_ai"],
    "Perceived Personality": FILLS["block_personality"],
    "Chatbot Evaluation":    FILLS["block_eval"],
    "Quality & Engagement":  FILLS["block_quality"],
    "H3":                    FILLS["block_ai"],
    "DEMOGRAPHICS":          FILLS["block_demo"],
    "Descriptives":          FILLS["block_demo"],
    "ANCOVA":                FILLS["block_engage"],
}


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
def _write_value(cell, value) -> None:
    """Write a value to a cell, converting numpy types."""
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
                  bg: str = None) -> None:
    """Style a header row."""
    bg = bg or config.COLORS["header_bg"]
    for col in range(1, n_cols + 1):
        cell       = ws.cell(row=row_num, column=col)
        cell.fill  = _fill(bg)
        cell.font  = Font(
            bold=True,
            color=config.COLORS["header_font"],
            size=10,
        )
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )
        cell.border = THIN_BORDER
    ws.row_dimensions[row_num].height = 30


def _style_data_row(ws, row_num: int, n_cols: int,
                    alt: bool = False,
                    block_fill: PatternFill = None,
                    highlight: bool = False) -> None:
    for col in range(1, n_cols + 1):
        cell           = ws.cell(row=row_num, column=col)
        cell.font      = Font(size=10)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        if highlight:
            cell.fill = FILLS["sig_highlight"]
        elif block_fill:
            cell.fill = block_fill
        elif alt:
            cell.fill = FILLS["alt_row"]


def _autofit(ws, min_w: int = 10, max_w: int = 45) -> None:
    for col in ws.columns:
        max_len = 0
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


def _is_sig(row: pd.Series) -> bool:
    if "Sig." in row.index:
        if "*" in str(row["Sig."]):
            return True
    for p_col in ["p_fdr", "p", "p-value"]:
        if p_col in row.index:
            try:
                if float(row[p_col]) < config.ALPHA:
                    return True
            except (ValueError, TypeError):
                pass
    return False


# ---------------------------------------------------------------------------
# Section title helper
# ---------------------------------------------------------------------------
def _write_section_title(ws, row_num: int, title: str,
                          n_cols: int, color: str = "1F3864") -> int:
    """Write a bold section title spanning all columns. Returns next row."""
    cell       = ws.cell(row=row_num, column=1)
    cell.value = title
    cell.font  = Font(bold=True, size=11, color="FFFFFF")
    cell.fill  = _fill(color)
    cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[row_num].height = 22
    # Blank out remaining cells in the row
    for col in range(2, n_cols + 1):
        ws.cell(row=row_num, column=col).fill = _fill(color)
    return row_num + 1


# ---------------------------------------------------------------------------
# Generic DataFrame writer
# ---------------------------------------------------------------------------
def _write_df(ws, df: pd.DataFrame, start_row: int,
              use_sig: bool = True,
              block_col: str = None) -> int:
    """
    Write a DataFrame to ws starting at start_row.
    Returns the next available row after writing.
    """
    if df is None or df.empty:
        cell       = ws.cell(row=start_row, column=1)
        cell.value = "No results."
        cell.font  = Font(italic=True, color="888888")
        return start_row + 2

    headers = list(df.columns)
    n_cols  = len(headers)

    # Headers
    for col_idx, h in enumerate(headers, start=1):
        ws.cell(row=start_row, column=col_idx).value = h
    _style_header(ws, start_row, n_cols)
    current_row = start_row + 1

    # Data
    for row_idx, (_, row) in enumerate(df.iterrows()):
        alt        = (row_idx % 2 == 0)
        highlight  = use_sig and _is_sig(row)
        block_fill = None
        if block_col and block_col in row.index:
            block_fill = BLOCK_FILLS.get(str(row[block_col]))

        _style_data_row(
            ws, current_row, n_cols,
            alt=alt, block_fill=block_fill, highlight=highlight
        )
        for col_idx, value in enumerate(row, start=1):
            _write_value(ws.cell(row=current_row, column=col_idx), value)
        current_row += 1

    return current_row + 1  # blank row after table


# ---------------------------------------------------------------------------
# Sheet writers
# ---------------------------------------------------------------------------
def _write_recap_then_full(
    ws,
    data: dict,
    recap_key: str,
    full_sections: list,
    n_cols_hint: int = 15,
) -> None:
    """
    Write a recap table at the top, then full result sections below.

    Args:
        ws:             Worksheet
        data:           Dict of DataFrames
        recap_key:      Key for the recap DataFrame in data
        full_sections:  List of (title, key, block_col) tuples
        n_cols_hint:    Approximate column count for section titles
    """
    current_row = 1

    # --- RECAP ---
    current_row = _write_section_title(
        ws, current_row,
        "RECAP — Significant results only",
        n_cols_hint, color="1F3864"
    )
    recap = data.get(recap_key, pd.DataFrame())
    current_row = _write_df(ws, recap, current_row, use_sig=True)

    # --- FULL RESULTS ---
    for title, key, block_col in full_sections:
        current_row = _write_section_title(
            ws, current_row, title, n_cols_hint, color="2E75B6"
        )
        df_section = data.get(key, pd.DataFrame())
        current_row = _write_df(
            ws, df_section, current_row,
            use_sig=True, block_col=block_col
        )

    _autofit(ws)
    _freeze(ws)


def _write_sheet_toc(ws, toc_data: pd.DataFrame) -> None:
    """Write Table of Contents sheet."""
    ws.sheet_view.showGridLines = False
    headers = list(toc_data.columns)
    n_cols  = len(headers)

    for col_idx, h in enumerate(headers, start=1):
        ws.cell(row=1, column=col_idx).value = h
    _style_header(ws, 1, n_cols, bg="1F3864")

    for row_idx, (_, row) in enumerate(toc_data.iterrows(), start=2):
        alt = (row_idx % 2 == 0)
        _style_data_row(ws, row_idx, n_cols, alt=alt)
        for col_idx, value in enumerate(row, start=1):
            cell       = ws.cell(row=row_idx, column=col_idx)
            cell.value = str(value) if value else ""
            cell.font  = Font(size=10)

    _autofit(ws, min_w=20, max_w=80)


def _write_sheet_l(ws, df: pd.DataFrame) -> None:
    """Write Variable Definitions sheet."""
    headers = list(df.columns)
    n_cols  = len(headers)

    for col_idx, h in enumerate(headers, start=1):
        ws.cell(row=1, column=col_idx).value = h
    _style_header(ws, 1, n_cols)

    for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
        alt = (row_idx % 2 == 0)
        _style_data_row(ws, row_idx, n_cols, alt=alt)
        for col_idx, value in enumerate(row, start=1):
            _write_value(ws.cell(row=row_idx, column=col_idx), value)

    _autofit(ws, min_w=15, max_w=60)
    _freeze(ws)


def _write_sheet_k(ws, df: pd.DataFrame) -> None:
    """Write Cleaned Data sheet with color-coded headers by variable block."""

    def _header_color(col: str) -> str:
        if col.startswith("PM"):
            return config.COLORS["block_ai"]
        if col.startswith("Comp"):
            return config.COLORS["block_ai"]
        if col.startswith(("Ind","MA","MP")):
            return config.COLORS["block_ai"]
        if col.startswith("PP"):
            return config.COLORS["block_personality"]
        if col.startswith("E") and col[1:].isdigit():
            return config.COLORS["block_eval"]
        if col in ["composite","quantity_mean","quality_mean","emotions_mean",
                   "quantity_run1","quantity_run2","quantity_run3",
                   "quality_run1","quality_run2","quality_run3",
                   "emotions_run1","emotions_run2","emotions_run3"]:
            return config.COLORS["block_quality"]
        if col in ["engagement_score","z_avg_words","z_duration",
                   "avg_words_per_turn","chat_duration_sec",
                   "num_turns","total_user_words","conversation_completed"]:
            return config.COLORS["block_engage"]
        if col in ["age","gender","language"]:
            return config.COLORS["block_demo"]
        return config.COLORS["header_bg"]

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
    _write_recap_then_full(
        ws, data,
        recap_key="recap",
        full_sections=[
            ("AI Perceptions",        "ai",          "Block"),
            ("Perceived Personality", "personality", "Block"),
            ("Chatbot Evaluation",    "eval",        "Block"),
            ("Quality & Engagement",  "quality",     "Block"),
            ("H3 — Paired t-test: Comp1 vs Comp2", "h3", "Block"),
        ],
        n_cols_hint=15,
    )


def _write_sheet_g(ws, data) -> None:
    """Write GPT Scoring sheet."""
    if isinstance(data, dict):
        sheet_g   = data.get("scores",  pd.DataFrame())
        recap_g   = data.get("recap",   pd.DataFrame())
        summary_g = data.get("summary", pd.DataFrame())
    else:
        sheet_g   = data
        recap_g   = pd.DataFrame()
        summary_g = pd.DataFrame()

    current_row = 1
    n_cols = max(
        len(sheet_g.columns)   if not sheet_g.empty   else 1,
        len(summary_g.columns) if not summary_g.empty else 1,
    )

    # Summary by condition
    if not summary_g.empty:
        current_row = _write_section_title(
            ws, current_row,
            "SUMMARY BY TONE CONDITION",
            n_cols, color="1F3864"
        )
        current_row = _write_df(ws, summary_g, current_row, use_sig=False)

    # Full scores
    current_row = _write_section_title(
        ws, current_row,
        "FULL SCORES — All participants",
        n_cols, color="2E75B6"
    )
    current_row = _write_df(ws, sheet_g, current_row, use_sig=False)

    _autofit(ws)
    _freeze(ws)


def _write_sheet_h(ws, data: dict) -> None:
    """Write Word Frequencies sheet — 4 sections × freq + TF-IDF."""
    if not data:
        ws.cell(row=1, column=1).value = "No word frequency results."
        return

    section_labels = {
        "participant_friendly_freq":  "PARTICIPANT MESSAGES — Friendly condition — Word Frequencies",
        "participant_friendly_tfidf": "PARTICIPANT MESSAGES — Friendly condition — TF-IDF",
        "participant_pro_freq":       "PARTICIPANT MESSAGES — Professional condition — Word Frequencies",
        "participant_pro_tfidf":      "PARTICIPANT MESSAGES — Professional condition — TF-IDF",
        "chatbot_friendly_freq":      "CHATBOT MESSAGES — Friendly condition — Word Frequencies",
        "chatbot_friendly_tfidf":     "CHATBOT MESSAGES — Friendly condition — TF-IDF",
        "chatbot_pro_freq":           "CHATBOT MESSAGES — Professional condition — Word Frequencies",
        "chatbot_pro_tfidf":          "CHATBOT MESSAGES — Professional condition — TF-IDF",
    }

    ordered_keys = [
        "participant_friendly_freq",  "participant_friendly_tfidf",
        "participant_pro_freq",       "participant_pro_tfidf",
        "chatbot_friendly_freq",      "chatbot_friendly_tfidf",
        "chatbot_pro_freq",           "chatbot_pro_tfidf",
    ]

    current_row = 1
    n_cols      = 6

    for key in ordered_keys:
        if key not in data:
            continue
        df_section = data[key]
        if df_section is None or df_section.empty:
            continue

        label       = section_labels.get(key, key)
        current_row = _write_section_title(
            ws, current_row, label, n_cols, color="1F3864"
        )
        current_row = _write_df(
            ws, df_section, current_row, use_sig=False
        )

    _autofit(ws)


def _write_sheet_i(ws, data: dict) -> None:
    """Write Demographics & Robustness sheet."""
    current_row = 1
    n_cols      = 8

    sections = [
        ("DESCRIPTIVE STATISTICS",           "descriptives",   None,    False),
        ("ANCOVA — Recap (significant only)", "ancova_recap",   "Sig.",  True),
        ("ANCOVA — Full results",             "ancova",         "Sig.",  True),
        ("INTERACTIONS — Recap",              "inter_recap",    "Sig.",  True),
        ("INTERACTIONS — Full results",       "interactions",   "Sig.",  True),
    ]

    for title, key, _, use_sig in sections:
        current_row = _write_section_title(
            ws, current_row, title, n_cols,
            color="1F3864" if "Recap" in title or "DESCRIPTIVE" in title
            else "2E75B6"
        )
        df_section  = data.get(key, pd.DataFrame())
        current_row = _write_df(
            ws, df_section, current_row, use_sig=use_sig
        )

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
    """
    Write all analysis results to a single Excel file.

    Args:
        results:     Dict mapping sheet key → data
        sheet_order: Ordered list of sheet keys
        output_path: Output .xlsx path
    """
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

        elif key == "B":
            _write_sheet_b(ws, data)

        elif key == "G":
            _write_sheet_g(ws, data)

        elif key == "H":
            _write_sheet_h(ws, data if isinstance(data, dict) else {})

        elif key == "I":
            _write_sheet_i(ws, data if isinstance(data, dict) else {})

        elif key in ("A", "C"):
            # recap + full
            _write_recap_then_full(
                ws, data,
                recap_key="recap",
                full_sections=[("Full results", "full", None)],
                n_cols_hint=12,
            )

        elif key in ("D", "E"):
            _write_recap_then_full(
                ws, data,
                recap_key="recap",
                full_sections=[("Full results", "full", None)],
                n_cols_hint=13,
            )

        elif key == "F":
            _write_recap_then_full(
                ws, data,
                recap_key="recap",
                full_sections=[("Full results", "full", None)],
                n_cols_hint=16,
            )

        else:
            # Fallback
            if isinstance(data, pd.DataFrame):
                _write_df(ws, data, start_row=1)
            else:
                ws.cell(row=1, column=1).value = f"No handler for sheet {key}."

        _autofit(ws)

    try:
        wb.save(output_path)
        log.info(f"Saved: {output_path}")
    except Exception as e:
        log.error(f"Failed to save: {e}")
        raise
