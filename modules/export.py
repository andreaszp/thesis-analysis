"""
export.py
---------
Final step of the pipeline: write all analysis results to a
single well-formatted Excel file.

Responsibilities:
    - Write each sheet in the correct order (L→K→J→A→B→C→D→E→F→G→H→I)
    - Apply consistent formatting: bold headers, color blocks,
      highlighted significant results, alternating row colors
    - Handle Sheet H (dict of sub-tables) as labelled sections
    - Auto-fit column widths for readability
"""

import logging
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color fills (pre-built from config.COLORS)
# ---------------------------------------------------------------------------
def _fill(hex_color: str) -> PatternFill:
    return PatternFill(
        start_color=hex_color,
        end_color=hex_color,
        fill_type="solid",
    )

FILLS = {k: _fill(v) for k, v in config.COLORS.items()}

# Thin border for header row
THIN_BORDER = Border(
    bottom=Side(style="thin", color="AAAAAA")
)

# ---------------------------------------------------------------------------
# Block color mapping — which fill to use per variable block
# ---------------------------------------------------------------------------
BLOCK_FILLS = {
    "AI Perceptions":      FILLS["block_ai"],
    "Perceived Personality": FILLS["block_personality"],
    "Chatbot Evaluation":  FILLS["block_eval"],
    "Quality & Engagement": FILLS["block_quality"],
    "H3 — Hypothesis test": FILLS["block_ai"],
    "DEMOGRAPHICS":        FILLS["block_demo"],
    "Descriptives":        FILLS["block_demo"],
    "ANCOVA":              FILLS["block_engage"],
}


# ---------------------------------------------------------------------------
# Core formatting helpers
# ---------------------------------------------------------------------------
def _style_header_row(ws, row_num: int, n_cols: int) -> None:
    """Apply dark navy background + white bold font to a header row."""
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=row_num, column=col)
        cell.fill = FILLS["header_bg"]
        cell.font = Font(
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


def _style_data_row(
    ws,
    row_num: int,
    n_cols: int,
    alt: bool = False,
    block_fill: PatternFill = None,
    highlight: bool = False,
) -> None:
    """
    Style a data row:
        - Significant results → yellow highlight
        - Block color if provided
        - Alternating light blue rows otherwise
    """
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=row_num, column=col)
        cell.font = Font(size=10)
        cell.alignment = Alignment(vertical="center", wrap_text=True)

        if highlight:
            cell.fill = FILLS["sig_highlight"]
        elif block_fill:
            cell.fill = block_fill
        elif alt:
            cell.fill = FILLS["alt_row"]


def _autofit_columns(ws, min_width: int = 10, max_width: int = 45) -> None:
    """Auto-fit column widths based on content length."""
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                cell_len = len(str(cell.value)) if cell.value else 0
                max_len  = max(max_len, cell_len)
            except Exception:
                pass
        adjusted = min(max(max_len + 2, min_width), max_width)
        ws.column_dimensions[col_letter].width = adjusted


def _freeze_header(ws) -> None:
    """Freeze the first row (header)."""
    ws.freeze_panes = "A2"


def _is_significant(row: pd.Series) -> bool:
    """
    Return True if the row contains a significant result.
    Checks 'Sig.' column for *, **, *** or 'p_fdr'/'p' < 0.05.
    """
    if "Sig." in row.index:
        sig = str(row["Sig."])
        if any(s in sig for s in ["*"]):
            return True
    for p_col in ["p_fdr", "p"]:
        if p_col in row.index:
            try:
                p_val = float(row[p_col])
                if p_val < config.ALPHA:
                    return True
            except (ValueError, TypeError):
                pass
    return False


# ---------------------------------------------------------------------------
# Generic sheet writer
# ---------------------------------------------------------------------------
def _write_sheet(
    ws,
    df: pd.DataFrame,
    block_col: str = "Block",
    use_sig_highlight: bool = True,
    use_block_colors: bool = False,
) -> None:
    """
    Write a DataFrame to an openpyxl worksheet with full formatting.

    Args:
        ws:                 openpyxl Worksheet object
        df:                 DataFrame to write
        block_col:          Column name used to determine block fill color
        use_sig_highlight:  Highlight significant rows in yellow
        use_block_colors:   Apply block-level background colors
    """
    if df is None or df.empty:
        ws.cell(row=1, column=1).value = "No results generated."
        return

    headers = list(df.columns)
    n_cols  = len(headers)

    # Write header row
    for col_idx, header in enumerate(headers, start=1):
        ws.cell(row=1, column=col_idx).value = header
    _style_header_row(ws, row_num=1, n_cols=n_cols)
    ws.row_dimensions[1].height = 30

    # Write data rows
    for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
        alt       = (row_idx % 2 == 0)
        highlight = use_sig_highlight and _is_significant(row)

        # Determine block fill
        block_fill = None
        if use_block_colors and block_col in row.index:
            block_name = str(row[block_col])
            block_fill = BLOCK_FILLS.get(block_name)

        _style_data_row(
            ws, row_idx, n_cols,
            alt=alt,
            block_fill=block_fill,
            highlight=highlight,
        )

        for col_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)
            # Convert numpy types to native Python for Excel compatibility
            if isinstance(value, (np.integer,)):
                cell.value = int(value)
            elif isinstance(value, (np.floating,)):
                cell.value = float(value) if not np.isnan(value) else None
            elif isinstance(value, float) and np.isnan(value):
                cell.value = None
            else:
                cell.value = value

    _autofit_columns(ws)
    _freeze_header(ws)


# ---------------------------------------------------------------------------
# Sheet H — special handler (dict of sub-tables)
# ---------------------------------------------------------------------------
def _write_sheet_h(ws, data: dict) -> None:
    """
    Write Sheet H (word frequencies + TF-IDF) as labelled sections
    within a single worksheet, separated by blank rows.

    Each key in the dict becomes a section header.
    """
    if not data:
        ws.cell(row=1, column=1).value = "No word frequency results generated."
        return

    section_labels = {
        "freq_FR":              "Word Frequencies — French responses",
        "freq_EN":              "Word Frequencies — English responses",
        "freq_tone_friendly":   "Word Frequencies — Friendly condition",
        "freq_tone_professional": "Word Frequencies — Professional condition",
        "tfidf_friendly_FR":    "TF-IDF — Friendly condition (FR)",
        "tfidf_pro_FR":         "TF-IDF — Professional condition (FR)",
        "tfidf_friendly_EN":    "TF-IDF — Friendly condition (EN)",
        "tfidf_pro_EN":         "TF-IDF — Professional condition (EN)",
    }

    current_row = 1

    for key, df_section in data.items():
        if df_section is None or df_section.empty:
            continue

        # Section title
        label = section_labels.get(key, key)
        title_cell = ws.cell(row=current_row, column=1)
        title_cell.value = label
        title_cell.font  = Font(bold=True, size=11, color="1F3864")
        title_cell.fill  = FILLS["block_engage"]
        current_row += 1

        # Column headers
        headers = list(df_section.columns)
        n_cols  = len(headers)
        for col_idx, header in enumerate(headers, start=1):
            ws.cell(row=current_row, column=col_idx).value = header
        _style_header_row(ws, row_num=current_row, n_cols=n_cols)
        current_row += 1

        # Data rows
        for row_idx_local, (_, row) in enumerate(df_section.iterrows()):
            alt = (row_idx_local % 2 == 0)
            _style_data_row(ws, current_row, n_cols, alt=alt)
            for col_idx, value in enumerate(row, start=1):
                cell = ws.cell(row=current_row, column=col_idx)
                if isinstance(value, (np.integer,)):
                    cell.value = int(value)
                elif isinstance(value, (np.floating,)):
                    cell.value = float(value) if not np.isnan(value) else None
                elif isinstance(value, float) and np.isnan(value):
                    cell.value = None
                else:
                    cell.value = value
            current_row += 1

        # Blank row between sections
        current_row += 1

    _autofit_columns(ws)


# ---------------------------------------------------------------------------
# Sheet K — cleaned data (special: no sig highlight, block colors by variable)
# ---------------------------------------------------------------------------
def _write_sheet_k(ws, df: pd.DataFrame) -> None:
    """
    Write the cleaned data sheet.
    Color-codes column headers by variable block for readability.
    No significance highlighting.
    """
    if df is None or df.empty:
        ws.cell(row=1, column=1).value = "No data."
        return

    # Map variable prefixes to header fill colors
    def _header_fill_for(col_name: str) -> PatternFill:
        if col_name.startswith("PM"):
            return FILLS["block_ai"]
        if col_name.startswith("Comp"):
            return FILLS["block_ai"]
        if col_name.startswith("Ind") or col_name.startswith("MA") or \
           col_name.startswith("MP"):
            return FILLS["block_ai"]
        if col_name.startswith("PP"):
            return FILLS["block_personality"]
        if col_name.startswith("E") and col_name[1:].isdigit():
            return FILLS["block_eval"]
        if col_name in ["composite", "quantity_mean", "quality_mean",
                        "emotions_mean"]:
            return FILLS["block_quality"]
        if col_name in ["engagement_score", "z_avg_words", "z_duration",
                        "avg_words_per_turn", "chat_duration_sec",
                        "num_turns", "total_user_words"]:
            return FILLS["block_engage"]
        if col_name in ["age", "gender", "language"]:
            return FILLS["block_demo"]
        return FILLS["header_bg"]

    headers = list(df.columns)
    n_cols  = len(headers)

    # Write headers with variable-specific colors
    for col_idx, header in enumerate(headers, start=1):
        cell      = ws.cell(row=1, column=col_idx)
        cell.value = header
        cell.fill  = _header_fill_for(header)
        cell.font  = Font(bold=True, color="FFFFFF", size=9)
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )

    ws.row_dimensions[1].height = 35

    # Write data rows (alternating, no highlight)
    for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
        alt = (row_idx % 2 == 0)
        _style_data_row(ws, row_idx, n_cols, alt=alt)
        for col_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)
            if isinstance(value, (np.integer,)):
                cell.value = int(value)
            elif isinstance(value, (np.floating,)):
                cell.value = float(value) if not np.isnan(value) else None
            elif isinstance(value, float) and np.isnan(value):
                cell.value = None
            elif isinstance(value, bool):
                cell.value = str(value)
            else:
                cell.value = str(value) if value is not None else None

    _autofit_columns(ws, min_width=8, max_width=30)
    _freeze_header(ws)


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
        results:     Dict mapping sheet key → DataFrame (or dict for Sheet H)
        sheet_order: List of sheet keys in desired order
        output_path: Path to write the .xlsx file
    """
    wb = Workbook()
    wb.remove(wb.active)  # remove default empty sheet

    # Sheet display names
    sheet_names = {
        "L": "L — Variable Definitions",
        "K": "K — Cleaned Data",
        "J": "J — Table of Contents",
        "A": "A — Correlations",
        "B": "B — Tone Effects",
        "C": "C — AI Perceptions",
        "D": "D — Feedback Quality",
        "E": "E — Chatbot Evaluation",
        "F": "F — Mediation",
        "G": "G — GPT Scoring",
        "H": "H — Word Frequencies",
        "I": "I — Demographics",
    }

    # Per-sheet formatting options
    sheet_config = {
        "L": {"use_sig_highlight": False, "use_block_colors": False},
        "K": {"special": "cleaned_data"},
        "J": {"use_sig_highlight": False, "use_block_colors": False},
        "A": {"use_sig_highlight": True,  "use_block_colors": False},
        "B": {"use_sig_highlight": True,  "use_block_colors": True,
              "block_col": "Block"},
        "C": {"use_sig_highlight": True,  "use_block_colors": False},
        "D": {"use_sig_highlight": True,  "use_block_colors": False},
        "E": {"use_sig_highlight": True,  "use_block_colors": False},
        "F": {"use_sig_highlight": True,  "use_block_colors": False},
        "G": {"use_sig_highlight": False, "use_block_colors": False},
        "H": {"special": "word_freq"},
        "I": {"use_sig_highlight": True,  "use_block_colors": True,
              "block_col": "Section"},
    }

    for key in sheet_order:
        if key not in results:
            log.warning(f"Sheet {key}: no results found — sheet skipped.")
            continue

        ws   = wb.create_sheet(title=sheet_names.get(key, key))
        data = results[key]
        cfg  = sheet_config.get(key, {})

        log.info(f"Writing sheet: {sheet_names.get(key, key)}")

        # Special handlers
        if cfg.get("special") == "cleaned_data":
            _write_sheet_k(ws, data)

        elif cfg.get("special") == "word_freq":
            _write_sheet_h(ws, data if isinstance(data, dict) else {})

        # Standard handler
        elif isinstance(data, pd.DataFrame):
            _write_sheet(
                ws, data,
                block_col=cfg.get("block_col", "Block"),
                use_sig_highlight=cfg.get("use_sig_highlight", True),
                use_block_colors=cfg.get("use_block_colors", False),
            )

        else:
            log.warning(
                f"Sheet {key}: unexpected data type "
                f"({type(data)}) — writing empty sheet."
            )
            ws.cell(row=1, column=1).value = f"No data for sheet {key}."

    # Save
    try:
        wb.save(output_path)
        log.info(f"Excel file saved: {output_path}")
    except Exception as e:
        log.error(f"Failed to save Excel file: {e}")
        raise
