"""
main.py
-------
Orchestration script for the SoundFlow thesis analysis pipeline.

Usage:
    python main.py
    python main.py --skip-gpt
    python main.py --data path/to.xlsx
    python main.py --gpt-only
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

import config
from modules.cleaning import load_and_clean
from modules.sheet_a_correlations import run_sheet_a
from modules.sheet_b_tone_effects import run_sheet_b
from modules.sheet_c_ai_perceptions import run_sheet_c
from modules.sheet_d_feedback_quality import run_sheet_d
from modules.sheet_e_chatbot_eval import run_sheet_e
from modules.sheet_f_mediation import run_sheet_f
from modules.sheet_g_gpt_scoring import run_sheet_g
from modules.sheet_h_word_freq import run_sheet_h
from modules.sheet_i_demographics import run_sheet_i
from modules.sheet_j_toc import run_sheet_j
from modules.sheet_l_variable_definitions import run_sheet_l
from modules.formatting import write_excel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SoundFlow thesis analysis pipeline"
    )
    parser.add_argument("--data", type=str, default=config.DATA_PATH)
    parser.add_argument("--output", type=str, default=config.OUTPUT_PATH)
    parser.add_argument("--skip-gpt", action="store_true")
    parser.add_argument("--gpt-only", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_api_key() -> None:
    if not config.OPENAI_API_KEY:
        log.warning(
            "OPENAI_API_KEY is not set. "
            "Sheet G (GPT scoring) will fail. "
            "Use --skip-gpt to bypass it."
        )

def ensure_output_dir(output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

def section(title: str) -> None:
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(data_path: str, output_path: str, skip_gpt: bool) -> None:

    pipeline_start = time.time()

    section("STEP 1 — Loading and cleaning data")
    if not os.path.exists(data_path):
        log.error(f"Data file not found: {data_path}")
        sys.exit(1)

    df = load_and_clean(data_path)
    log.info(f"Clean dataset: {len(df)} participants, {len(df.columns)} columns")

    results: dict = {}

    section("STEP 2 — Variable definitions (Sheet L)")
    results["L"] = run_sheet_l()

    section("STEP 3 — Cleaned data export (Sheet K)")
    results["K"] = df.copy()

    section("STEP 4 — Correlation matrix (Sheet A)")
    results["A"] = run_sheet_a(df)

    section("STEP 5 — Effect of tone (Sheet B)")
    results["B"] = run_sheet_b(df)

    section("STEP 6 — AI perception regressions (Sheet C)")
    results["C"] = run_sheet_c(df)

    section("STEP 7 — Predictors of feedback quality (Sheet D)")
    results["D"] = run_sheet_d(df)

    section("STEP 8 — Predictors of chatbot evaluation (Sheet E)")
    results["E"] = run_sheet_e(df)

    section("STEP 9 — Mediation analyses (Sheet F)")
    results["F"] = run_sheet_f(df)

    section("STEP 10 — GPT-4o scoring (Sheet G)")
    if skip_gpt:
        log.info("--skip-gpt flag set: skipping GPT scoring.")
        results["G"] = pd.DataFrame({"note": ["GPT scoring skipped (--skip-gpt flag)"]})
    else:
        results["G"] = run_sheet_g(df)

    section("STEP 11 — Word frequencies & TF-IDF (Sheet H)")
    results["H"] = run_sheet_h(df)

    section("STEP 12 — Demographics & robustness checks (Sheet I)")
    results["I"] = run_sheet_i(df)

    section("STEP 13 — Table of contents (Sheet J)")
    results["J"] = run_sheet_j(results)

    section("STEP 14 — Writing Excel output")
    ensure_output_dir(output_path)
    sheet_order = ["L", "K", "J", "A", "B", "C", "D", "E", "F", "G", "H", "I"]
    write_excel(results, sheet_order, output_path)
    log.info(f"Excel file written to: {output_path}")

    elapsed = time.time() - pipeline_start
    section(f"PIPELINE COMPLETE — {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    check_api_key()

    if args.gpt_only:
        log.info("--gpt-only flag: running GPT scoring step only.")
        df = load_and_clean(args.data)
        gpt_results = run_sheet_g(df)
        gpt_path = args.output.replace(".xlsx", "_gpt_only.xlsx")
        ensure_output_dir(gpt_path)
        write_excel({"G": gpt_resu
