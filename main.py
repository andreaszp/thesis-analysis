"""
main.py
-------
Orchestration script for the SoundFlow thesis analysis pipeline.

Usage:
    python main.py
    python main.py --skip-gpt
    python main.py --data path/to/export.xlsx
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
from modules.metrics import compute_all_metrics
from modules.analyses import run_all_analyses
from modules.gpt_scoring import run_gpt_scoring
from modules.word_freq import run_word_freq
from modules.export import write_excel

# ---------------------------------------------------------------------------
# Logging — writes to console and to pipeline.log
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
    parser.add_argument(
        "--data", type=str, default=config.DATA_PATH,
        help="Path to the raw Qualtrics Excel export"
    )
    parser.add_argument(
        "--output", type=str, default=config.OUTPUT_PATH,
        help="Path for the final Excel output"
    )
    parser.add_argument(
        "--skip-gpt", action="store_true",
        help="Skip GPT-4o scoring (Sheet G)"
    )
    parser.add_argument(
        "--gpt-only", action="store_true",
        help="Run GPT-4o scoring only and exit"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_api_key() -> None:
    """Warn early if OpenAI API key is missing."""
    if not config.OPENAI_API_KEY:
        log.warning(
            "OPENAI_API_KEY is not set. "
            "Sheet G (GPT-4o scoring) will fail. "
            "Use --skip-gpt to bypass, or set the key before running."
        )

def ensure_output_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def section(title: str) -> None:
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(data_path: str, output_path: str, skip_gpt: bool) -> None:

    start = time.time()

    # ------------------------------------------------------------------
    # Step 1 — Load and clean raw Qualtrics data
    # Handles: skip row 1 (long labels), rename columns, decode tone,
    # filter ineligible participants, reconstruct transcripts
    # ------------------------------------------------------------------
    section("STEP 1 — Loading and cleaning data")
    if not os.path.exists(data_path):
        log.error(f"Data file not found: {data_path}")
        sys.exit(1)
    df = load_and_clean(data_path)
    log.info(f"Clean dataset: {len(df)} participants, {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # Step 2 — Compute all derived variables
    # Handles: Cronbach's alpha, composite scores, engagement score
    # ------------------------------------------------------------------
    section("STEP 2 — Computing derived variables")
    df = compute_all_metrics(df)

    # ------------------------------------------------------------------
    # Step 3 — Run all statistical analyses
    # Returns dict: { sheet_name → DataFrame }
    # Sheets: A (correlations), B (tone effects), C (AI perceptions),
    #         D (feedback quality), E (chatbot eval), F (mediation),
    #         I (demographics)
    # ------------------------------------------------------------------
    section("STEP 3 — Running statistical analyses")
    results = run_all_analyses(df)

    # ------------------------------------------------------------------
    # Step 4 — GPT-4o scoring (Sheet G)
    # 3 independent runs per conversation, temperature=0
    # Skipped if --skip-gpt flag is set
    # ------------------------------------------------------------------
    section("STEP 4 — GPT-4o scoring (Sheet G)")
    if skip_gpt:
        log.info("--skip-gpt flag set: skipping GPT-4o scoring.")
        results["G"] = pd.DataFrame({
            "note": ["GPT scoring skipped (--skip-gpt flag)"]
        })
    else:
        results["G"] = run_gpt_scoring(df)

    # ------------------------------------------------------------------
    # Step 5 — Word frequencies & TF-IDF (Sheet H)
    # Bilingual FR/EN processing
    # ------------------------------------------------------------------
    section("STEP 5 — Word frequencies & TF-IDF (Sheet H)")
    results["H"] = run_word_freq(df)

    # ------------------------------------------------------------------
    # Step 6 — Add cleaned data and variable definitions
    # Sheet K = cleaned data (one row per participant)
    # Sheet L = variable definitions table
    # Sheet J = table of contents (built last)
    # ------------------------------------------------------------------
    section("STEP 6 — Preparing remaining sheets")
    results["K"] = df.copy()
    results["L"] = _build_variable_definitions()
    results["J"] = _build_toc(results)

    # ------------------------------------------------------------------
    # Step 7 — Write Excel output
    # Sheet order: L → K → J → A → B → C → D → E → F → G → H → I
    # ------------------------------------------------------------------
    section("STEP 7 — Writing Excel output")
    ensure_output_dir(output_path)
    sheet_order = ["L", "K", "J", "A", "B", "C", "D", "E", "F", "G", "H", "I"]
    write_excel(results, sheet_order, output_path)
    log.info(f"Output written to: {output_path}")

    section(f"PIPELINE COMPLETE — {time.time() - start:.1f}s")


# ---------------------------------------------------------------------------
# Sheet L — Variable definitions (built directly in main for simplicity)
# ---------------------------------------------------------------------------
def _build_variable_definitions() -> pd.DataFrame:
    """
    Returns the variable definitions table for Sheet L.
    Anchors reflect the three Likert scale types used in the questionnaire:
      - Agree scale (1=Strongly disagree … 7=Strongly agree): E1-E6, PM1-PM4
      - Capable scale (1=Not at all capable … 7=Very capable): Comp1-Comp2
      - Amount scale (1=Not at all … 7=A great deal): MA1-MA2, MP1-MP2, Ind1-Ind2
    """
    rows = [
        # --- Manipulated IV ---
        ("tone", "Chatbot Tone", "0=Professional / 1=Friendly", "Decoded from FL_13_DO", "Binary", "Between-subjects tone condition"),

        # --- Chatbot Evaluation ---
        ("E1", "Required effort",       "1=Strongly disagree / 4=Neither / 7=Strongly agree", "Questionnaire", "Likert 1-7", "How much effort did the conversation require?"),
        ("E2", "Engagement felt",        "1=Strongly disagree / 4=Neither / 7=Strongly agree", "Questionnaire", "Likert 1-7", "How engaged did you feel?"),
        ("E3", "Chatbot appreciation",   "1=Strongly disagree / 4=Neither / 7=Strongly agree", "Questionnaire", "Likert 1-7", "Overall appreciation of the chatbot"),
        ("E4", "Conversation utility",   "1=Strongly disagree / 4=Neither / 7=Strongly agree", "Questionnaire", "Likert 1-7", "Usefulness for expressing feedback"),
        ("E5", "Reuse intention",        "1=Strongly disagree / 4=Neither / 7=Strongly agree", "Questionnaire", "Likert 1-7", "Willingness to use again"),
        ("E6", "Chatbot preference",     "1=Strongly disagree / 4=Neither / 7=Strongly agree", "Questionnaire", "Likert 1-7", "Preference over human interviewer"),

        # --- Perceived Manipulation ---
        ("PM1", "Threat to freedom",     "1=Strongly disagree / 4=Neither / 7=Strongly agree", "Questionnaire", "Likert 1-7", "The chatbot threatened my freedom to choose"),
        ("PM2", "Decision override",     "1=Strongly disagree / 4=Neither / 7=Strongly agree", "Questionnaire", "Likert 1-7", "The chatbot tried to make decisions for me"),
        ("PM3", "Manipulation attempt",  "1=Strongly disagree / 4=Neither / 7=Strongly agree", "Questionnaire", "Likert 1-7", "The chatbot tried to manipulate me"),
        ("PM4", "Pressure felt",         "1=Strongly disagree / 4=Neither / 7=Strongly agree", "Questionnaire", "Likert 1-7", "The chatbot tried to put pressure on me"),
        ("PM_score", "Perceived Manipulation score", "Mean of PM1-PM4", "Computed", "Continuous", "Only computed if Cronbach alpha >= 0.70"),

        # --- Competence to Judge ---
        ("Comp1", "Competence — skills",   "1=Not at all capable / 7=Very capable", "Questionnaire", "Likert 1-7", "AI capable of judging skills/competence"),
        ("Comp2", "Competence — morality", "1=Not at all capable / 7=Very capable", "Questionnaire", "Likert 1-7", "AI capable of judging moral behaviour"),
        ("Comp_score", "Competence score", "Mean of Comp1-Comp2", "Computed", "Continuous", "Only computed if Cronbach alpha >= 0.70"),

        # --- Moral Agency ---
        ("MA1", "Moral gravity AI→human", "1=Not at all / 7=A great deal", "Questionnaire", "Likert 1-7", "How morally wrong for AI to harm a person"),
        ("MA2", "AI moral responsibility", "1=Not at all / 7=A great deal", "Questionnaire", "Likert 1-7", "AI deserves to be held morally responsible"),
        ("MA_score", "Moral Agency score", "Mean of MA1-MA2", "Computed", "Continuous", "Only computed if Cronbach alpha >= 0.70"),

        # --- Moral Patiency ---
        ("MP1", "Moral gravity human→AI", "1=Not at all / 7=A great deal", "Questionnaire", "Likert 1-7", "How morally wrong for a person to harm AI"),
        ("MP2", "AI right to moral consideration", "1=Not at all / 7=A great deal", "Questionnaire", "Likert 1-7", "AI deserves to be treated with moral concern"),
        ("MP_score", "Moral Patiency score", "Mean of MP1-MP2", "Computed", "Continuous", "Only computed if Cronbach alpha >= 0.70"),

        # --- Perceived Autonomy ---
        ("Ind1", "AI plans & goals",   "1=Not at all / 7=A great deal", "Questionnaire", "Likert 1-7", "AI capable of making plans and working toward goals"),
        ("Ind2", "AI self-control",    "1=Not at all / 7=A great deal", "Questionnaire", "Likert 1-7", "AI capable of exercising self-control"),
        ("Ind_score", "Autonomy score", "Mean of Ind1-Ind2", "Computed", "Continuous", "Only computed if Cronbach alpha >= 0.70"),

        # --- Perceived Personality ---
        ("PP1", "Friendly",      "1=Strongly disagree / 7=Strongly agree", "Questionnaire", "Likert 1-7", "The chatbot seemed friendly"),
        ("PP2", "Professional",  "1=Strongly disagree / 7=Strongly agree", "Questionnaire", "Likert 1-7", "The chatbot seemed professional"),
        ("PP3", "Approachable",  "1=Strongly disagree / 7=Strongly agree", "Questionnaire", "Likert 1-7", "The chatbot seemed approachable"),
        ("PP4", "Warm",          "1=Strongly disagree / 7=Strongly agree", "Questionnaire", "Likert 1-7", "The chatbot seemed warm"),
        ("PP5", "Formal",        "1=Strongly disagree / 7=Strongly agree", "Questionnaire", "Likert 1-7", "The chatbot seemed formal"),

        # --- Feedback Quality (GPT-4o scored) ---
        ("quantity",  "Feedback quantity",  "1=No axis covered / 5=All four axes covered", "GPT-4o (3 runs)", "Ordinal 1-5", "How many feedback axes are covered"),
        ("quality",   "Feedback quality",   "1=No depth / 5=Rich and precise",             "GPT-4o (3 runs)", "Ordinal 1-5", "Average depth across covered axes"),
        ("emotions",  "Emotional expression","1=No emotion / 5=Rich emotional expression",  "GPT-4o (3 runs)", "Ordinal 1-5", "Presence and intensity of emotional expression"),
        ("composite", "Feedback composite", "(Quantity x Quality) / 5",                    "Computed",        "Continuous",  "Overall feedback quality score"),

        # --- Engagement ---
        ("avg_words_per_turn",    "Avg words per turn",    "Continuous",  "JavaScript", "Continuous", "Total user words / number of turns"),
        ("chat_duration_sec",     "Chat duration (sec)",   "Continuous",  "JavaScript", "Continuous", "Time from page load to last message"),
        ("engagement_score",      "Engagement score",      "Z-score composite", "Computed", "Continuous", "Mean of z(avg_words_per_turn) and z(chat_duration_sec)"),
        ("conversation_completed","Conversation completed", "True/False",  "JavaScript", "Binary",     "Control variable only — not in engagement composite"),
        ("total_user_words",      "Total user words",      "Continuous",  "JavaScript", "Continuous", "Descriptive only"),
        ("num_turns",             "Number of turns",       "Continuous",  "JavaScript", "Continuous", "Descriptive only"),

        # --- Demographics ---
        ("age",      "Age",      "Continuous", "Questionnaire", "Continuous",  "Participant age"),
        ("gender",   "Gender",   "Categorical","Questionnaire", "Categorical", "Participant gender"),
        ("language", "Language", "FR/EN",      "Qualtrics",     "Binary",      "Response language detected by Qualtrics"),
    ]

    return pd.DataFrame(rows, columns=[
        "Variable", "Label", "Scale / Anchors", "Source", "Type", "Definition"
    ])


# ---------------------------------------------------------------------------
# Sheet J — Table of contents (built last, after all results are ready)
# ---------------------------------------------------------------------------
def _build_toc(results: dict) -> pd.DataFrame:
    descriptions = {
        "L": "Variable definitions — labels, anchors, sources, types",
        "K": "Cleaned data — one row per participant, all variables",
        "J": "Table of contents",
        "A": "Correlation matrix — Pearson r, FDR-corrected, significant only",
        "B": "Effect of tone — t-tests / Mann-Whitney by condition",
        "C": "AI perception regressions — simple OLS, defined a priori",
        "D": "Predictors of feedback quality — hierarchical regressions",
        "E": "Predictors of chatbot evaluation — hierarchical regressions",
        "F": "Mediation analyses — bootstrapped, 5000 iter., FDR-corrected",
        "G": "GPT-4o scoring — 3 runs per conversation, reliability stats",
        "H": "Word frequencies & TF-IDF — bilingual FR/EN",
        "I": "Demographics & robustness checks — ANCOVA, interactions",
    }
    rows = [
        (sheet, descriptions.get(sheet, ""), "Yes" if sheet in results else "No")
        for sheet in ["L", "K", "J", "A", "B", "C", "D", "E", "F", "G", "H", "I"]
    ]
    return pd.DataFrame(rows, columns=["Sheet", "Content", "Generated"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    check_api_key()

    if args.gpt_only:
        log.info("--gpt-only flag: running GPT-4o scoring step only.")
        df = load_and_clean(args.data)
        df = compute_all_metrics(df)
        gpt_results = run_gpt_scoring(df)
        gpt_path = args.output.replace(".xlsx", "_gpt_only.xlsx")
        ensure_output_dir(gpt_path)
        write_excel({"G": gpt_results}, ["G"], gpt_path)
        log.info(f"GPT-only output written to: {gpt_path}")
        sys.exit(0)

    run_pipeline(args.data, args.output, skip_gpt=args.skip_gpt)
