"""
main.py
-------
Orchestration script for the SoundFlow thesis analysis pipeline.

Sheet order in output Excel:
    Table of Contents → Variable Definitions → Cleaned Data →
    Correlations → Effect of Tone → AI Perception Regressions →
    Predictors of Feedback Quality → Predictors of Chatbot Evaluation →
    Mediation Analyses → GPT Scoring → Word Frequencies →
    Demographics & Robustness

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

import numpy as np
import pandas as pd

import config
from modules.cleaning import load_and_clean
from modules.metrics import compute_all_metrics
from modules.analyses import run_all_analyses
from modules.gpt_scoring import run_gpt_scoring
from modules.word_freq import run_word_freq
from modules.export import write_excel

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
    parser.add_argument(
        "--data", type=str, default=config.DATA_PATH,
        help="Path to raw Qualtrics Excel export"
    )
    parser.add_argument(
        "--output", type=str, default=config.OUTPUT_PATH,
        help="Path for final Excel output"
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
    if not config.OPENAI_API_KEY:
        log.warning(
            "OPENAI_API_KEY not set. "
            "Sheet G will fail. Use --skip-gpt to bypass."
        )

def ensure_output_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def section(title: str) -> None:
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Sheet L — Variable definitions
# ---------------------------------------------------------------------------
def _build_sheet_l() -> pd.DataFrame:
    """
    Variable definitions table.
    Anchors sourced from:
      - Qualtrics question labels (row 2 of raw export)
      - Code definitions for computed variables
    Composite scores note: "Computed only if Cronbach alpha >= 0.70.
    If alpha < 0.70, individual items are used in analyses instead."
    """
    COMPOSITE_NOTE = (
        "Computed only if Cronbach alpha >= 0.70. "
        "If alpha < 0.70, individual items are used in analyses instead."
    )

    rows = [
        # --- Experimental design ---
        (
            "tone", "Chatbot Tone",
            "0 = Professional (FL_22) / 1 = Friendly (FL_21)",
            "Decoded from Qualtrics block randomizer column FL_13_DO",
            "Binary categorical",
            "Between-subjects manipulation. Participants were randomly "
            "assigned to one of two chatbot versions differing only in "
            "communication style.",
        ),

        # --- Chatbot Evaluation ---
        (
            "E1", "Required effort",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: Answering this survey required "
            "a lot of effort.",
        ),
        (
            "E2", "Engagement felt",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: I felt engaged while answering "
            "this survey.",
        ),
        (
            "E3", "Chatbot appreciation",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: I enjoyed interacting with "
            "the chatbot.",
        ),
        (
            "E4", "Conversation utility",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: I found the conversation useful "
            "for providing my feedback.",
        ),
        (
            "E5", "Reuse intention",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: I would consider using this "
            "chatbot again.",
        ),
        (
            "E6", "Chatbot preference",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: I would prefer to provide "
            "feedback to a chatbot rather than a human in the future.",
        ),

        # --- Perceived Manipulation ---
        (
            "PM1", "Threat to freedom",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire — Cox et al. (2026)",
            "Likert 1-7",
            "To what extent do you agree: The chatbot threatened my "
            "freedom to choose.",
        ),
        (
            "PM2", "Decision override",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire — Cox et al. (2026)",
            "Likert 1-7",
            "To what extent do you agree: The chatbot tried to make a "
            "decision for me.",
        ),
        (
            "PM3", "Manipulation attempt",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire — Cox et al. (2026)",
            "Likert 1-7",
            "To what extent do you agree: The chatbot tried to "
            "manipulate me.",
        ),
        (
            "PM4", "Pressure felt",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire — Cox et al. (2026)",
            "Likert 1-7",
            "To what extent do you agree: The chatbot tried to pressure me.",
        ),
        (
            "PM_score", "Perceived Manipulation composite",
            "Mean of PM1, PM2, PM3, PM4 — range 1-7",
            "Computed from questionnaire items",
            "Continuous",
            COMPOSITE_NOTE,
        ),

        # --- Competence to Judge ---
        (
            "Comp1", "Competence — skills",
            "1 = Not at all capable / 4 = Neither / 7 = Very capable",
            "Questionnaire — Oliveira et al. (2024)",
            "Likert 1-7",
            "How much do you believe that an AI system, in the present "
            "day, is able to form a judgment about: The abilities of "
            "someone?",
        ),
        (
            "Comp2", "Competence — morality",
            "1 = Not at all capable / 4 = Neither / 7 = Very capable",
            "Questionnaire — Oliveira et al. (2024)",
            "Likert 1-7",
            "How much do you believe that an AI system, in the present "
            "day, is able to form a judgment about: How moral someone's "
            "behavior is?",
        ),
        (
            "Comp_score", "Competence composite",
            "Mean of Comp1, Comp2 — range 1-7",
            "Computed from questionnaire items",
            "Continuous",
            COMPOSITE_NOTE,
        ),

        # --- Moral Agency ---
        (
            "MA1", "Moral gravity AI→human",
            "1 = Not at all / 4 = Moderate amount / 7 = A great deal",
            "Questionnaire — Ladak et al. (2025)",
            "Likert 1-7",
            "To what extent do you agree: How morally wrong would it be "
            "for AI to harm a person?",
        ),
        (
            "MA2", "AI moral responsibility",
            "1 = Not at all / 4 = Moderate amount / 7 = A great deal",
            "Questionnaire — Ladak et al. (2025)",
            "Likert 1-7",
            "To what extent do you agree: To what extent would AI deserve "
            "to be held morally responsible for causing a negative outcome?",
        ),
        (
            "MA_score", "Moral Agency composite",
            "Mean of MA1, MA2 — range 1-7",
            "Computed from questionnaire items",
            "Continuous",
            COMPOSITE_NOTE,
        ),

        # --- Moral Patiency ---
        (
            "MP1", "Moral gravity human→AI",
            "1 = Not at all / 4 = Moderate amount / 7 = A great deal",
            "Questionnaire — Ladak et al. (2025)",
            "Likert 1-7",
            "To what extent do you agree: How morally wrong would it be "
            "for a person to harm AI?",
        ),
        (
            "MP2", "AI right to moral consideration",
            "1 = Not at all / 4 = Moderate amount / 7 = A great deal",
            "Questionnaire — Ladak et al. (2025)",
            "Likert 1-7",
            "To what extent do you agree: To what extent does AI deserve "
            "to be treated with moral concern?",
        ),
        (
            "MP_score", "Moral Patiency composite",
            "Mean of MP1, MP2 — range 1-7",
            "Computed from questionnaire items",
            "Continuous",
            COMPOSITE_NOTE,
        ),

        # --- Perceived Autonomy ---
        (
            "Ind1", "AI plans & goals",
            "1 = Not at all / 4 = Moderate amount / 7 = A great deal",
            "Questionnaire — Ladak et al. (2025)",
            "Likert 1-7",
            "To what extent do you agree: To what extent can AI make "
            "plans and work towards goals?",
        ),
        (
            "Ind2", "AI self-control",
            "1 = Not at all / 4 = Moderate amount / 7 = A great deal",
            "Questionnaire — Ladak et al. (2025)",
            "Likert 1-7",
            "To what extent do you agree: To what extent can AI exercise "
            "self-control?",
        ),
        (
            "Ind_score", "Autonomy composite",
            "Mean of Ind1, Ind2 — range 1-7",
            "Computed from questionnaire items",
            "Continuous",
            COMPOSITE_NOTE,
        ),

        # --- Perceived Personality ---
        (
            "PP1", "Friendly",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: The chatbot seemed friendly.",
        ),
        (
            "PP2", "Professional",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: The chatbot seemed professional.",
        ),
        (
            "PP3", "Approachable",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: The chatbot seemed approachable.",
        ),
        (
            "PP4", "Warm",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: The chatbot seemed warm.",
        ),
        (
            "PP5", "Formal",
            "1 = Strongly disagree / 4 = Neither / 7 = Strongly agree",
            "Questionnaire",
            "Likert 1-7",
            "To what extent do you agree: The chatbot seemed formal.",
        ),

        # --- Feedback Quality (GPT-4o scored) ---
        (
            "quantity_run1/2/3", "Feedback quantity — per run",
            "1 = No axis covered / 2 = One axis skimmed / "
            "3 = One axis covered or two skimmed / "
            "4 = Two or three axes covered / 5 = All four axes covered",
            "GPT-4o scoring — 3 independent runs at temperature=0",
            "Ordinal 1-5",
            "How many of the four feedback axes (usage patterns, positive "
            "experiences, frustrations, desired improvements) are covered "
            "with at least one grounding element across the full "
            "conversation transcript.",
        ),
        (
            "quantity_mean", "Feedback quantity — mean",
            "Mean of 3 runs — range 1-5",
            "Computed from GPT-4o scoring runs",
            "Continuous",
            "Mean of quantity_run1, quantity_run2, quantity_run3. "
            "SD reported separately as quantity_sd.",
        ),
        (
            "quality_run1/2/3", "Feedback quality — per run",
            "1 = No depth / 2 = Very superficial / "
            "3 = At least one concrete element / "
            "4 = Several axes with context and examples / "
            "5 = Rich, precise, multi-dimensional",
            "GPT-4o scoring — 3 independent runs at temperature=0",
            "Ordinal 1-5",
            "Average depth across covered axes. Evaluates richness, "
            "specificity, and nuance of the information provided.",
        ),
        (
            "quality_mean", "Feedback quality — mean",
            "Mean of 3 runs — range 1-5",
            "Computed from GPT-4o scoring runs",
            "Continuous",
            "Mean of quality_run1, quality_run2, quality_run3. "
            "SD reported separately as quality_sd.",
        ),
        (
            "emotions_run1/2/3", "Emotional expression — per run",
            "1 = No emotion / 2 = Weak or implicit / "
            "3 = Explicit but isolated / "
            "4 = Explicit emotion with context / "
            "5 = Rich and developed emotional expression",
            "GPT-4o scoring — 3 independent runs at temperature=0",
            "Ordinal 1-5",
            "Presence and intensity of emotional expression across "
            "participant responses. Scored on richness and explicitness "
            "of emotional language.",
        ),
        (
            "emotions_mean", "Emotional expression — mean",
            "Mean of 3 runs — range 1-5",
            "Computed from GPT-4o scoring runs",
            "Continuous",
            "Mean of emotions_run1, emotions_run2, emotions_run3. "
            "SD reported separately as emotions_sd.",
        ),
        (
            "composite", "Feedback composite score",
            "Range 1-5 — formula: (quantity_mean × quality_mean) / 5",
            "Computed from GPT-4o scoring means",
            "Continuous",
            "Overall feedback quality score combining breadth (quantity) "
            "and depth (quality). Higher scores indicate broader coverage "
            "and greater depth.",
        ),

        # --- Engagement ---
        (
            "avg_words_per_turn", "Average words per turn",
            "Continuous — words per message",
            "JavaScript tracking (computed as total_user_words / num_turns)",
            "Continuous",
            "Average number of words written by the participant per "
            "chatbot turn. Proxy for effort invested per response.",
        ),
        (
            "chat_duration_sec", "Chat duration (seconds)",
            "Continuous — seconds",
            "JavaScript tracking (time from page load to last message)",
            "Continuous",
            "Total duration of the conversation in seconds. "
            "Limitation: may be inflated by inactivity periods.",
        ),
        (
            "z_avg_words", "Z-score: avg words per turn",
            "Continuous — z-score (M=0, SD~1)",
            "Computed from avg_words_per_turn",
            "Continuous",
            "Z-standardised avg_words_per_turn across all participants. "
            "Component of engagement_score composite.",
        ),
        (
            "z_duration", "Z-score: chat duration",
            "Continuous — z-score (M=0, SD~1)",
            "Computed from chat_duration_sec",
            "Continuous",
            "Z-standardised chat_duration_sec across all participants. "
            "Component of engagement_score composite.",
        ),
        (
            "engagement_score", "Engagement score (composite)",
            "Continuous — z-score composite (M=0, SD~1). "
            "Formula: (z_avg_words + z_duration) / 2",
            "Computed from JavaScript tracking variables",
            "Continuous",
            "Behavioural engagement composite. Equal-weighted mean of "
            "z_avg_words and z_duration. conversation_completed is "
            "excluded from the composite (control variable only) for "
            "theoretical and technical reasons: engagement can be high "
            "without completion, and the completion flag depends on "
            "correct token generation by the chatbot.",
        ),
        (
            "conversation_completed", "Conversation completed flag",
            "True / False",
            "JavaScript tracking",
            "Binary",
            "True if the chatbot generated the END_OF_INTERVIEW closing "
            "token. Used as control variable only — not included in "
            "engagement_score composite.",
        ),
        (
            "total_user_words", "Total user words",
            "Continuous — word count",
            "JavaScript tracking",
            "Continuous",
            "Total number of words written by the participant across "
            "all turns. Descriptive variable only.",
        ),
        (
            "num_turns", "Number of turns",
            "Continuous — integer",
            "JavaScript tracking",
            "Continuous",
            "Number of messages sent by the participant. "
            "Descriptive variable only.",
        ),

        # --- Demographics ---
        (
            "age", "Age",
            "Continuous — years",
            "Questionnaire: What is your age?",
            "Continuous",
            "Participant age in years.",
        ),
        (
            "gender", "Gender",
            "Male / Female / Non-binary / Prefer not to say",
            "Questionnaire: What is your gender? "
            "(1=Male, 2=Female, 3=Non-binary, 4=Prefer not to say)",
            "Categorical",
            "Participant gender. Recoded from numeric Qualtrics codes "
            "to text labels in cleaning.py.",
        ),
        (
            "language", "Response language",
            "FR / EN",
            "Qualtrics UserLanguage column",
            "Binary categorical",
            "Language detected by Qualtrics based on browser/survey "
            "settings. Used as covariate in robustness checks.",
        ),
    ]

    return pd.DataFrame(rows, columns=[
        "Variable", "Label", "Scale / Anchors",
        "Source", "Type", "Definition",
    ])


# ---------------------------------------------------------------------------
# Sheet J — Table of contents
# ---------------------------------------------------------------------------
def _build_sheet_j() -> pd.DataFrame:
    rows = [
        (
            "Table of Contents",
            "This sheet — index of all sheets with description of content.",
        ),
        (
            "Variable Definitions",
            "All variables with labels, scale anchors, source, type, "
            "and full definitions. Composite scores noted with Cronbach "
            "alpha condition.",
        ),
        (
            "Cleaned Data",
            "One row per valid participant (n=129 at time of writing). "
            "All variables after cleaning, exclusions, and recoding. "
            "Color-coded headers by variable block.",
        ),
        (
            "Correlations",
            "Pearson correlations between all continuous variables. "
            "FDR correction (Benjamini-Hochberg). Recap: significant "
            "only. Full table: all tested pairs.",
        ),
        (
            "Effect of Tone",
            "Independent samples t-tests (Welch) or Mann-Whitney U "
            "(if non-normal) comparing friendly vs professional condition "
            "on all variable blocks. H3: paired t-test Comp1 vs Comp2. "
            "Columns: Variable, Label, N/Mean/SD per condition, Delta, "
            "t, p-value, Cohen's d, Effect size.",
        ),
        (
            "AI Perception Regressions",
            "Simple OLS regressions between AI perception variables, "
            "all defined a priori. Composite scores used where "
            "alpha >= 0.70, individual items otherwise. H4 flagged. "
            "Columns: IV, Scale, DV, β, SE, t, p, R², F, Sig.",
        ),
        (
            "Predictors of Feedback Quality",
            "Hierarchical OLS regressions for composite and "
            "engagement_score. 3 blocks: (1) tone, (2) tone + AI "
            "perceptions, (3) + E1 + E2. ΔR² and F-change reported. "
            "Columns include Scale for each predictor.",
        ),
        (
            "Predictors of Chatbot Evaluation",
            "Hierarchical OLS regressions for E3, E4, E5, E6. "
            "3 blocks: (1) tone, (2) tone + AI perceptions, "
            "(3) + composite + engagement_score. ΔR² and F-change reported.",
        ),
        (
            "Mediation Analyses",
            "OLS bootstrapped mediation (Preacher & Hayes, 2008). "
            "5000 iterations, bias-corrected 95% CI. FDR corrected. "
            "Series 1: tone as IV. Series 2: AI perceptions as IV. "
            "DVs: composite, engagement_score, emotions_mean, E1-E6. "
            "Multicollinearity warning if r_IV_M > 0.80.",
        ),
        (
            "GPT Scoring",
            "GPT-4o scoring of conversation transcripts (temperature=0, "
            "3 independent runs). Scores: quantity, quality, emotions "
            "(1-5 each). Composite = (quantity × quality) / 5. "
            "Quotes and explanations per dimension. Summary by condition.",
        ),
        (
            "Word Frequencies",
            "Top 50 word frequencies and top 30 TF-IDF terms. "
            "4 sections: participant messages (friendly / professional) "
            "and chatbot messages (friendly / professional). "
            "Combined FR+EN stopwords applied.",
        ),
        (
            "Demographics & Robustness",
            "Descriptive statistics (age, gender, language, tone balance). "
            "ANCOVA: tone effect on all main DVs controlling for age, "
            "gender, language. Interaction tests: tone × language, "
            "tone × gender.",
        ),
    ]

    return pd.DataFrame(rows, columns=["Sheet", "Content"])


# ---------------------------------------------------------------------------
# GPT scoring summary by condition
# ---------------------------------------------------------------------------
def _build_gpt_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary table of GPT scores by tone condition.
    Only computed if GPT scoring has been run (columns present).
    """
    needed = ["tone", "quantity_mean", "quality_mean",
              "emotions_mean", "composite"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    if df["composite"].isna().all():
        return pd.DataFrame()

    rows = []
    for tone_val, tone_label in config.TONE_LABELS.items():
        sub = df[df["tone"] == tone_val]
        rows.append({
            "Condition":       tone_label,
            "N":               len(sub),
            "Mean quantity":   round(sub["quantity_mean"].mean(), 3),
            "SD quantity":     round(sub["quantity_mean"].std(),  3),
            "Mean quality":    round(sub["quality_mean"].mean(),  3),
            "SD quality":      round(sub["quality_mean"].std(),   3),
            "Mean emotions":   round(sub["emotions_mean"].mean(), 3),
            "SD emotions":     round(sub["emotions_mean"].std(),  3),
            "Mean composite":  round(sub["composite"].mean(),     3),
            "SD composite":    round(sub["composite"].std(),      3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(data_path: str, output_path: str, skip_gpt: bool) -> None:

    start = time.time()

    # ------------------------------------------------------------------
    # Step 1 — Load and clean
    # ------------------------------------------------------------------
    section("STEP 1 — Loading and cleaning data")
    if not os.path.exists(data_path):
        log.error(f"Data file not found: {data_path}")
        sys.exit(1)
    df = load_and_clean(data_path)
    log.info(
        f"Clean dataset: {len(df)} participants, "
        f"{len(df.columns)} columns"
    )

    # ------------------------------------------------------------------
    # Step 2 — Compute metrics
    # ------------------------------------------------------------------
    section("STEP 2 — Computing metrics")
    df = compute_all_metrics(df)

    # ------------------------------------------------------------------
    # Step 3 — GPT scoring (optional)
    # ------------------------------------------------------------------
    section("STEP 3 — GPT-4o scoring")
    if skip_gpt:
        log.info("--skip-gpt: skipping GPT scoring.")
        sheet_g = {
            "scores":  pd.DataFrame({"note": ["GPT scoring skipped."]}),
            "recap":   pd.DataFrame(),
            "summary": pd.DataFrame(),
        }
    else:
        gpt_df  = run_gpt_scoring(df)
        # Merge GPT scores back into df for analyses
        gpt_cols = [
            "response_id",
            "quantity_run1","quantity_run2","quantity_run3","quantity_mean",
            "quality_run1", "quality_run2", "quality_run3", "quality_mean",
            "emotions_run1","emotions_run2","emotions_run3","emotions_mean",
            "composite",
        ]
        merge_cols = [c for c in gpt_cols if c in gpt_df.columns]
        for col in merge_cols:
            if col == "response_id":
                continue
            df[col] = df["response_id"].map(
                gpt_df.set_index("response_id")[col]
            )
        # Recompute composite via metrics
        df = compute_all_metrics(df)

        sheet_g = {
            "scores":  gpt_df,
            "recap":   pd.DataFrame(),
            "summary": _build_gpt_summary(df),
        }

    # ------------------------------------------------------------------
    # Step 4 — Statistical analyses
    # ------------------------------------------------------------------
    section("STEP 4 — Statistical analyses")
    results = run_all_analyses(df)

    # ------------------------------------------------------------------
    # Step 5 — Word frequencies
    # ------------------------------------------------------------------
    section("STEP 5 — Word frequencies & TF-IDF")
    results["H"] = run_word_freq(df)

    # ------------------------------------------------------------------
    # Step 6 — Assemble all sheets
    # ------------------------------------------------------------------
    section("STEP 6 — Assembling sheets")
    results["G"] = sheet_g
    results["K"] = df.copy()
    results["L"] = _build_sheet_l()
    results["J"] = _build_sheet_j()

    # ------------------------------------------------------------------
    # Step 7 — Write Excel
    # Sheet order: J → L → K → A → B → C → D → E → F → G → H → I
    # ------------------------------------------------------------------
    section("STEP 7 — Writing Excel output")
    ensure_output_dir(output_path)
    sheet_order = ["J","L","K","A","B","C","D","E","F","G","H","I"]
    write_excel(results, sheet_order, output_path)
    log.info(f"Output written to: {output_path}")

    section(f"PIPELINE COMPLETE — {time.time()-start:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    check_api_key()

    if args.gpt_only:
        log.info("--gpt-only: running GPT scoring only.")
        df = load_and_clean(args.data)
        df = compute_all_metrics(df)
        gpt_df  = run_gpt_scoring(df)
        gpt_path = args.output.replace(".xlsx", "_gpt_only.xlsx")
        ensure_output_dir(gpt_path)
        sheet_g = {
            "scores":  gpt_df,
            "recap":   pd.DataFrame(),
            "summary": _build_gpt_summary(df),
        }
        write_excel({"G": sheet_g}, ["G"], gpt_path)
        log.info(f"GPT-only output: {gpt_path}")
        sys.exit(0)

    run_pipeline(args.data, args.output, skip_gpt=args.skip_gpt)
