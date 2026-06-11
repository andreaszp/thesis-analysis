"""
config.py
---------
Central configuration for the SoundFlow thesis analysis pipeline.
All paths, constants, and column mappings are defined here.
This is the only file you need to edit if the data structure changes.
"""

import os

# ---------------------------------------------------------------------------
# API KEY
# ---------------------------------------------------------------------------
# In Colab: set via google.colab.userdata (see run_on_colab.ipynb)
# Locally:  export OPENAI_API_KEY="sk-..." in your terminal
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATA_PATH: str        = "data/raw/qualtrics_export.xlsx"
OUTPUT_PATH: str      = "outputs/results.xlsx"
GPT_PROMPT_PATH: str  = "prompts/gpt_scoring_prompt.txt"

# ---------------------------------------------------------------------------
# QUALTRICS IMPORT
# ---------------------------------------------------------------------------
# Row 0 = short column names (used as headers)
# Row 1 = long question labels shown to participants (skipped on import)
QUALTRICS_HEADER_ROW: int  = 0
QUALTRICS_SKIP_ROWS: list  = [1]

# ---------------------------------------------------------------------------
# EXPERIMENTAL DESIGN
# ---------------------------------------------------------------------------
# Last column in Qualtrics export encodes the tone condition
# FL_21 = friendly, FL_22 = professional
TONE_COL_RAW: str   = "FL_13_DO"
TONE_MAP: dict      = {"FL_21": 1, "FL_22": 0}
TONE_LABELS: dict   = {1: "Casual", 0: "Formal"}

# ---------------------------------------------------------------------------
# COLUMN RENAMING
# Raw Qualtrics name → clean variable name used throughout the pipeline
# ---------------------------------------------------------------------------
COLUMN_RENAME: dict = {

    # --- metadata ---
    "ResponseId":             "response_id",
    "StartDate":              "start_date",
    "EndDate":                "end_date",
    "Duration (in seconds)":  "duration_sec",
    "Finished":               "finished",
    "Progress": "progress",
    "UserLanguage":           "language",
    "FL_13_DO":               "tone_raw",

    # --- screener ---
    "User ?":                 "eligible",

    # --- chatbot evaluation (E1–E6) ---
    "evaluation_1":           "E1",   # effort required
    "evaluation_2":           "E2",   # engagement felt
    "evaluation_3":           "E3",   # chatbot appreciation
    "evaluation_4":           "E4",   # conversation utility
    "evaluation_5":           "E5",   # reuse intention
    "evaluation_6":           "E6",   # preference over human

    # --- perceived manipulation (PM1–PM4) ---
    "Perceived Manipulati_1": "PM1",  # threat to freedom
    "Perceived Manipulati_2": "PM2",  # decision override
    "Perceived Manipulati_3": "PM3",  # manipulation attempt
    "Perceived Manipulati_4": "PM4",  # pressure felt

    # --- competence to judge (Comp1–Comp2) ---
    "Competence_1":           "Comp1", # competence — skills
    "Competence_2":           "Comp2", # competence — morality

    # --- moral agency (MA1–MA2) ---
    "Moral Responsibility_1": "MA1",  # morally wrong for AI to harm human
    "Moral Responsibility_2": "MA2",  # AI deserves moral responsibility

    # --- moral patiency (MP1–MP2) ---
    "Moral Responsibility_3": "MP1",  # morally wrong for human to harm AI
    "Moral Responsibility_4": "MP2",  # AI deserves moral concern

    # --- perceived autonomy (Ind1–Ind2) ---
    "Sense of Independenc_1": "Ind1", # AI makes plans and goals
    "Sense of Independenc_2": "Ind2", # AI exercises self-control

    # --- perceived personality (PP1–PP5) ---
    "personnality - Manip_1": "PP1",  # friendly
    "personnality - Manip_2": "PP2",  # professional
    "personnality - Manip_3": "PP3",  # approachable
    "personnality - Manip_4": "PP4",  # warm
    "personnality - Manip_5": "PP5",  # formal

    # --- demographics ---
    "age":                    "age",
    "gender":                 "gender",

    # --- chatbot technical metadata ---
    "system":                 "system_prompt",
    "conversation_completed": "conversation_completed",
    "chat_duration_sec":      "chat_duration_sec",
    "num_turns":              "num_turns",
    "avg_words_per_turn":     "avg_words_per_turn",
    "total_user_words":       "total_user_words",
    "ai_model":               "ai_model",
}

# --- conversation message columns (kept as-is, not renamed) ---
MSG_COLS: list      = [f"msg_{i}" for i in range(1, 21)]
RESPONSE_COLS: list = [f"response_{i}" for i in range(1, 21)]

# ---------------------------------------------------------------------------
# SCALE ITEMS — used for Cronbach's alpha + composite score computation
# ---------------------------------------------------------------------------
SCALE_ITEMS: dict = {
    "PM":   ["PM1", "PM2", "PM3", "PM4"],
    "Comp": ["Comp1", "Comp2"],
    "Ind":  ["Ind1", "Ind2"],
    "MA":   ["MA1", "MA2"],
    "MP":   ["MP1", "MP2"],
}

# Composite score variable names (mean of items, only if alpha >= 0.70)
COMPOSITE_SCORES: list = [
    "PM_score", "Comp_score", "Ind_score", "MA_score", "MP_score"
]

# Likert scale bounds (all questionnaire items use 1–7)
LIKERT_MIN: int = 1
LIKERT_MAX: int = 7

# ---------------------------------------------------------------------------
# GPT SCORING
# ---------------------------------------------------------------------------
GPT_MODEL: str        = "gpt-4o"
GPT_TEMPERATURE: float = 0.0
GPT_RUNS: int         = 3
GPT_MAX_TOKENS: int   = 500
# Cache — saved to Drive to survive session disconnections and credit issues
# If the scoring is interrupted, relaunching will skip already-scored
# participants and continue from where it stopped.
GPT_CACHE_PATH: str = "/content/drive/MyDrive/soundflow/gpt_cache.json"

# ---------------------------------------------------------------------------
# STATISTICAL THRESHOLDS
# ---------------------------------------------------------------------------
ALPHA: float              = 0.05
FDR_METHOD: str           = "fdr_bh"       # Benjamini-Hochberg
BOOTSTRAP_ITERATIONS: int = 5000
CRONBACH_MIN: float       = 0.70
NORMALITY_TEST: str       = "shapiro"

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# EXCEL FORMATTING
# ---------------------------------------------------------------------------

# Main color palette — teal / burnt orange / sage green / violet
COLORS: dict = {
    # Headers
    "header_bg":        "1B4F72",   # deep navy — main headers
    "header_font":      "FFFFFF",   # white

    # Block colors for Sheet K — vivid, readable
    "block_ai":         "1A5276",   # deep teal — AI perceptions
    "block_eval":       "784212",   # burnt orange — chatbot evaluation
    "block_quality":    "1D6A39",   # sage green — feedback quality
    "block_engage":     "4A235A",   # deep violet — engagement
    "block_demo":       "424949",   # dark grey — demographics
    "block_personality":"7B241C",   # deep red — perceived personality
    "block_tone":       "1F618D",   # steel blue — tone/condition

    # Section title colors
    "section_primary":  "1B4F72",   # deep navy
    "section_secondary":"2E86C1",   # medium blue

    # Row alternation
    "alt_row":          "EBF5FB",   # very light blue
    "row_white":        "FFFFFF",   # white

    # p-value colors (applied to p-value cells)
    "p_ns":             "E8E8E8",   # light grey — not significant (p > 0.05)
    "p_05":             "F39C12",   # orange — p < 0.05
    "p_01":             "27AE60",   # green — p < 0.01
    "p_001":            "1E8449",   # dark green — p < 0.001

    # Legacy (kept for compatibility)
    "sig_highlight":    "F39C12",
}

# p-value color thresholds
P_VALUE_COLORS: dict = {
    0.05:  "F39C12",   # orange
    0.01:  "27AE60",   # green
    0.001: "1E8449",   # dark green
    999:   "E8E8E8",   # grey (ns)
}
# ---------------------------------------------------------------------------
# SCALE STATUS — populated at runtime by metrics.py
# 'composite' = alpha >= 0.70, use scale_score
# 'separate'  = alpha < 0.70, use individual items
# ---------------------------------------------------------------------------
SCALE_STATUS: dict = {}
