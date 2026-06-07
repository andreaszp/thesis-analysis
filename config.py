"""
config.py
---------
Central configuration for the SoundFlow thesis analysis pipeline.
All paths, constants, and column mappings are defined here.
Edit this file to adapt the pipeline to any changes in the data structure.
"""

import os

# ---------------------------------------------------------------------------
# API KEY
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATA_PATH: str = "data/raw/qualtrics_export.xlsx"
OUTPUT_PATH: str = "outputs/results.xlsx"
GPT_PROMPT_PATH: str = "prompts/gpt_scoring_prompt.txt"

# ---------------------------------------------------------------------------
# EXPERIMENTAL DESIGN
# ---------------------------------------------------------------------------
TONE_MAP: dict = {
    "FL_21": 1,   # friendly
    "FL_22": 0,   # professional
}
TONE_LABELS: dict = {1: "Friendly", 0: "Professional"}

# ---------------------------------------------------------------------------
# QUALTRICS COLUMN MAPPINGS
# ---------------------------------------------------------------------------
COLUMN_RENAME: dict = {
    "ResponseId":               "response_id",
    "StartDate":                "start_date",
    "EndDate":                  "end_date",
    "Duration (in seconds)":    "duration_sec",
    "Finished":                 "finished",
    "UserLanguage":             "language",
    "FL_13_DO":                 "randomizer_raw",
    "User ?":                   "eligible",
    "evaluation_1":             "E1",
    "evaluation_2":             "E2",
    "evaluation_3":             "E3",
    "evaluation_4":             "E4",
    "evaluation_5":             "E5",
    "evaluation_6":             "E6",
    "Perceived Manipulati_1":   "PM1",
    "Perceived Manipulati_2":   "PM2",
    "Perceived Manipulati_3":   "PM3",
    "Perceived Manipulati_4":   "PM4",
    "Competence_1":             "Comp1",
    "Competence_2":             "Comp2",
    "Moral Responsibility_1":   "MA1",
    "Moral Responsibility_2":   "MA2",
    "Moral Responsibility_3":   "MP1",
    "Moral Responsibility_4":   "MP2",
    "Sense of Independenc_1":   "Ind1",
    "Sense of Independenc_2":   "Ind2",
    "personnality - Manip_1":   "PP1",
    "personnality - Manip_2":   "PP2",
    "personnality - Manip_3":   "PP3",
    "personnality - Manip_4":   "PP4",
    "personnality - Manip_5":   "PP5",
    "age":                      "age",
    "gender":                   "gender",
    "system":                   "system_prompt",
    "conversation_completed":   "conversation_completed",
    "chat_duration_sec":        "chat_duration_sec",
    "num_turns":                "num_turns",
    "avg_words_per_turn":       "avg_words_per_turn",
    "total_user_words":         "total_user
