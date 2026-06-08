"""
cleaning.py
-----------
Step 1 of the pipeline: load the raw Qualtrics export, clean it,
and return an analysis-ready DataFrame.

Responsibilities:
    - Skip Qualtrics' second header row (long question labels)
    - Rename columns to clean variable names (see config.COLUMN_RENAME)
    - Decode tone condition from FL_13_DO (FL_21=1, FL_22=0)
    - Decode gender from numeric codes (1=Male, 2=Female, 3=Non-binary, 4=Prefer not to say)
    - Filter out ineligible and incomplete responses
    - Filter out participants with no chatbot messages
    - Reconstruct full conversation transcripts from msg_1...msg_20
      and response_1...response_20
    - Basic type casting and Likert range validation
"""

import logging
import pandas as pd

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gender decoding
# ---------------------------------------------------------------------------
GENDER_MAP = {
    "1": "Male",
    "2": "Female",
    "3": "Non-binary",
    "4": "Prefer not to say",
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def load_and_clean(data_path: str) -> pd.DataFrame:
    """
    Load and clean the raw Qualtrics export.

    Args:
        data_path: Path to the Qualtrics .xlsx export file.

    Returns:
        Cleaned DataFrame with one row per valid participant.
    """
    log.info(f"Loading data from: {data_path}")

    # ------------------------------------------------------------------
    # 1. Load raw file
    # Row 0 = short column names (used as headers)
    # Row 1 = long question labels shown to participants (skipped)
    # ------------------------------------------------------------------
    df_raw = pd.read_excel(
        data_path,
        header=config.QUALTRICS_HEADER_ROW,
        skiprows=config.QUALTRICS_SKIP_ROWS,
        dtype=str,
    )
    log.info(f"Raw data loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # ------------------------------------------------------------------
    # 2. Rename columns to clean variable names
    # ------------------------------------------------------------------
    df = df_raw.rename(columns=config.COLUMN_RENAME)

    renamed = set(config.COLUMN_RENAME.values())
    unknown = [
        c for c in df.columns
        if c not in renamed
        and c not in config.MSG_COLS
        and c not in config.RESPONSE_COLS
    ]
    if unknown:
        log.warning(f"Unknown columns kept as-is: {unknown}")

    # ------------------------------------------------------------------
  # ------------------------------------------------------------------
    # 3. Decode tone condition
    # FL_13_DO → tone (1=friendly, 0=professional)
    # Rows with unrecognised tone values (NaN) are dropped with a warning.
    # ------------------------------------------------------------------
    if "tone_raw" not in df.columns:
        log.error("Column 'tone_raw' (FL_13_DO) not found.")
        raise ValueError("Missing tone condition column FL_13_DO.")

    df["tone"] = df["tone_raw"].map(config.TONE_MAP)
    n_missing_tone = df["tone"].isna().sum()
    if n_missing_tone > 0:
        log.warning(
            f"{n_missing_tone} rows have unrecognised tone value "
            f"(not FL_21 or FL_22) — dropped."
        )
        df = df.dropna(subset=["tone"])

    df["tone"] = df["tone"].astype(int)
    log.info(
        f"Tone decoded — Friendly (1): {(df['tone']==1).sum()}, "
        f"Professional (0): {(df['tone']==0).sum()}"
    )

   
    # ------------------------------------------------------------------
    # 4. Filter ineligible participants
    # Keep only participants who confirmed using a streaming platform
    # (User ? == 1)
    # ------------------------------------------------------------------
    if "eligible" in df.columns:
        before = len(df)
        df     = df[df["eligible"].astype(str).str.strip() == "1"].copy()
        dropped = before - len(df)
        if dropped > 0:
            log.info(
                f"Dropped {dropped} ineligible participant(s) "
                f"(User ? != 1)"
            )

    # ------------------------------------------------------------------
    # 5. Filter incomplete responses and no chatbot interaction
    #
    # Criteria kept:
    #   a) Progress = 100 — participant completed the full survey
    #   b) At least 1 message sent to the chatbot
    #
    # Criteria removed:
    #   - Finished flag: may be recorded at the wrong moment in
    #     Qualtrics and is redundant with Progress = 100
    #   - Tone condition check: already handled in Step 3
    # ------------------------------------------------------------------

    # a) Progress = 100
    if "Progress" in df.columns:
        before  = len(df)
        df      = df[
            df["Progress"].astype(str).str.strip() == "100"
        ].copy()
        dropped = before - len(df)
        if dropped > 0:
            log.info(
                f"Dropped {dropped} participant(s) with Progress != 100"
            )

    # b) At least 1 message sent to the chatbot
    msg_cols_present = [c for c in config.MSG_COLS if c in df.columns]
    if msg_cols_present:
        before      = len(df)
        has_message = df[msg_cols_present].apply(
            lambda row: any(
                str(v).strip() not in ("", "nan", "None")
                for v in row
            ), axis=1
        )
        df      = df[has_message].copy()
        dropped = before - len(df)
        if dropped > 0:
            log.info(
                f"Dropped {dropped} participant(s) with no chatbot "
                f"message (no JavaScript tracking data)"
            )
    else:
        log.warning(
            "No msg columns found — cannot filter empty conversations"
        )

    # ------------------------------------------------------------------
    # 6. Cast Likert scale columns to numeric
    # Out-of-range values (outside 1-7) set to NaN
    # ------------------------------------------------------------------
    likert_cols = (
        [f"E{i}" for i in range(1, 7)] +
        [f"PM{i}" for i in range(1, 5)] +
        ["Comp1", "Comp2"] +
        ["MA1", "MA2", "MP1", "MP2"] +
        ["Ind1", "Ind2"] +
        ["PP1", "PP2", "PP3", "PP4", "PP5"]
    )
    for col in likert_cols:
        if col not in df.columns:
            log.warning(f"Expected Likert column not found: {col}")
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        out_of_range = df[col].notna() & (
            (df[col] < config.LIKERT_MIN) | (df[col] > config.LIKERT_MAX)
        )
        if out_of_range.sum() > 0:
            log.warning(
                f"Column {col}: {out_of_range.sum()} value(s) outside "
                f"{config.LIKERT_MIN}-{config.LIKERT_MAX} → set to NaN"
            )
            df.loc[out_of_range, col] = pd.NA

    # ------------------------------------------------------------------
    # 7. Cast numeric metadata columns
    # ------------------------------------------------------------------
    numeric_cols = [
        "age", "chat_duration_sec", "num_turns",
        "avg_words_per_turn", "total_user_words"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # conversation_completed → boolean
    if "conversation_completed" in df.columns:
        df["conversation_completed"] = (
            df["conversation_completed"]
            .astype(str).str.strip().str.lower()
            .map({"true": True, "1": True, "false": False, "0": False})
        )

    # ------------------------------------------------------------------
    # 8. Decode gender (1=Male, 2=Female, 3=Non-binary, 4=Prefer not to say)
    # ------------------------------------------------------------------
    if "gender" in df.columns:
        df["gender"] = (
            df["gender"]
            .astype(str).str.strip()
            .map(GENDER_MAP)
        )
        n_unknown_gender = df["gender"].isna().sum()
        if n_unknown_gender > 0:
            log.warning(f"Gender: {n_unknown_gender} unrecognised value(s) → NaN")

    # ------------------------------------------------------------------
    # 9. Reconstruct conversation transcripts
    # Format:
    #   [Participant] message text
    #   [Chatbot] response text
    # ------------------------------------------------------------------
    def build_transcript(row: pd.Series) -> str:
        turns = []
        for i in range(1, 21):
            msg  = str(row.get(f"msg_{i}",      "")).strip()
            resp = str(row.get(f"response_{i}", "")).strip()
            if msg  and msg  not in ("nan", "None", ""):
                turns.append(f"[Participant] {msg}")
            if resp and resp not in ("nan", "None", ""):
                turns.append(f"[Chatbot] {resp}")
        return "\n".join(turns)

    df["transcript"] = df.apply(build_transcript, axis=1)
    log.info("Conversation transcripts reconstructed.")

    # ------------------------------------------------------------------
    # 10. Normalise language column
    # ------------------------------------------------------------------
    if "language" in df.columns:
        df["language"] = (
            df["language"]
            .astype(str).str.strip().str.upper()
            .replace({"NAN": pd.NA, "NONE": pd.NA})
        )

    # ------------------------------------------------------------------
    # 11. Move tone_raw to last column for readability
    # ------------------------------------------------------------------
    if "tone_raw" in df.columns:
        cols = [c for c in df.columns if c != "tone_raw"] + ["tone_raw"]
        df = df[cols]

    # ------------------------------------------------------------------
    # 12. Reset index and log final state
    # ------------------------------------------------------------------
    df = df.reset_index(drop=True)
    log.info(
        f"Cleaning complete — {len(df)} valid participants, "
        f"{len(df.columns)} columns."
    )

    missing_summary = {
        col: int(df[col].isna().sum())
        for col in likert_cols + ["tone", "age", "gender", "language"]
        if col in df.columns and df[col].isna().sum() > 0
    }
    if missing_summary:
        log.info(f"Missing values in key columns: {missing_summary}")

    return df
