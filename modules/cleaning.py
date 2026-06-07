"""
cleaning.py
-----------
Step 1 of the pipeline: load the raw Qualtrics export, clean it,
and return a analysis-ready DataFrame.

Responsibilities:
    - Skip Qualtrics' second header row (long question labels)
    - Rename columns to clean variable names (see config.COLUMN_RENAME)
    - Decode tone condition from FL_13_DO (FL_21=1, FL_22=0)
    - Filter out ineligible and incomplete responses
    - Reconstruct full conversation transcripts from msg_1...msg_20
      and response_1...response_20
    - Basic type casting and Likert range validation
"""

import logging
import pandas as pd

import config

log = logging.getLogger(__name__)


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
        dtype=str,   # load everything as string first — cast later
    )
    log.info(f"Raw data loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # ------------------------------------------------------------------
    # 2. Rename columns to clean variable names
    # Only columns listed in config.COLUMN_RENAME are renamed.
    # msg_1...msg_20 and response_1...response_20 are kept as-is.
    # Unknown columns are kept with a warning.
    # ------------------------------------------------------------------
    df = df_raw.rename(columns=config.COLUMN_RENAME)

    renamed = set(config.COLUMN_RENAME.values())
    unknown = [c for c in df.columns
               if c not in renamed
               and c not in config.MSG_COLS
               and c not in config.RESPONSE_COLS]
    if unknown:
        log.warning(f"Unknown columns kept as-is: {unknown}")

    # ------------------------------------------------------------------
    # 3. Decode tone condition
    # FL_13_DO contains "FL_21" (friendly) or "FL_22" (professional)
    # → creates a clean binary column: tone (1=friendly, 0=professional)
    # ------------------------------------------------------------------
    if "tone_raw" not in df.columns:
        log.error("Column 'tone_raw' (FL_13_DO) not found. Cannot decode tone.")
        raise ValueError("Missing tone condition column FL_13_DO.")

    df["tone"] = df["tone_raw"].map(config.TONE_MAP)
    n_missing_tone = df["tone"].isna().sum()
    if n_missing_tone > 0:
        log.warning(
            f"{n_missing_tone} rows have unrecognised tone value "
            f"(not FL_21 or FL_22) — they will be dropped."
        )
    df = df.dropna(subset=["tone"])
    df["tone"] = df["tone"].astype(int)
    log.info(
        f"Tone decoded — Friendly (1): {(df['tone']==1).sum()}, "
        f"Professional (0): {(df['tone']==0).sum()}"
    )

    # ------------------------------------------------------------------
    # 4. Filter ineligible participants
    # eligible == "1" means the participant confirmed they use
    # a music streaming platform.
    # ------------------------------------------------------------------
    if "eligible" in df.columns:
        before = len(df)
        df = df[df["eligible"].astype(str).str.strip() == "1"].copy()
        dropped = before - len(df)
        if dropped > 0:
            log.info(f"Dropped {dropped} ineligible participant(s) (eligible != 1)")

    # ------------------------------------------------------------------
    # 5. Filter incomplete responses
    # Qualtrics sets Finished = "1" for complete responses.
    # ------------------------------------------------------------------
    if "finished" in df.columns:
        before = len(df)
        df = df[df["finished"].astype(str).str.strip() == "1"].copy()
        dropped = before - len(df)
        if dropped > 0:
            log.info(f"Dropped {dropped} incomplete response(s) (Finished != 1)")

    # ------------------------------------------------------------------
    # 6. Cast Likert scale columns to numeric (integer)
    # Out-of-range values (outside 1–7) are set to NaN with a warning.
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
                f"{config.LIKERT_MIN}–{config.LIKERT_MAX} → set to NaN"
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

    # conversation_completed: cast to boolean
    if "conversation_completed" in df.columns:
        df["conversation_completed"] = (
            df["conversation_completed"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "1": True, "false": False, "0": False})
        )

    # ------------------------------------------------------------------
    # 8. Reconstruct conversation transcripts
    # Interleave msg_1/response_1 ... msg_20/response_20 into a single
    # readable string per participant, skipping empty turns.
    # Format:
    #   [Participant] message text
    #   [Chatbot] response text
    # ------------------------------------------------------------------
    def build_transcript(row: pd.Series) -> str:
        turns = []
        for i in range(1, 21):
            msg_col  = f"msg_{i}"
            resp_col = f"response_{i}"

            msg  = str(row.get(msg_col,  "")).strip()
            resp = str(row.get(resp_col, "")).strip()

            # skip empty turns (NaN or empty string)
            if msg  and msg  not in ("nan", "None", ""):
                turns.append(f"[Participant] {msg}")
            if resp and resp not in ("nan", "None", ""):
                turns.append(f"[Chatbot] {resp}")

        return "\n".join(turns)

    df["transcript"] = df.apply(build_transcript, axis=1)
    log.info("Conversation transcripts reconstructed.")

    # ------------------------------------------------------------------
    # 9. Detect response language per participant
    # Qualtrics UserLanguage column already contains FR or EN.
    # Normalise to uppercase two-letter code.
    # ------------------------------------------------------------------
    if "language" in df.columns:
        df["language"] = (
            df["language"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": pd.NA, "NONE": pd.NA})
        )

    # ------------------------------------------------------------------
    # 10. Reset index and log final state
    # ------------------------------------------------------------------
    df = df.reset_index(drop=True)
    log.info(
        f"Cleaning complete — {len(df)} valid participants, "
        f"{len(df.columns)} columns."
    )

    # Log missing values summary for key variables
    key_vars = likert_cols + ["tone", "age", "gender", "language",
                               "engagement_score" if "engagement_score" in df.columns
                               else "avg_words_per_turn"]
    missing_summary = {
        col: int(df[col].isna().sum())
        for col in key_vars
        if col in df.columns and df[col].isna().sum() > 0
    }
    if missing_summary:
        log.info(f"Missing values in key columns: {missing_summary}")

    return df
