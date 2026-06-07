"""
gpt_scoring.py
--------------
Step 4 of the pipeline: score all conversation transcripts using GPT-4o.

Responsibilities:
    - Run GPT-4o scoring 3 independent times per conversation (temperature=0)
    - Parse JSON output for each run (quantity, quality, emotions + justifications)
    - Compute mean and SD across the 3 runs for reliability assessment
    - Return a DataFrame ready for Sheet G export
    - Add run columns to main df for metrics.py composite computation

Each conversation is scored on:
    - Quantity  (1-5): how many feedback axes are covered
    - Quality   (1-5): average depth across covered axes
    - Emotions  (1-5): presence and intensity of emotional expression
    - Composite       : (quantity x quality) / 5 — computed in metrics.py
"""

import json
import logging
import time
import numpy as np
import pandas as pd
from openai import OpenAI

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load BARS scoring prompt from file
# ---------------------------------------------------------------------------
def _load_prompt() -> str:
    """Load the GPT scoring prompt from prompts/gpt_scoring_prompt.txt."""
    try:
        with open(config.GPT_PROMPT_PATH, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        if not prompt:
            raise ValueError("GPT scoring prompt file is empty.")
        log.info(f"GPT scoring prompt loaded from {config.GPT_PROMPT_PATH}")
        return prompt
    except FileNotFoundError:
        log.error(
            f"GPT scoring prompt not found at {config.GPT_PROMPT_PATH}. "
            f"Please paste your BARS prompt into that file."
        )
        raise


# ---------------------------------------------------------------------------
# Score a single conversation — one run
# ---------------------------------------------------------------------------
def _score_once(
    client: OpenAI,
    prompt: str,
    transcript: str,
    response_id: str,
    run_id: int,
) -> dict | None:
    """
    Send one conversation transcript to GPT-4o and parse the JSON response.

    Args:
        client:      OpenAI client instance
        prompt:      BARS scoring system prompt
        transcript:  Full conversation transcript as a string
        response_id: Participant identifier (for logging)
        run_id:      Run number (1, 2, or 3)

    Returns:
        Dict with quantity, quality, emotions, justifications
        or None if the call fails.
    """
    try:
        response = client.chat.completions.create(
            model=config.GPT_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Here is the conversation to evaluate:\n\n{transcript}"
                    ),
                },
            ],
            temperature=config.GPT_TEMPERATURE,
            max_tokens=config.GPT_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        result = json.loads(raw)

        # Validate expected keys
        required_keys = [
            "quantity", "quality", "emotions",
            "quantity_justification",
            "quality_justification",
            "emotions_justification",
        ]
        
        for key in required_keys:
            if key not in result:
                log.warning(
                    f"[{response_id}] Run {run_id}: missing key '{key}' "
                    f"in GPT response — run marked as failed."
                )
                return None
        
        if "key_quotes" not in result:
            result["key_quotes"] = {
                "quantity": "",
                "quality": "",
                "emotions": ""
            }
        # Validate score ranges (1–5)
        for dim in ["quantity", "quality", "emotions"]:
            val = result[dim]
            if not isinstance(val, int) or val < 1 or val > 5:
                log.warning(
                    f"[{response_id}] Run {run_id}: '{dim}' = {val} "
                    f"is out of range (expected 1–5) — run marked as failed."
                )
                return None

        result["run_id"] = run_id
        return result

    except json.JSONDecodeError as e:
        log.error(f"[{response_id}] Run {run_id}: JSON parse error — {e}")
        return None
    except Exception as e:
        log.error(f"[{response_id}] Run {run_id}: API error — {e}")
        return None


# ---------------------------------------------------------------------------
# Score a single conversation — 3 runs
# ---------------------------------------------------------------------------
def _score_conversation(
    client: OpenAI,
    prompt: str,
    transcript: str,
    response_id: str,
) -> dict:
    """
    Run GPT-4o scoring 3 times on one conversation and aggregate results.

    Args:
        client:      OpenAI client instance
        prompt:      BARS scoring system prompt
        transcript:  Full conversation transcript
        response_id: Participant identifier

    Returns:
        Dict with per-run scores, means, SDs, and justifications.
        Failed runs are excluded from mean/SD computation.
        If all 3 runs fail, returns NaN for all scores.
    """
    runs = []
    for run_id in range(1, config.GPT_RUNS + 1):
        result = _score_once(client, prompt, transcript, response_id, run_id)
        if result is not None:
            runs.append(result)
        # Small delay between runs to avoid rate limit issues
        time.sleep(0.5)

    dims = ["quantity", "quality", "emotions"]

    # Base output row
    row = {"response_id": response_id}

    # Per-run scores
    for i, run in enumerate(runs, start=1):
        for dim in dims:
            row[f"{dim}_run{i}"] = run.get(dim, np.nan)

    # Fill missing runs with NaN
    for i in range(len(runs) + 1, config.GPT_RUNS + 1):
        for dim in dims:
            row[f"{dim}_run{i}"] = np.nan

    # Mean and SD across successful runs
    if runs:
        for dim in dims:
            values = [r[dim] for r in runs if dim in r]
            row[f"{dim}_mean"] = round(np.mean(values), 3)
            row[f"{dim}_sd"]   = round(np.std(values), 3)
    else:
        log.error(
            f"[{response_id}] All {config.GPT_RUNS} runs failed — "
            f"scores set to NaN."
        )
        for dim in dims:
            row[f"{dim}_mean"] = np.nan
            row[f"{dim}_sd"]   = np.nan

    # Composite score: (quantity_mean x quality_mean) / 5
    if not np.isnan(row.get("quantity_mean", np.nan)) and \
       not np.isnan(row.get("quality_mean", np.nan)):
        row["composite"] = round(
            (row["quantity_mean"] * row["quality_mean"]) / 5, 3
        )
    else:
        row["composite"] = np.nan

    # Justifications — take from run 1 (most stable at temperature=0)
    if runs:
        row["quantity_justification"] = runs[0].get("quantity_justification", "")
        row["quality_justification"]  = runs[0].get("quality_justification", "")
        row["emotions_justification"] = runs[0].get("emotions_justification", "")
        
        # Key quotes — from run 1
        quotes = runs[0].get("key_quotes", {})
        row["quote_quantity"] = quotes.get("quantity", "")
        row["quote_quality"]  = quotes.get("quality",  "")
        row["quote_emotions"] = quotes.get("emotions", "")
    else:
        row["quantity_justification"] = "All runs failed"
        row["quality_justification"]  = "All runs failed"
        row["emotions_justification"] = "All runs failed"
        row["quote_quantity"] = ""
        row["quote_quality"]  = ""
        row["quote_emotions"] = ""

    # Key verbatim: first non-empty participant message
    key_verbatim = ""
    for line in transcript.split("\n"):
        if line.startswith("[Participant]"):
            text = line.replace("[Participant]", "").strip()
            if text:
                key_verbatim = text[:300]  # cap at 300 chars
                break
    row["key_verbatim"] = key_verbatim

    return row


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_gpt_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all conversation transcripts in the DataFrame.

    Requires:
        - df must have columns: response_id, transcript, tone
        - OPENAI_API_KEY must be set in the environment

    Returns:
        DataFrame formatted for Sheet G with columns:
            response_id, tone_label, num_turns,
            quantity_run1/2/3, quantity_mean, quantity_sd,
            quality_run1/2/3, quality_mean, quality_sd,
            emotions_run1/2/3, emotions_mean, emotions_sd,
            composite,
            quantity_justification, quality_justification,
            emotions_justification, key_verbatim
    """
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Cannot run GPT-4o scoring. "
            "Use --skip-gpt to bypass this step."
        )

    # Validate required columns
    required = ["response_id", "transcript", "tone"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"GPT scoring: missing columns {missing}")

    # Load prompt
    prompt = _load_prompt()

    # Initialise OpenAI client
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # Filter out rows with empty transcripts
    df_valid = df[df["transcript"].str.strip().astype(bool)].copy()
    n_total = len(df_valid)
    log.info(
        f"GPT-4o scoring: {n_total} conversations to score "
        f"({config.GPT_RUNS} runs each, model={config.GPT_MODEL})."
    )

    # Score all conversations
    rows = []
    for idx, (_, row) in enumerate(df_valid.iterrows(), start=1):
        response_id = str(row["response_id"])
        transcript  = str(row["transcript"])
        tone_label  = config.TONE_LABELS.get(int(row["tone"]), "Unknown")
        num_turns   = int(row["num_turns"]) if pd.notna(row.get("num_turns")) else None

        log.info(f"Scoring [{idx}/{n_total}] {response_id} ({tone_label})")

        scored = _score_conversation(client, prompt, transcript, response_id)
        scored["tone_label"] = tone_label
        scored["num_turns"]  = num_turns
        rows.append(scored)

    # Build Sheet G DataFrame
    sheet_g = pd.DataFrame(rows)

    # Reorder columns to match Sheet G specification
    col_order = [
        "response_id", "tone_label", "num_turns",
        "quantity_run1", "quantity_run2", "quantity_run3",
        "quantity_mean", "quantity_sd",
        "quality_run1",  "quality_run2",  "quality_run3",
        "quality_mean",  "quality_sd",
        "emotions_run1", "emotions_run2", "emotions_run3",
        "emotions_mean", "emotions_sd",
        "composite",
        "quantity_justification",
        "quote_quantity",
        "quality_justification",
        "quote_quality",
        "emotions_justification",
        "quote_emotions",
        "key_verbatim",
    ]
    sheet_g = sheet_g[[c for c in col_order if c in sheet_g.columns]]

    # Merge scoring results back into main df
    # (adds quantity_run1/2/3, quality_run1/2/3, emotions_run1/2/3,
    #  quantity_mean, quality_mean, emotions_mean, composite)
    score_cols = [
        "response_id",
        "quantity_run1", "quantity_run2", "quantity_run3", "quantity_mean",
        "quality_run1",  "quality_run2",  "quality_run3",  "quality_mean",
        "emotions_run1", "emotions_run2", "emotions_run3", "emotions_mean",
        "composite",
    ]
    merge_cols = [c for c in score_cols if c in sheet_g.columns]
    df_scores  = sheet_g[merge_cols]

    # Update main df in-place via index merge
    for col in merge_cols:
        if col == "response_id":
            continue
        df.loc[
            df["response_id"].isin(df_scores["response_id"]),
            col
        ] = df["response_id"].map(
            df_scores.set_index("response_id")[col]
        )

    log.info(
        f"GPT scoring complete. "
        f"Mean composite = {sheet_g['composite'].mean():.3f}, "
        f"SD = {sheet_g['composite'].std():.3f}"
    )

    return sheet_g
