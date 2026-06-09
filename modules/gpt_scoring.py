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
import os
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
            "quantity_quote", "quantity_explanation",
            "quality_quote",  "quality_explanation",
            "emotions_quote", "emotions_explanation",
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
        row["quantity_quote"]       = runs[0].get("quantity_quote", "")
        row["quantity_explanation"] = runs[0].get("quantity_explanation", "")
        row["quality_quote"]        = runs[0].get("quality_quote", "")
        row["quality_explanation"]  = runs[0].get("quality_explanation", "")
        row["emotions_quote"]       = runs[0].get("emotions_quote", "")
        row["emotions_explanation"] = runs[0].get("emotions_explanation", "")
    else:
        row["quantity_quote"]       = "All runs failed"
        row["quantity_explanation"] = "All runs failed"
        row["quality_quote"]        = "All runs failed"
        row["quality_explanation"]  = "All runs failed"
        row["emotions_quote"]       = "All runs failed"
        row["emotions_explanation"] = "All runs failed"

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
def run_gpt_scoring(
    df: pd.DataFrame,
    cache_path: str = None,
) -> pd.DataFrame:
    """
    Score all conversation transcripts using GPT-4o.
    
    Implements a per-participant cache saved to disk after each scoring.
    If interrupted, relaunching will skip already-scored participants
    and continue from where it stopped — no double billing.

    Args:
        df:         DataFrame with response_id, transcript, tone
        cache_path: Path to JSON cache file (on Google Drive ideally).
                    Defaults to config cache path if set, else local.

    Returns:
        DataFrame formatted for Sheet G.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Use --skip-gpt to bypass this step."
        )

    required = ["response_id", "transcript", "tone"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"GPT scoring: missing columns {missing}")

    # ------------------------------------------------------------------
    # Cache setup
    # ------------------------------------------------------------------
    import json

    if cache_path is None:
        cache_path = getattr(
            config, "GPT_CACHE_PATH",
            "outputs/gpt_cache.json"
        )

    # Load existing cache
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            log.info(
                f"GPT cache loaded: {len(cache)} participants "
                f"already scored."
            )
        except Exception as e:
            log.warning(f"Could not load cache: {e} — starting fresh.")
            cache = {}

    # ------------------------------------------------------------------
    # Load prompt
    # ------------------------------------------------------------------
    prompt = _load_prompt()
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # Filter valid transcripts
    df_valid = df[
        df["transcript"].str.strip().astype(bool)
    ].copy()
    n_total = len(df_valid)

    # Identify who still needs scoring
    already_done = set(cache.keys())
    df_todo      = df_valid[
        ~df_valid["response_id"].astype(str).isin(already_done)
    ]

    log.info(
        f"GPT scoring: {n_total} total, "
        f"{len(already_done)} already cached, "
        f"{len(df_todo)} to score."
    )

    if len(df_todo) == 0:
        log.info("All participants already scored — loading from cache.")
    
    # ------------------------------------------------------------------
    # Score remaining participants
    # ------------------------------------------------------------------
    for idx, (_, row) in enumerate(df_todo.iterrows(), start=1):
        response_id = str(row["response_id"])
        transcript  = str(row["transcript"])
        tone_label  = config.TONE_LABELS.get(int(row["tone"]), "Unknown")
        num_turns   = int(row["num_turns"]) \
                      if pd.notna(row.get("num_turns")) else None

        log.info(
            f"Scoring [{idx}/{len(df_todo)}] "
            f"{response_id} ({tone_label})"
        )

        scored = _score_conversation(
            client, prompt, transcript, response_id
        )
        scored["tone_label"] = tone_label
        scored["num_turns"]  = num_turns

        # Save to cache immediately after each participant
        cache[response_id] = scored
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(
                f"Could not save cache after {response_id}: {e}"
            )

    # ------------------------------------------------------------------
    # Build Sheet G from cache (all participants)
    # ------------------------------------------------------------------
    rows = []
    for response_id in df_valid["response_id"].astype(str):
        if response_id in cache:
            rows.append(cache[response_id])
        else:
            log.warning(
                f"Participant {response_id} not in cache "
                f"— skipped."
            )

    sheet_g = pd.DataFrame(rows)

    # Reorder columns
    col_order = [
        "response_id", "tone_label", "num_turns",
        "quantity_run1", "quantity_run2", "quantity_run3",
        "quantity_mean", "quantity_sd",
        "quality_run1",  "quality_run2",  "quality_run3",
        "quality_mean",  "quality_sd",
        "emotions_run1", "emotions_run2", "emotions_run3",
        "emotions_mean", "emotions_sd",
        "composite",
        "quantity_quote",
        "quantity_explanation",
        "quality_quote",
        "quality_explanation",
        "emotions_quote",
        "emotions_explanation",
        "key_verbatim",
    ]
    sheet_g = sheet_g[
        [c for c in col_order if c in sheet_g.columns]
    ]

    # Merge scores back into main df
    score_cols = [
        "response_id",
        "quantity_run1","quantity_run2","quantity_run3","quantity_mean",
        "quality_run1", "quality_run2", "quality_run3", "quality_mean",
        "emotions_run1","emotions_run2","emotions_run3","emotions_mean",
        "composite",
    ]
    merge_cols = [c for c in score_cols if c in sheet_g.columns]
    df_scores  = sheet_g[merge_cols]

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
        f"GPT scoring complete: {len(sheet_g)} participants scored. "
        f"Cache saved to: {cache_path}"
    )

    return sheet_g
