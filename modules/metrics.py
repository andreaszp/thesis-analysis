"""
metrics.py
----------
Step 2 of the pipeline: compute all derived variables from the
cleaned DataFrame produced by cleaning.py.

Responsibilities:
    - Verify Cronbach's alpha for each scale (min threshold: 0.70)
    - Compute composite scores (PM_score, Comp_score, etc.)
    - Compute engagement score (z-score composite)
    - Compute GPT feedback composite (quantity x quality / 5)
    - Log all reliability results clearly
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived variables and add them to the DataFrame.

    Args:
        df: Cleaned DataFrame from cleaning.py

    Returns:
        DataFrame with added composite score and engagement columns.
    """
    log.info("Computing derived variables...")

    df = _compute_composite_scores(df)
    df = _compute_engagement_score(df)
    df = _compute_gpt_composite(df)

    log.info("All derived variables computed.")
    return df


# ---------------------------------------------------------------------------
# 1. Composite scores (after Cronbach's alpha check)
# ---------------------------------------------------------------------------
def _compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each scale defined in config.SCALE_ITEMS:
        1. Compute Cronbach's alpha
        2. If alpha >= config.CRONBACH_MIN (0.70):
               → compute mean composite score
               → mark scale as 'composite' in SCALE_STATUS
        3. If alpha < threshold:
               → do NOT compute composite
               → keep items separate
               → mark scale as 'separate' in SCALE_STATUS

    SCALE_STATUS is stored in config at runtime so analyses.py
    knows automatically which variables to use for each scale.
    """
    log.info("--- Cronbach's alpha verification ---")

    # Runtime status dict — read by analyses.py
    # { scale_name: 'composite' or 'separate' }
    scale_status = {}

    for scale_name, items in config.SCALE_ITEMS.items():
        score_col = f"{scale_name}_score"

        # Check all items are present
        missing_items = [it for it in items if it not in df.columns]
        if missing_items:
            log.warning(
                f"Scale {scale_name}: missing items {missing_items} "
                f"— kept separate."
            )
            scale_status[scale_name] = "separate"
            continue

        # Extract item matrix
        item_matrix = df[items].apply(pd.to_numeric, errors="coerce")
        complete_matrix = item_matrix.dropna()

        if len(complete_matrix) < 10:
            log.warning(
                f"Scale {scale_name}: only {len(complete_matrix)} complete "
                f"rows — kept separate."
            )
            scale_status[scale_name] = "separate"
            continue

        # Compute Cronbach's alpha
        alpha = _cronbach_alpha(complete_matrix)
        n_complete = len(complete_matrix)

        if alpha is None:
            log.warning(f"Scale {scale_name}: could not compute alpha.")
            scale_status[scale_name] = "separate"
            continue

        if alpha >= config.CRONBACH_MIN:
            # ✅ Composite: compute mean score
            df[score_col] = item_matrix.mean(axis=1)
            scale_status[scale_name] = "composite"
            log.info(
                f"Scale {scale_name} ({len(items)} items, n={n_complete}): "
                f"alpha = {alpha:.3f} ✅ → composite {score_col} computed."
            )
        else:
            # ❌ Separate: keep items as-is, composite = NaN
            df[score_col] = np.nan
            scale_status[scale_name] = "separate"
            log.info(
                f"Scale {scale_name} ({len(items)} items, n={n_complete}): "
                f"alpha = {alpha:.3f} ❌ → items {items} kept separate."
            )

    # Store status in config for analyses.py to read
    config.SCALE_STATUS = scale_status

    # Log summary
    log.info("--- Scale status summary ---")
    for scale, status in scale_status.items():
        items = config.SCALE_ITEMS[scale]
        if status == "composite":
            log.info(f"  {scale}_score → COMPOSITE (used in analyses)")
        else:
            log.info(f"  {scale} → SEPARATE: {items} (used individually)")

    return df

def _cronbach_alpha(item_matrix: pd.DataFrame) -> float | None:
    """
    Compute Cronbach's alpha for a matrix of item scores.

    Formula:
        alpha = (k / (k-1)) * (1 - sum(item_variances) / total_variance)
    where k = number of items.

    Args:
        item_matrix: DataFrame with participants as rows, items as columns.
                     Must have no missing values.

    Returns:
        Cronbach's alpha as float, or None if computation is impossible.
    """
    k = item_matrix.shape[1]
    n = item_matrix.shape[0]

    if k < 2 or n < 2:
        return None

    item_variances = item_matrix.var(axis=0, ddof=1)
    total_variance = item_matrix.sum(axis=1).var(ddof=1)

    if total_variance == 0:
        return None

    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return round(float(alpha), 4)


# ---------------------------------------------------------------------------
# 2. Engagement score
# ---------------------------------------------------------------------------
def _compute_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the engagement composite score.

    Components:
        - avg_words_per_turn: effort invested per response
        - chat_duration_sec:  total time invested in the conversation

    Method:
        1. Z-standardise each component across all participants
        2. Engagement score = mean of both z-scores

    conversation_completed is kept as a separate control variable
    and is NOT included in the composite for two reasons:
        1. Theoretical: strong engagement is possible without completion
           (fatigue, interruption, external factors).
        2. Technical: depends on chatbot generating <END_OF_INTERVIEW>
           token — may undercount true completions due to token errors.
    """
    required = ["avg_words_per_turn", "chat_duration_sec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning(
            f"Engagement score: missing columns {missing} "
            f"— engagement_score will not be computed."
        )
        return df

    # Check for missing values
    for col in required:
        n_na = df[col].isna().sum()
        if n_na > 0:
            log.warning(
                f"Engagement: {n_na} missing value(s) in '{col}' "
                f"— those participants will have NaN engagement_score."
            )

    # Z-standardise
    df["z_avg_words"] = _zscore_series(df["avg_words_per_turn"])
    df["z_duration"]  = _zscore_series(df["chat_duration_sec"])

    # Composite = mean of both z-scores
    df["engagement_score"] = df[["z_avg_words", "z_duration"]].mean(axis=1)

    # Check internal consistency of the two components
    _check_engagement_consistency(df)

    log.info(
        f"Engagement score computed for {df['engagement_score'].notna().sum()} "
        f"participants (mean={df['engagement_score'].mean():.3f}, "
        f"SD={df['engagement_score'].std():.3f})."
    )
    return df


def _zscore_series(series: pd.Series) -> pd.Series:
    """Z-standardise a pandas Series, ignoring NaN values."""
    mean = series.mean()
    std  = series.std()
    if std == 0:
        log.warning(f"Z-score: zero variance in '{series.name}' — returning zeros.")
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def _check_engagement_consistency(df: pd.DataFrame) -> None:
    """
    Check Pearson correlation between the two engagement components.
    A positive and significant r supports treating them as a composite.
    Logs a warning if the correlation is weak or non-significant.
    """
    clean = df[["z_avg_words", "z_duration"]].dropna()
    if len(clean) < 3:
        log.warning("Engagement consistency check: not enough data.")
        return

    r, p = stats.pearsonr(clean["z_avg_words"], clean["z_duration"])
    log.info(
        f"Engagement component correlation: r = {r:.3f}, p = {p:.4f} "
        f"({'✓ supports composite' if r > 0 and p < config.ALPHA else '⚠ weak — consider separate analysis'})"
    )


# ---------------------------------------------------------------------------
# 3. GPT feedback composite score
# ---------------------------------------------------------------------------
def _compute_gpt_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the GPT feedback composite score.

    Formula: composite = (quantity_mean * quality_mean) / 5

    These columns are added by gpt_scoring.py after GPT-4o scoring.
    If they are not yet present (e.g. --skip-gpt was used), this step
    is skipped silently and composite is set to NaN.

    Also computes mean and SD across the 3 runs for each dimension
    if the individual run columns are present.
    """
    run_cols = {
        "quantity": ["quantity_run1", "quantity_run2", "quantity_run3"],
        "quality":  ["quality_run1",  "quality_run2",  "quality_run3"],
        "emotions": ["emotions_run1", "emotions_run2", "emotions_run3"],
    }

    for dim, cols in run_cols.items():
        present = [c for c in cols if c in df.columns]
        if len(present) == 3:
            df[f"{dim}_mean"] = df[cols].mean(axis=1).round(3)
            df[f"{dim}_sd"]   = df[cols].std(axis=1).round(3)
        elif len(present) > 0:
            log.warning(
                f"GPT scoring: only {len(present)}/3 run columns found "
                f"for '{dim}' — mean/SD not computed."
            )

    # Compute composite only if both quantity_mean and quality_mean exist
    if "quantity_mean" in df.columns and "quality_mean" in df.columns:
        df["composite"] = (
            (df["quantity_mean"] * df["quality_mean"]) / 5
        ).round(3)
        log.info(
            f"GPT composite computed "
            f"(mean={df['composite'].mean():.3f}, "
            f"SD={df['composite'].std():.3f})."
        )
    else:
        if "composite" not in df.columns:
            df["composite"] = np.nan
        log.info(
            "GPT composite: quantity_mean/quality_mean not yet available "
            "(run GPT scoring first). composite set to NaN."
        )

    return df
