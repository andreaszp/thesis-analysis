"""
word_freq.py
------------
Step 5 of the pipeline: word frequencies and TF-IDF analysis.

Bilingual processing (FR/EN):
    - Detects language per participant response (from 'language' column)
    - Applies language-specific stopwords
    - Computes word frequencies and TF-IDF scores separately for:
        * Participant messages only (not chatbot responses)
        * By tone condition (friendly vs professional)
        * By language (FR vs EN)
    - Aggregates results into a single Sheet H DataFrame
"""

import logging
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

import config

log = logging.getLogger(__name__)

# Download stopwords if not already present
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    log.info("Downloading NLTK stopwords...")
    nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Stopwords — FR + EN combined with custom additions
# ---------------------------------------------------------------------------
FR_STOPWORDS = set(stopwords.words("french"))
EN_STOPWORDS = set(stopwords.words("english"))

# Custom additions specific to this study context
CUSTOM_STOPWORDS = {
    # French fillers
    "c'est", "ca", "ça", "je", "tu", "il", "elle", "on", "nous", "vous",
    "ils", "elles", "un", "une", "des", "le", "la", "les", "de", "du",
    "en", "et", "ou", "mais", "donc", "or", "ni", "car", "que", "qui",
    "quoi", "dont", "où", "pas", "plus", "très", "bien", "aussi", "tout",
    "comme", "si", "quand", "même", "encore", "toujours", "jamais",
    # English fillers
    "i", "its", "it", "like", "just", "really", "also", "much", "lot",
    "get", "use", "used", "using", "think", "know", "thing", "things",
    "yeah", "yes", "okay", "ok", "kind", "sort", "bit",
    # Platform-neutral terms (too generic to be informative)
    "soundflow", "platform", "app", "application", "music", "spotify",
    "apple", "youtube",
}

FR_STOPWORDS.update(CUSTOM_STOPWORDS)
EN_STOPWORDS.update(CUSTOM_STOPWORDS)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation, numbers, and extra whitespace.
    Preserves accented characters (important for French).
    """
    text = text.lower()
    text = re.sub(r"[^\w\sàâäéèêëîïôùûüçœæ]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_participant_text(transcript: str) -> str:
    """
    Extract only participant messages from a transcript.
    Ignores lines starting with [Chatbot].
    """
    lines = transcript.split("\n")
    participant_lines = [
        line.replace("[Participant]", "").strip()
        for line in lines
        if line.startswith("[Participant]")
    ]
    return " ".join(participant_lines)


def _tokenize(text: str, lang: str) -> list:
    """
    Tokenize cleaned text and remove stopwords for the given language.

    Args:
        text: Cleaned text string
        lang: "FR" or "EN"

    Returns:
        List of meaningful tokens
    """
    stopwords_set = FR_STOPWORDS if lang == "FR" else EN_STOPWORDS
    tokens = text.split()
    return [t for t in tokens if t not in stopwords_set and len(t) > 2]


# ---------------------------------------------------------------------------
# Word frequency analysis
# ---------------------------------------------------------------------------
def _word_frequencies(
    texts: list,
    lang: str,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Compute word frequencies for a list of texts.

    Args:
        texts: List of raw transcript strings
        lang:  Language code ("FR" or "EN")
        top_n: Number of top words to return

    Returns:
        DataFrame with columns: word, frequency, rank
    """
    all_tokens = []
    for text in texts:
        cleaned = _clean_text(_extract_participant_text(text))
        tokens  = _tokenize(cleaned, lang)
        all_tokens.extend(tokens)

    if not all_tokens:
        return pd.DataFrame(columns=["word", "frequency", "rank"])

    counts = Counter(all_tokens)
    top    = counts.most_common(top_n)

    return pd.DataFrame(
        [(word, freq, rank + 1) for rank, (word, freq) in enumerate(top)],
        columns=["word", "frequency", "rank"],
    )


# ---------------------------------------------------------------------------
# TF-IDF analysis
# ---------------------------------------------------------------------------
def _tfidf_analysis(
    texts: list,
    lang: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Compute TF-IDF scores for a corpus of texts.

    Args:
        texts: List of raw transcript strings
        lang:  Language code ("FR" or "EN")
        top_n: Number of top terms to return

    Returns:
        DataFrame with columns: term, tfidf_mean, tfidf_max, document_freq
    """
    stopwords_set = FR_STOPWORDS if lang == "FR" else EN_STOPWORDS

    # Extract and clean participant text only
    cleaned_texts = [
        _clean_text(_extract_participant_text(t))
        for t in texts
    ]

    # Filter empty texts
    cleaned_texts = [t for t in cleaned_texts if t.strip()]
    if len(cleaned_texts) < 2:
        log.warning(f"TF-IDF ({lang}): fewer than 2 documents — skipped.")
        return pd.DataFrame(
            columns=["term", "tfidf_mean", "tfidf_max", "document_freq"]
        )

    vectorizer = TfidfVectorizer(
        stop_words=list(stopwords_set),
        min_df=2,           # term must appear in at least 2 documents
        max_df=0.85,        # ignore terms in more than 85% of documents
        ngram_range=(1, 2), # unigrams and bigrams
        token_pattern=r"(?u)\b[a-zA-ZàâäéèêëîïôùûüçœæÀÂÄÉÈÊËÎÏÔÙÛÜÇŒÆ]{3,}\b",
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    except ValueError as e:
        log.warning(f"TF-IDF ({lang}): vectorizer error — {e}")
        return pd.DataFrame(
            columns=["term", "tfidf_mean", "tfidf_max", "document_freq"]
        )

    feature_names = vectorizer.get_feature_names_out()
    tfidf_array   = tfidf_matrix.toarray()

    # Compute summary stats per term
    tfidf_mean   = tfidf_array.mean(axis=0)
    tfidf_max    = tfidf_array.max(axis=0)
    doc_freq     = (tfidf_array > 0).sum(axis=0)

    # Sort by mean TF-IDF score
    top_indices = tfidf_mean.argsort()[::-1][:top_n]

    return pd.DataFrame([
        {
            "term":          feature_names[i],
            "tfidf_mean":    round(tfidf_mean[i], 4),
            "tfidf_max":     round(tfidf_max[i],  4),
            "document_freq": int(doc_freq[i]),
        }
        for i in top_indices
    ])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_word_freq(df: pd.DataFrame) -> dict:
    """
    Run word frequency and TF-IDF analyses on participant transcripts.

    Produces 6 sub-tables for Sheet H:
        1. Word frequencies — FR participants
        2. Word frequencies — EN participants
        3. Word frequencies — Friendly condition
        4. Word frequencies — Professional condition
        5. TF-IDF — Friendly vs Professional (FR)
        6. TF-IDF — Friendly vs Professional (EN)

    Args:
        df: DataFrame with columns: transcript, language, tone

    Returns:
        Dict of DataFrames keyed by sub-table name.
        export.py will write each as a labelled section in Sheet H.
    """
    required = ["transcript", "language", "tone"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Sheet H: missing columns {missing}")
        return {}

    results = {}

    # ------------------------------------------------------------------
    # Split by language
    # ------------------------------------------------------------------
    df_fr = df[df["language"] == "FR"].copy()
    df_en = df[df["language"] == "EN"].copy()

    log.info(
        f"Sheet H: {len(df_fr)} FR responses, {len(df_en)} EN responses."
    )

    # ------------------------------------------------------------------
    # 1 & 2 — Word frequencies by language
    # ------------------------------------------------------------------
    if len(df_fr) > 0:
        results["freq_FR"] = _word_frequencies(
            df_fr["transcript"].tolist(), lang="FR"
        )
        results["freq_FR"]["language"] = "FR"
    else:
        log.warning("Sheet H: no FR responses found.")

    if len(df_en) > 0:
        results["freq_EN"] = _word_frequencies(
            df_en["transcript"].tolist(), lang="EN"
        )
        results["freq_EN"]["language"] = "EN"
    else:
        log.warning("Sheet H: no EN responses found.")

    # ------------------------------------------------------------------
    # 3 & 4 — Word frequencies by tone condition
    # ------------------------------------------------------------------
    for tone_val, tone_label in config.TONE_LABELS.items():
        df_tone = df[df["tone"] == tone_val].copy()
        if len(df_tone) == 0:
            continue

        # Detect dominant language in this condition
        lang_counts = df_tone["language"].value_counts()
        dominant_lang = lang_counts.index[0] if len(lang_counts) > 0 else "EN"

        key = f"freq_tone_{tone_label.lower()}"
        results[key] = _word_frequencies(
            df_tone["transcript"].tolist(), lang=dominant_lang
        )
        results[key]["tone"] = tone_label
        log.info(
            f"Sheet H: word frequencies for {tone_label} condition "
            f"({len(df_tone)} participants)."
        )

    # ------------------------------------------------------------------
    # 5 & 6 — TF-IDF by tone condition, split by language
    # ------------------------------------------------------------------
    for lang in ["FR", "EN"]:
        df_lang = df[df["language"] == lang].copy()
        if len(df_lang) < 4:
            log.warning(
                f"Sheet H: TF-IDF ({lang}) — fewer than 4 responses, skipped."
            )
            continue

        df_friendly = df_lang[df_lang["tone"] == 1]
        df_pro      = df_lang[df_lang["tone"] == 0]

        if len(df_friendly) >= 2:
            key = f"tfidf_friendly_{lang}"
            results[key] = _tfidf_analysis(
                df_friendly["transcript"].tolist(), lang=lang
            )
            results[key]["condition"] = "Friendly"
            results[key]["language"]  = lang

        if len(df_pro) >= 2:
            key = f"tfidf_pro_{lang}"
            results[key] = _tfidf_analysis(
                df_pro["transcript"].tolist(), lang=lang
            )
            results[key]["condition"] = "Professional"
            results[key]["language"]  = lang

    log.info(f"Sheet H: {len(results)} sub-tables generated.")
    return results
