"""
word_freq.py
------------
Step 5 of the pipeline: word frequencies and TF-IDF analysis.

Four sections per output:
    1. Participant messages — Friendly condition
    2. Participant messages — Professional condition
    3. Chatbot messages — Friendly condition
    4. Chatbot messages — Professional condition

Bilingual FR+EN processing:
    - Combined stopwords FR + EN + custom study-specific terms
    - TF-IDF on full corpus (no language split — honest with bilingual data)
    - Top 50 word frequencies + top 30 TF-IDF terms per section
"""

import logging
import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

import config

log = logging.getLogger(__name__)

# Download stopwords if needed
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Stopwords — FR + EN + custom
# ---------------------------------------------------------------------------
FR_STOP = set(stopwords.words("french"))
EN_STOP = set(stopwords.words("english"))

CUSTOM_STOPWORDS = {
    # Contractions FR
    "c'est","c'était","j'ai","j'avais","j'aime","j'aimais","j'aimerais",
    "j'aimerai","j'utilise","j'utilisais","j'aurais","j'espère","j'adore",
    "n'est","n'a","n'y","qu'il","qu'elle","qu'on","qu'ils","qu'elles",
    "qu'une","qu'un","lorsqu'il","lorsqu'on","puisqu'il","puisqu'on",
    "t'as","t'en","c'en","s'il","s'en","m'a","m'en","d'un","d'une",
    "d'abord","d'accord","d'ailleurs","d'autres","d'avoir","est-ce",
    "qu'est-ce","a-t-il","avez-vous","pouvez-vous",
    # Contractions EN
    "i'm","i've","i'd","i'll","it's","it'd","it'll","that's","that'd",
    "don't","didn't","doesn't","won't","wouldn't","can't","couldn't",
    "isn't","aren't","wasn't","weren't","there's","there'd","they're",
    "they've","they'd","they'll","we're","we've","we'd","we'll",
    "you're","you've","you'd","you'll","he's","she's","let's","who's",
    # Generic FR fillers
    "merci","donc","alors","voilà","voila","bah","ben","bof","hein",
    "quoi","hm","ah","oh","eh","bon","nan","ouais","ouai","yep",
    "lorsque","puisque","tandis","pourtant","cependant","néanmoins",
    "toutefois","notamment","également","ainsi","puis","ensuite","enfin",
    "surtout","vraiment","totalement","complètement","absolument",
    "franchement","honnêtement","clairement","exactement","effectivement",
    "parfois","souvent","toujours","jamais","encore","déjà","maintenant",
    "aujourd","hui","demain","hier","fois","moment","temps","jour","jours",
    # Generic EN fillers
    "merci","thank","thanks","thankyou","sharing","shared",
    "okay","ok","yeah","yep","nope","well","actually","basically",
    "honestly","literally","definitely","absolutely","exactly","clearly",
    "sometimes","often","always","never","already","now","today",
    "tomorrow","yesterday","time","moment","day","days","thing","things",
    "something","someone","anything","nothing","everything","someone",
    "much","many","every","really","just","maybe","sure","though",
    "bit","lot","one","two","three","also","even","still","back",
    "kind","sort","quite","rather","pretty","very","too","so","as",
    # Study-specific terms (too generic to be informative)
    "soundflow","platform","plateforme","application","app","streaming",
    "spotify","apple","youtube","music","musique","song","songs",
    "musiques","services","chatbot","survey","questionnaire",
    "question","questions","réponse","réponses","answer","answers",
    # Action verbs FR (too generic)
    "faire","avoir","être","aller","venir","voir","savoir","pouvoir",
    "vouloir","devoir","mettre","prendre","donner","parler","penser",
    "trouver","utiliser","utilise","utilisez","utilises","ajouter",
    "permet","partager","partage","apprécier","appréciez","apprécies",
    "apprécie","commenter","commentaires","suggestions","connais","sais",
    "laisser","essayer","chercher","regarder","écouter","continuer",
    "changer","améliorer","proposer","demander","répondre","expliquer",
    # Action verbs EN (too generic)
    "make","makes","making","want","wanted","get","got","give","say",
    "said","come","go","going","would","could","should","might","must",
    "need","needs","needed","try","tried","look","looking","feel",
    "feels","felt","seem","seems","seemed","mean","means","meant",
    "know","knew","think","thought","find","found","use","used","using",
    "let","put","take","keep","keep","start","stop","help","show",
    # Pronouns and articles already in NLTK but adding variants
    "je","tu","il","elle","on","nous","vous","ils","elles",
    "le","la","les","un","une","des","du","de","en","au","aux",
    "et","ou","ni","car","donc","or","mais","the","this","that",
    "these","those","and","but","for","with","from","have","has",
    "are","was","were","been","being","its","it","they","their",
    "them","our","your","his","her","you","me","my","we","us",
    "oui","non","yes","no","nan",
}

ALL_STOPWORDS = FR_STOP | EN_STOP | CUSTOM_STOPWORDS


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Lowercase, remove punctuation and numbers, normalise whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\sàâäéèêëîïôùûüçœæ]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_messages(transcript: str, speaker: str) -> str:
    """
    Extract messages from a specific speaker in a transcript.

    Args:
        transcript: Full conversation transcript string
        speaker:    "Participant" or "Chatbot"

    Returns:
        Concatenated messages from that speaker
    """
    tag   = f"[{speaker}]"
    lines = transcript.split("\n")
    msgs  = [
        line.replace(tag, "").strip()
        for line in lines
        if line.startswith(tag)
    ]
    return " ".join(msgs)


def _tokenize(text: str) -> list:
    """Tokenize and remove stopwords. Min token length = 3."""
    tokens = _clean_text(text).split()
    return [t for t in tokens if t not in ALL_STOPWORDS and len(t) >= 3]


# ---------------------------------------------------------------------------
# Word frequency
# ---------------------------------------------------------------------------
def _word_frequencies(texts: list, top_n: int = 50) -> pd.DataFrame:
    """
    Compute word frequencies for a list of texts.

    Returns DataFrame with columns: rank, word, frequency
    """
    all_tokens = []
    for text in texts:
        all_tokens.extend(_tokenize(text))

    if not all_tokens:
        return pd.DataFrame(columns=["rank", "word", "frequency"])

    counts = Counter(all_tokens)
    top    = counts.most_common(top_n)

    return pd.DataFrame(
        [{"rank": i+1, "word": w, "frequency": f} for i, (w, f) in enumerate(top)]
    )


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
def _tfidf_analysis(texts: list, top_n: int = 30) -> pd.DataFrame:
    """
    Compute TF-IDF scores for a corpus.

    Returns DataFrame with columns: rank, term, tfidf_mean, tfidf_max, document_freq
    """
    cleaned = [_clean_text(t) for t in texts if t.strip()]
    cleaned = [t for t in cleaned if t.strip()]

    if len(cleaned) < 2:
        log.warning("TF-IDF: fewer than 2 documents — skipped.")
        return pd.DataFrame(
            columns=["rank", "term", "tfidf_mean", "tfidf_max", "document_freq"]
        )

    vectorizer = TfidfVectorizer(
        stop_words=list(ALL_STOPWORDS),
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
        token_pattern=(
            r"(?u)\b[a-zA-ZàâäéèêëîïôùûüçœæÀÂÄÉÈÊËÎÏÔÙÛÜÇŒÆ]{3,}\b"
        ),
    )

    try:
        matrix       = vectorizer.fit_transform(cleaned)
    except ValueError as e:
        log.warning(f"TF-IDF vectorizer error: {e}")
        return pd.DataFrame(
            columns=["rank", "term", "tfidf_mean", "tfidf_max", "document_freq"]
        )

    features     = vectorizer.get_feature_names_out()
    arr          = matrix.toarray()
    tfidf_mean   = arr.mean(axis=0)
    tfidf_max    = arr.max(axis=0)
    doc_freq     = (arr > 0).sum(axis=0)
    top_indices  = tfidf_mean.argsort()[::-1][:top_n]

    return pd.DataFrame([
        {
            "rank":         i + 1,
            "term":         features[idx],
            "tfidf_mean":   round(tfidf_mean[idx], 4),
            "tfidf_max":    round(tfidf_max[idx],  4),
            "document_freq": int(doc_freq[idx]),
        }
        for i, idx in enumerate(top_indices)
    ])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_word_freq(df: pd.DataFrame) -> dict:
    """
    Run word frequency and TF-IDF analyses on conversation transcripts.

    Produces 4 sections:
        1. Participant messages — Friendly condition
        2. Participant messages — Professional condition
        3. Chatbot messages — Friendly condition
        4. Chatbot messages — Professional condition

    Each section contains:
        - Word frequency table (top 50)
        - TF-IDF table (top 30)

    Args:
        df: DataFrame with columns: transcript, tone

    Returns:
        Dict of DataFrames keyed by section name.
    """
    required = ["transcript", "tone"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Sheet H: missing columns {missing}")
        return {}

    results = {}

    sections = [
        ("participant_friendly", "Participant", 1, "Friendly"),
        ("participant_pro",      "Participant", 0, "Professional"),
        ("chatbot_friendly",     "Chatbot",     1, "Friendly"),
        ("chatbot_pro",          "Chatbot",     0, "Professional"),
    ]

    for key, speaker, tone_val, tone_label in sections:
        df_cond = df[df["tone"] == tone_val].copy()

        if len(df_cond) == 0:
            log.warning(f"Sheet H: no data for {tone_label} condition — skipped.")
            continue

        # Extract messages for this speaker
        texts = [
            _extract_messages(t, speaker)
            for t in df_cond["transcript"].tolist()
        ]
        texts = [t for t in texts if t.strip()]

        if not texts:
            log.warning(f"Sheet H: no {speaker} messages in {tone_label} — skipped.")
            continue

        log.info(
            f"Sheet H: {speaker} / {tone_label} — "
            f"{len(texts)} conversations"
        )

        # Word frequencies
        freq_df = _word_frequencies(texts)
        freq_df["speaker"]   = speaker
        freq_df["condition"] = tone_label

        # TF-IDF
        tfidf_df = _tfidf_analysis(texts)
        tfidf_df["speaker"]   = speaker
        tfidf_df["condition"] = tone_label

        results[f"{key}_freq"]  = freq_df
        results[f"{key}_tfidf"] = tfidf_df

    log.info(f"Sheet H: {len(results)} sub-tables generated.")
    return results
