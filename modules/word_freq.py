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
import os
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
    Run word frequency and TF-IDF analyses.

    Returns dict with:
        'freq_table'  — combined frequency table (all 4 conditions side by side)
        'tfidf_table' — combined Delta TF-IDF table
        'wordcloud_paths' — list of PNG file paths (generated separately)
    """
    required = ["transcript", "tone"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Sheet H: missing columns {missing}")
        return {}

    # Extract texts per condition and speaker
    sections = {
        "participant_friendly": (df[df["tone"]==1], "Participant"),
        "participant_pro":      (df[df["tone"]==0], "Participant"),
        "chatbot_friendly":     (df[df["tone"]==1], "Chatbot"),
        "chatbot_pro":          (df[df["tone"]==0], "Chatbot"),
    }

    texts = {}
    for key, (df_cond, speaker) in sections.items():
        raw = [
            _extract_messages(t, speaker)
            for t in df_cond["transcript"].tolist()
        ]
        texts[key] = [t for t in raw if t.strip()]

    # ------------------------------------------------------------------
    # Table 1 — Word frequencies side by side
    # Columns: Rank | Word FL21 Part | Freq | Word FL22 Part | Freq |
    #          Word Chatbot FL21 | Freq | Word Chatbot FL22 | Freq
    # ------------------------------------------------------------------
    freq_data = {}
    for key, text_list in texts.items():
        freq_data[key] = _word_frequencies(text_list, top_n=50)

    max_rows = max(len(v) for v in freq_data.values()) if freq_data else 0
    freq_rows = []
    for i in range(max_rows):
        row = {"Rank": i + 1}
        for key, label in [
            ("participant_friendly", "Word FL_21 (Participant)"),
            ("participant_pro",      "Word FL_22 (Participant)"),
            ("chatbot_friendly",     "Word Chatbot FL_21"),
            ("chatbot_pro",          "Word Chatbot FL_22"),
        ]:
            df_freq = freq_data.get(key, pd.DataFrame())
            if i < len(df_freq):
                row[label]            = df_freq.iloc[i]["word"]
                row[f"Freq_{label}"]  = int(df_freq.iloc[i]["frequency"])
            else:
                row[label]           = ""
                row[f"Freq_{label}"] = ""
        freq_rows.append(row)

    freq_table = pd.DataFrame(freq_rows)
    # Reorder columns
    freq_cols = ["Rank"]
    for label in [
        "Word FL_21 (Participant)", "Word FL_22 (Participant)",
        "Word Chatbot FL_21",       "Word Chatbot FL_22",
    ]:
        freq_cols += [label, f"Freq_{label}"]
    freq_table = freq_table[
        [c for c in freq_cols if c in freq_table.columns]
    ]

    # ------------------------------------------------------------------
    # Table 2 — Delta TF-IDF
    # Delta = TF-IDF score friendly - TF-IDF score professional
    # Positive = more distinctive in friendly
    # Negative = more distinctive in professional
    # ------------------------------------------------------------------
    tfidf_data = {}
    for key, text_list in texts.items():
        tfidf_data[key] = _tfidf_analysis(text_list, top_n=50)

    def _compute_delta(df_friendly: pd.DataFrame,
                       df_pro: pd.DataFrame) -> pd.DataFrame:
        """Compute Delta TF-IDF = friendly score - pro score."""
        if df_friendly.empty or df_pro.empty:
            return pd.DataFrame()

        merged = pd.merge(
            df_friendly[["term","tfidf_mean"]].rename(
                columns={"tfidf_mean": "tfidf_friendly"}
            ),
            df_pro[["term","tfidf_mean"]].rename(
                columns={"tfidf_mean": "tfidf_pro"}
            ),
            on="term", how="outer"
        ).fillna(0)

        merged["delta"] = (
            merged["tfidf_friendly"] - merged["tfidf_pro"]
        ).round(4)
        merged = merged.sort_values("delta", ascending=False)
        return merged

    delta_part = _compute_delta(
        tfidf_data.get("participant_friendly", pd.DataFrame()),
        tfidf_data.get("participant_pro",      pd.DataFrame()),
    )
    delta_chat = _compute_delta(
        tfidf_data.get("chatbot_friendly", pd.DataFrame()),
        tfidf_data.get("chatbot_pro",      pd.DataFrame()),
    )

    # Build side-by-side delta table
    max_delta = max(
        len(delta_part) if not delta_part.empty else 0,
        len(delta_chat) if not delta_chat.empty else 0,
    )
    tfidf_rows = []
    for i in range(min(max_delta, 30)):
        row = {"Rank": i + 1}
        if i < len(delta_part):
            r = delta_part.iloc[i]
            row["Distinctive FL_21 (Participant)"] = r["term"]
            row["Delta TF-IDF (Participant)"]      = r["delta"]
        else:
            row["Distinctive FL_21 (Participant)"] = ""
            row["Delta TF-IDF (Participant)"]      = ""
        if i < len(delta_chat):
            r = delta_chat.iloc[i]
            row["Distinctive FL_21 (Chatbot)"] = r["term"]
            row["Delta TF-IDF (Chatbot)"]      = r["delta"]
        else:
            row["Distinctive FL_21 (Chatbot)"] = ""
            row["Delta TF-IDF (Chatbot)"]      = ""
        tfidf_rows.append(row)

    tfidf_table = pd.DataFrame(tfidf_rows)

    # ------------------------------------------------------------------
    # Wordclouds — generate PNG files
    # ------------------------------------------------------------------
    wordcloud_paths = _generate_wordclouds(texts)

    log.info(
        f"Sheet H: freq table {len(freq_table)} rows, "
        f"tfidf table {len(tfidf_table)} rows, "
        f"{len(wordcloud_paths)} wordclouds generated."
    )

    return {
        "freq_table":      freq_table,
        "tfidf_table":     tfidf_table,
        "wordcloud_paths": wordcloud_paths,
    }


def _generate_wordclouds(texts: dict) -> list:
    """
    Generate wordcloud PNG files for each condition/speaker.
    Returns list of file paths.
    """
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning(
            "wordcloud library not installed. "
            "Run: pip install wordcloud"
        )
        return []

    os.makedirs("outputs/wordclouds", exist_ok=True)

    colors = {
        "participant_friendly": "#1A5276",
        "participant_pro":      "#784212",
        "chatbot_friendly":     "#1D6A39",
        "chatbot_pro":          "#4A235A",
    }

    titles = {
        "participant_friendly": "Participant Messages — Friendly",
        "participant_pro":      "Participant Messages — Professional",
        "chatbot_friendly":     "Chatbot Messages — Friendly",
        "chatbot_pro":          "Chatbot Messages — Professional",
    }

    paths = []
    for key, text_list in texts.items():
        if not text_list:
            continue

        # Combine all texts and tokenize
        combined = " ".join(text_list)
        tokens   = _tokenize(combined)
        if not tokens:
            continue

        token_text = " ".join(tokens)

        try:
            wc = WordCloud(
                width=800, height=400,
                background_color="white",
                colormap="Blues",
                max_words=100,
                stopwords=ALL_STOPWORDS,
                collocations=False,
            ).generate(token_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(
                titles.get(key, key),
                fontsize=14, fontweight="bold",
                pad=15
            )

            path = f"outputs/wordclouds/{key}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            paths.append(path)
            log.info(f"Wordcloud saved: {path}")

        except Exception as e:
            log.warning(f"Wordcloud {key}: {e}")

    return paths
