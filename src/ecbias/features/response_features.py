"""Explainable measures of how executives respond to analyst questions."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import spacy
import yaml
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - dependency is declared in pyproject.toml
    SentimentIntensityAnalyzer = None

from ..argmining.rules import load_lexicon

_WORD_RE = re.compile(r"\b[\w'-]+\b")
_FINANCE_STOPWORDS = {"company", "year", "quarter", "business", "question", "answer"}


def _words(text: object) -> list[str]:
    return _WORD_RE.findall(str(text or "").lower())


def _phrase_count(text: str, phrases: list[str]) -> int:
    lowered = text.lower()
    return sum(len(re.findall(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered)) for phrase in phrases)


def _has_opener(text: str, phrases: list[str]) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(phrase.lower()) for phrase in phrases)


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else []
    return []


def _is_operator_turn(turn: object) -> bool:
    if not isinstance(turn, dict):
        return False
    value = (
        f"{turn.get('speaker', '')} {turn.get('speaker_type', '')} {turn.get('role', '')}".lower()
    )
    return "operator" in value or "moderator" in value


def _noun_lemmas(text: str, nlp) -> set[str]:
    doc = nlp(text)
    try:
        chunks = doc.noun_chunks
        values = {chunk.root.lemma_.lower() for chunk in chunks if chunk.root.lemma_}
        if values:
            return values
    except (AttributeError, ValueError):
        pass
    return {word for word in _words(text) if len(word) > 2}


def _sentiment_analyzer():
    if SentimentIntensityAnalyzer is None:
        raise RuntimeError("vaderSentiment is required for response sentiment features")
    return SentimentIntensityAnalyzer()


def _load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


def _answer_turn_words(answer_turns: object) -> tuple[int, int]:
    if not isinstance(answer_turns, list):
        return 0, 0
    counts = [
        len(_words(turn.get("text", "")))
        for turn in answer_turns
        if isinstance(turn, dict) and not _is_operator_turn(turn)
    ]
    return sum(counts), max(counts, default=0)


def build_response_features(
    exchanges: pd.DataFrame,
    config_path: str | Path = "configs/feature_spec.yaml",
) -> pd.DataFrame:
    """Return one deterministic response-feature row per exchange."""
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    response_config = spec.get("response_config", {})
    stopwords = {str(word).lower() for word in response_config.get("stopwords", [])}
    stopwords |= _FINANCE_STOPWORDS
    lexicon_root = Path(response_config.get("lexicon_root", "configs/lexicons"))
    courtesy = load_lexicon(lexicon_root / "praise_openers.txt")
    direct = load_lexicon(lexicon_root / "direct_openers.txt")
    deflections = load_lexicon(lexicon_root / "deflections.txt")
    hedges = load_lexicon(lexicon_root / "hedges.txt")
    certainty = load_lexicon(lexicon_root / "certainty.txt")
    nlp = _load_nlp()
    vectorizer = TfidfVectorizer(stop_words=sorted(stopwords), ngram_range=(1, 2))
    corpus = (
        exchanges["question_text"].fillna("").astype(str).tolist()
        + exchanges["answer_text"].fillna("").astype(str).tolist()
    )
    matrix = (
        vectorizer.fit_transform(corpus)
        if any(text.strip() for text in corpus)
        else csr_matrix((len(corpus), 0))
    )
    analyzer = _sentiment_analyzer()
    rows = []
    for index, row in exchanges.iterrows():
        question = str(row.get("question_text", "") or "")
        answer = str(row.get("answer_text", "") or "")
        q_index, a_index = index, index + len(exchanges)
        relevance = (
            float(cosine_similarity(matrix[q_index], matrix[a_index])[0, 0])
            if matrix.shape[1]
            else 0.0
        )
        q_lemmas = _noun_lemmas(question, nlp)
        a_lemmas = _noun_lemmas(answer, nlp)
        keyword_overlap = (
            len(q_lemmas & a_lemmas) / len(q_lemmas | a_lemmas) if q_lemmas | a_lemmas else 0.0
        )
        answer_turns = _as_list(row.get("answer_turns", []))
        answer_words, top_words = _answer_turn_words(answer_turns)
        if not answer_words:
            answer_words = len(_words(answer))
            top_words = answer_words
        question_words = len(_words(question))
        operator_only = (
            isinstance(answer_turns, list)
            and answer_turns
            and all(_is_operator_turn(turn) for turn in answer_turns)
        )
        answer_numeric = sum(any(char.isdigit() for char in word) for word in _words(answer))
        rows.append(
            {
                "exchange_id": row["exchange_id"],
                "relevance_tfidf": relevance,
                "keyword_overlap": keyword_overlap,
                "courtesy_opener": bool(not operator_only and _has_opener(answer, courtesy)),
                "direct_opener": bool(not operator_only and _has_opener(answer, direct)),
                "deflection_rate": _phrase_count(answer, deflections) / max(answer_words, 1),
                "hedge_rate": _phrase_count(answer, hedges) / max(answer_words, 1),
                "certainty_rate": _phrase_count(answer, certainty) / max(answer_words, 1),
                "answer_words": answer_words,
                "question_words": question_words,
                "answer_to_question_ratio": (
                    answer_words / question_words if question_words else float("nan")
                ),
                "answer_numeric_density": (
                    answer_numeric / answer_words * 100 if answer_words else float("nan")
                ),
                "top_responder_word_share": top_words / answer_words
                if answer_words
                else float("nan"),
                "sentiment_question": analyzer.polarity_scores(question)["compound"],
                "sentiment_answer": analyzer.polarity_scores(answer)["compound"],
                "tone_gap": analyzer.polarity_scores(answer)["compound"]
                - analyzer.polarity_scores(question)["compound"],
            }
        )
    return pd.DataFrame(rows)
