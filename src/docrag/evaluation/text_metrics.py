"""Text metrics that compare extracted text with ground-truth text."""

import collections
import dataclasses
import re
import unicodedata

import rapidfuzz


def normalise_text(text: str) -> str:
    """Returns a text in a canonical form for comparison.

    The result is in Unicode NFKC form, with whitespace runs collapsed to
    one space and none at the ends. Case and punctuation are kept.

    Args:
        text: The text to normalise.

    Returns:
        The normalised text.
    """
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def normalised_edit_distance(reference: str, candidate: str) -> float:
    """Returns the character edit distance between two texts, from 0 to 1.

    The distance is the Levenshtein distance divided by the longer text's
    length.

    Args:
        reference: The ground-truth text.
        candidate: The text to score.

    Returns:
        The distance. Identical texts, including two empty texts, give 0.0.
    """
    return rapidfuzz.distance.Levenshtein.normalized_distance(
        reference, candidate
    )


@dataclasses.dataclass(frozen=True)
class BagOfWordsScores:
    """How well the words of a candidate text match a reference text.

    Attributes:
        precision: Shared words over candidate words.
        recall: Shared words over reference words.
        f1: Harmonic mean of precision and recall, 0.0 when both are 0.0.
    """

    precision: float
    recall: float
    f1: float


def bag_of_words_scores(reference: str, candidate: str) -> BagOfWordsScores:
    """Scores the words of a candidate text against a reference text.

    Words are split on whitespace and counted with repeats, and their order
    is ignored. If both texts have no words, all scores are 1.0. If only one
    has none, all are 0.0.

    Args:
        reference: The ground-truth text.
        candidate: The text to score.

    Returns:
        The precision, recall, and F1 of the shared words.
    """
    reference_words = reference.split()
    candidate_words = candidate.split()

    if not reference_words and not candidate_words:
        return BagOfWordsScores(precision=1.0, recall=1.0, f1=1.0)
    if not reference_words or not candidate_words:
        return BagOfWordsScores(precision=0.0, recall=0.0, f1=0.0)

    shared = sum(
        (
            collections.Counter(reference_words)
            & collections.Counter(candidate_words)
        ).values()
    )
    precision = shared / len(candidate_words)
    recall = shared / len(reference_words)
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return BagOfWordsScores(precision=precision, recall=recall, f1=f1)


@dataclasses.dataclass(frozen=True)
class OCRSampleScores:
    """Scores of a candidate text against a reference text.

    Attributes:
        ned: Normalised edit distance, from 0.0 for identical texts to 1.0.
        bow: Precision, recall, and F1 of the shared words.
    """

    ned: float
    bow: BagOfWordsScores


def evaluate_ocr_text(reference: str, candidate: str) -> OCRSampleScores:
    """Scores a candidate text against a reference text.

    Both texts are scored in normalised form: Unicode NFKC, whitespace runs
    collapsed to one space and none at the ends, case and punctuation kept.

    Args:
        reference: The ground-truth text.
        candidate: The text to score.

    Returns:
        The normalised edit distance and the bag-of-words scores.
    """
    normalised_reference = normalise_text(reference)
    normalised_candidate = normalise_text(candidate)

    ned = normalised_edit_distance(normalised_reference, normalised_candidate)
    bow = bag_of_words_scores(normalised_reference, normalised_candidate)

    return OCRSampleScores(ned=ned, bow=bow)
