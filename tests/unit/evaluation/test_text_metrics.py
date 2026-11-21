"""Tests for docrag.evaluation.text_metrics."""

import dataclasses

import pytest

from docrag.evaluation import text_metrics


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("file A1", "file A1", id="plain-text"),
        pytest.param(
            "\N{LATIN SMALL LIGATURE FI}le "
            "\N{FULLWIDTH LATIN CAPITAL LETTER A}\N{FULLWIDTH DIGIT ONE}",
            "file A1",
            id="ligature-and-fullwidth",
        ),
        pytest.param(
            "e\N{COMBINING ACUTE ACCENT}",
            "\N{LATIN SMALL LETTER E WITH ACUTE}",
            id="combining-accent",
        ),
    ],
)
def test_normalise_text_compatibility_forms(text: str, expected: str) -> None:
    assert text_metrics.normalise_text(text) == expected


@pytest.mark.parametrize(
    "whitespace",
    [
        pytest.param(" ", id="one-space"),
        pytest.param("   ", id="several-spaces"),
        pytest.param("\t", id="tab"),
        pytest.param("\n", id="newline"),
        pytest.param(" \t\n ", id="mixed"),
        pytest.param("\r\n", id="carriage-return-line-feed"),
        pytest.param("\x0b\x0c", id="vertical-tab-form-feed"),
    ],
)
def test_normalise_text_whitespace_runs(whitespace: str) -> None:
    assert text_metrics.normalise_text(f"a{whitespace}b") == "a b"


@pytest.mark.parametrize(
    "whitespace",
    [
        pytest.param("\N{NO-BREAK SPACE}", id="no-break-space"),
        pytest.param("\N{IDEOGRAPHIC SPACE}", id="ideographic-space"),
        pytest.param("\N{EM SPACE}", id="em-space"),
        pytest.param(
            "\N{LINE SEPARATOR}\N{PARAGRAPH SEPARATOR}",
            id="line-and-paragraph-separators",
        ),
    ],
)
def test_normalise_text_unicode_spaces(whitespace: str) -> None:
    assert text_metrics.normalise_text(f"a{whitespace}b") == "a b"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("a b", id="no-whitespace-at-ends"),
        pytest.param(" a b", id="leading-space"),
        pytest.param("a b ", id="trailing-space"),
        pytest.param("  a b \n", id="both-ends"),
    ],
)
def test_normalise_text_ends(text: str) -> None:
    assert text_metrics.normalise_text(text) == "a b"


def test_normalise_text_case_and_punctuation() -> None:
    assert text_metrics.normalise_text("Hello, World!") == "Hello, World!"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param(" \n ", id="whitespace-only"),
    ],
)
def test_normalise_text_blank(text: str) -> None:
    assert text_metrics.normalise_text(text) == ""


def test_normalised_edit_distance_identical() -> None:
    assert text_metrics.normalised_edit_distance("page", "page") == 0.0


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        pytest.param("page", "pages", 1 / 5, id="one-insertion"),
        pytest.param("kitten", "sitting", 3 / 7, id="kitten-sitting"),
        pytest.param("ab", "abcd", 1 / 2, id="different-lengths"),
        pytest.param("\N{GRINNING FACE}", "x", 1.0, id="character-outside-bmp"),
    ],
)
def test_normalised_edit_distance_known_value(
    reference: str, candidate: str, expected: float
) -> None:
    assert text_metrics.normalised_edit_distance(
        reference, candidate
    ) == pytest.approx(expected)


def test_normalised_edit_distance_disjoint() -> None:
    assert text_metrics.normalised_edit_distance("abc", "xyz") == 1.0


def test_normalised_edit_distance_symmetric() -> None:
    assert text_metrics.normalised_edit_distance(
        "kitten", "sitting"
    ) == text_metrics.normalised_edit_distance("sitting", "kitten")


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        pytest.param("", "", 0.0, id="both-empty"),
        pytest.param("", "ab", 1.0, id="empty-reference"),
        pytest.param("ab", "", 1.0, id="empty-candidate"),
    ],
)
def test_normalised_edit_distance_empty(
    reference: str, candidate: str, expected: float
) -> None:
    assert (
        text_metrics.normalised_edit_distance(reference, candidate) == expected
    )


def test_normalised_edit_distance_case_sensitive() -> None:
    assert text_metrics.normalised_edit_distance("a", "A") == 1.0


def test_bag_of_words_scores_identical() -> None:
    scores = text_metrics.bag_of_words_scores("a b c", "a b c")

    assert dataclasses.astuple(scores) == (1.0, 1.0, 1.0)


def test_bag_of_words_scores_partial_overlap() -> None:
    scores = text_metrics.bag_of_words_scores("a b c d", "a b x")

    assert dataclasses.astuple(scores) == pytest.approx((2 / 3, 1 / 2, 4 / 7))


def test_bag_of_words_scores_repeated_words() -> None:
    scores = text_metrics.bag_of_words_scores("a a b", "a a a")

    assert dataclasses.astuple(scores) == pytest.approx((2 / 3, 2 / 3, 2 / 3))


def test_bag_of_words_scores_word_order() -> None:
    scores = text_metrics.bag_of_words_scores("a b", "b a")

    assert dataclasses.astuple(scores) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        pytest.param("a b", "c d", id="different-words"),
        pytest.param("Word", "word", id="different-case"),
        pytest.param("world,", "world", id="different-punctuation"),
    ],
)
def test_bag_of_words_scores_no_overlap(reference: str, candidate: str) -> None:
    scores = text_metrics.bag_of_words_scores(reference, candidate)

    assert dataclasses.astuple(scores) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        pytest.param("", "", 1.0, id="both-empty"),
        pytest.param("", "a", 0.0, id="empty-reference"),
        pytest.param("a", "", 0.0, id="empty-candidate"),
        pytest.param(" \n", "", 1.0, id="whitespace-and-empty"),
        pytest.param(" ", "a", 0.0, id="whitespace-reference"),
        pytest.param("a", "\t", 0.0, id="whitespace-candidate"),
    ],
)
def test_bag_of_words_scores_empty(
    reference: str, candidate: str, expected: float
) -> None:
    scores = text_metrics.bag_of_words_scores(reference, candidate)

    assert dataclasses.astuple(scores) == (expected, expected, expected)


def test_bag_of_words_scores_frozen() -> None:
    scores = text_metrics.bag_of_words_scores("a", "a")

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(scores, "f1", 0.0)


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        pytest.param("Hello, world", "Hello, world", id="already-normalised"),
        pytest.param(
            " Hello,\N{NO-BREAK SPACE} world\n",
            "Hello, world",
            id="whitespace",
        ),
        pytest.param(
            "\N{LATIN SMALL LIGATURE FI}le", "file", id="compatibility-forms"
        ),
    ],
)
def test_evaluate_ocr_text_normalises_both_texts(
    reference: str, candidate: str
) -> None:
    scores = text_metrics.evaluate_ocr_text(reference, candidate)

    assert scores.ned == 0.0
    assert dataclasses.astuple(scores.bow) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("reference", "candidate", "expected_ned", "expected_bow"),
    [
        pytest.param("a b c", "a b c", 0.0, (1.0, 1.0, 1.0), id="identical"),
        pytest.param(
            "a b c d",
            "a b x",
            3 / 7,
            (2 / 3, 1 / 2, 4 / 7),
            id="partial-overlap",
        ),
        pytest.param("", "", 0.0, (1.0, 1.0, 1.0), id="both-empty"),
        pytest.param("", "a b", 1.0, (0.0, 0.0, 0.0), id="empty-reference"),
        pytest.param("a b", "", 1.0, (0.0, 0.0, 0.0), id="empty-candidate"),
        pytest.param(
            "a b", " \n ", 1.0, (0.0, 0.0, 0.0), id="whitespace-candidate"
        ),
    ],
)
def test_evaluate_ocr_text_scores(
    reference: str,
    candidate: str,
    expected_ned: float,
    expected_bow: tuple[float, float, float],
) -> None:
    scores = text_metrics.evaluate_ocr_text(reference, candidate)

    assert scores.ned == pytest.approx(expected_ned)
    assert dataclasses.astuple(scores.bow) == pytest.approx(expected_bow)
