"""Tests for docrag.evaluation.text_comparison."""

import pandas as pd
import pytest

from docrag.evaluation import text_comparison

_SCORE_COLUMNS = ["ned", "precision", "recall", "f1"]


def _pairs() -> pd.DataFrame:
    """Returns one identical pair and one candidate missing two words."""
    return pd.DataFrame(
        {"reference": ["a b", "a b c"], "candidate": ["a b", "a"]}
    )


def _scores() -> pd.DataFrame:
    """Returns scores of two pairs in one group and one in another.

    Only the pair in group y has a blank candidate and a reference that is
    not blank.
    """
    return pd.DataFrame(
        {
            "group": ["y", "x", "x"],
            "reference": ["c", "a b", ""],
            "candidate": [" ", "a b", ""],
            "ned": [0.5, 0.0, 0.2],
            "precision": [0.5, 1.0, 0.6],
            "recall": [0.25, 1.0, 0.4],
            "f1": [0.4, 1.0, 0.8],
        }
    )


def test_score_text_pairs_scores() -> None:
    scores = text_comparison.score_text_pairs(_pairs())

    expected = _pairs().assign(
        ned=[0.0, 4 / 5],
        precision=[1.0, 1.0],
        recall=[1.0, 1 / 3],
        f1=[1.0, 0.5],
    )
    pd.testing.assert_frame_equal(scores, expected)


def test_score_text_pairs_normalises() -> None:
    pairs = pd.DataFrame({"reference": ["a  b"], "candidate": [" a b "]})

    scores = text_comparison.score_text_pairs(pairs)

    assert scores.loc[0, "ned"] == 0.0
    assert scores.loc[0, "f1"] == 1.0


def test_score_text_pairs_keeps_columns() -> None:
    pairs = pd.DataFrame(
        {
            "group": ["x", "y"],
            "reference": ["a b", "c"],
            "candidate": ["a b", "c"],
        },
        index=[20, 10],
    )

    scores = text_comparison.score_text_pairs(pairs)

    assert list(scores.columns) == [*pairs.columns, *_SCORE_COLUMNS]
    pd.testing.assert_frame_equal(scores[list(pairs.columns)], pairs)


def test_score_text_pairs_input_unchanged() -> None:
    pairs = _pairs()

    text_comparison.score_text_pairs(pairs)

    pd.testing.assert_frame_equal(pairs, _pairs())


def test_score_text_pairs_no_rows() -> None:
    pairs = _pairs().iloc[0:0]

    scores = text_comparison.score_text_pairs(pairs)

    assert scores.empty
    assert list(scores.columns) == ["reference", "candidate", *_SCORE_COLUMNS]
    assert (scores[_SCORE_COLUMNS].dtypes == "float64").all()


@pytest.mark.parametrize("column", ["reference", "candidate"])
def test_score_text_pairs_missing_column(column: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        text_comparison.score_text_pairs(_pairs().drop(columns=[column]))

    assert str(excinfo.value) == f"pairs is missing columns ['{column}']"


@pytest.mark.parametrize("column", ["reference", "candidate"])
def test_score_text_pairs_null_text(column: str) -> None:
    pairs = _pairs()
    pairs[column] = pd.Series(["a b", None], dtype="str")

    with pytest.raises(ValueError) as excinfo:
        text_comparison.score_text_pairs(pairs)

    assert (
        str(excinfo.value)
        == f"pairs has missing values in columns ['{column}']"
    )


@pytest.mark.parametrize("column", ["reference", "candidate"])
def test_score_text_pairs_non_text_value(column: str) -> None:
    pairs = _pairs()
    pairs[column] = pd.Series(["a b", 5], dtype="object")

    with pytest.raises(ValueError) as excinfo:
        text_comparison.score_text_pairs(pairs)

    assert str(excinfo.value) == (
        f"pairs has non-text values in columns ['{column}']"
    )


@pytest.mark.parametrize("column", _SCORE_COLUMNS)
def test_score_text_pairs_existing_score_column(column: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        text_comparison.score_text_pairs(_pairs().assign(**{column: 0.0}))

    assert str(excinfo.value) == f"pairs already has score columns ['{column}']"


@pytest.mark.parametrize(
    "by",
    ["group", pytest.param("empty_candidate", id="empty-candidate"), "self"],
)
def test_summarise_text_scores_groups(by: str) -> None:
    scores = _scores().rename(columns={"group": by})
    summary = text_comparison.summarise_text_scores(scores, by)

    expected = pd.DataFrame(
        {
            "count": [2, 1, 3],
            "empty_candidates": [0, 1, 1],
            "ned_mean": [0.1, 0.5, 0.7 / 3],
            "ned_median": [0.1, 0.5, 0.2],
            "ned_p90": [0.18, 0.5, 0.44],
            "precision_mean": [0.8, 0.5, 0.7],
            "precision_median": [0.8, 0.5, 0.6],
            "precision_p10": [0.64, 0.5, 0.52],
            "recall_mean": [0.7, 0.25, 0.55],
            "recall_median": [0.7, 0.25, 0.4],
            "recall_p10": [0.46, 0.25, 0.28],
            "f1_mean": [0.9, 0.4, 2.2 / 3],
            "f1_median": [0.9, 0.4, 0.8],
            "f1_p10": [0.82, 0.4, 0.48],
        },
        index=pd.Index(["x", "y", "overall"], name=by),
    )
    pd.testing.assert_frame_equal(summary, expected)


def test_summarise_text_scores_named_index() -> None:
    scores = _scores()
    expected = text_comparison.summarise_text_scores(scores, "group")

    summary = text_comparison.summarise_text_scores(
        scores.set_index("group", drop=False), "group"
    )

    pd.testing.assert_frame_equal(summary, expected)


@pytest.mark.parametrize("column", ["reference", "candidate"])
def test_summarise_text_scores_categorical_text(column: str) -> None:
    scores = _scores().iloc[[0]].copy()
    expected = text_comparison.summarise_text_scores(scores, "group")
    scores[column] = scores[column].astype("category")

    summary = text_comparison.summarise_text_scores(scores, "group")

    pd.testing.assert_frame_equal(summary, expected)


@pytest.mark.parametrize(
    "column", ["group", "reference", "candidate", *_SCORE_COLUMNS]
)
def test_summarise_text_scores_missing_column(column: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        text_comparison.summarise_text_scores(
            _scores().drop(columns=[column]), "group"
        )

    assert str(excinfo.value) == f"scores is missing columns ['{column}']"


@pytest.mark.parametrize("by", ["reference", "candidate", *_SCORE_COLUMNS])
def test_summarise_text_scores_by_required_column(by: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        text_comparison.summarise_text_scores(_scores(), by)

    assert str(excinfo.value) == f"'{by}': is a text or score column"


@pytest.mark.parametrize(
    "value, reason",
    [
        pytest.param(5, "non-text values", id="number"),
        pytest.param(None, "missing values", id="missing-value"),
    ],
)
@pytest.mark.parametrize("column", ["reference", "candidate"])
def test_summarise_text_scores_invalid_text(
    column: str, value: object, reason: str
) -> None:
    scores = _scores()
    scores[column] = pd.Series(["a b", value, ""], dtype="object")

    with pytest.raises(ValueError) as excinfo:
        text_comparison.summarise_text_scores(scores, "group")

    assert str(excinfo.value) == f"scores has {reason} in columns ['{column}']"


def test_summarise_text_scores_no_rows() -> None:
    with pytest.raises(ValueError) as excinfo:
        text_comparison.summarise_text_scores(_scores().iloc[0:0], "group")

    assert str(excinfo.value) == "scores is empty"


def test_summarise_text_scores_null_group_value() -> None:
    scores = _scores().assign(group=pd.Series(["x", None, "x"], dtype="str"))

    with pytest.raises(ValueError) as excinfo:
        text_comparison.summarise_text_scores(scores, "group")

    assert str(excinfo.value) == "'group': column has missing values"


def test_summarise_text_scores_overall_value() -> None:
    scores = _scores().assign(group=["x", "overall", "x"])

    with pytest.raises(ValueError) as excinfo:
        text_comparison.summarise_text_scores(scores, "group")

    assert str(excinfo.value) == "'group': column contains 'overall'"
