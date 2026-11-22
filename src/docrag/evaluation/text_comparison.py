"""Scores of candidate texts against reference texts, and their summaries."""

from collections.abc import Callable

import pandas as pd

from docrag.evaluation import text_metrics

_TEXT_COLUMNS = ("reference", "candidate")
_SCORE_COLUMNS = ("ned", "precision", "recall", "f1")

# A statistic for a named aggregation: a pandas function name, or a
# function of a column's values.
_Statistic = str | Callable[["pd.Series[float]"], object]


def _text_columns_with_missing_values(frame: pd.DataFrame) -> list[str]:
    """Lists the text columns with missing values.

    Args:
        frame: Rows with reference and candidate columns.

    Returns:
        The names of the affected columns.
    """
    return [column for column in _TEXT_COLUMNS if frame[column].isna().any()]


def _non_text_columns(frame: pd.DataFrame) -> list[str]:
    """Lists the text columns that contain a value other than a string.

    Args:
        frame: Rows with reference and candidate columns.

    Returns:
        The names of the affected columns.
    """
    return [
        column
        for column in _TEXT_COLUMNS
        if not all(isinstance(value, str) for value in frame[column])
    ]


def score_text_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Scores the candidate text of each pair against its reference text.

    Both texts are scored in normalised form: Unicode NFKC, whitespace runs
    collapsed to one space and none at the ends, case and punctuation kept.
    pairs is not changed.

    Args:
        pairs: One row per pair, with the string columns reference and
            candidate.

    Returns:
        A copy of pairs with the float columns ned, precision, recall, and
        f1 added. ned is the normalised edit distance, from 0.0 for
        identical texts to 1.0. precision, recall, and f1 score the shared
        words. The other columns, the index, and the row order are kept.

    Raises:
        ValueError: pairs lacks the reference or candidate column, either
            column contains a missing or non-string value, or pairs already
            has a score column.
    """
    missing = [column for column in _TEXT_COLUMNS if column not in pairs]
    if missing:
        raise ValueError(f"pairs is missing columns {missing}")
    with_nulls = _text_columns_with_missing_values(pairs)
    if with_nulls:
        raise ValueError(f"pairs has missing values in columns {with_nulls}")
    present = [column for column in _SCORE_COLUMNS if column in pairs]
    if present:
        raise ValueError(f"pairs already has score columns {present}")
    not_text = _non_text_columns(pairs)
    if not_text:
        raise ValueError(f"pairs has non-text values in columns {not_text}")

    scores = [
        text_metrics.evaluate_ocr_text(reference, candidate)
        for reference, candidate in zip(
            pairs["reference"], pairs["candidate"], strict=True
        )
    ]
    values = {
        "ned": [score.ned for score in scores],
        "precision": [score.bow.precision for score in scores],
        "recall": [score.bow.recall for score in scores],
        "f1": [score.bow.f1 for score in scores],
    }
    return pairs.assign(
        **{
            column: pd.Series(column_values, index=pairs.index, dtype="float64")
            for column, column_values in values.items()
        }
    )


def summarise_text_scores(scores: pd.DataFrame, by: str) -> pd.DataFrame:
    """Summarises text scores per group and over all rows.

    Args:
        scores: One row per scored pair, with the column by, the text
            columns reference and candidate, and the score columns ned,
            precision, recall, and f1.
        by: The column whose values form the groups.

    Returns:
        A frame indexed by the groups in sorted order, followed by a row
        named overall that covers every row. The columns are:

        - count: the number of pairs.
        - empty_candidates: the number of pairs whose candidate is blank
          and whose reference is not.
        - For each score, its mean, its median, and a percentile, in that
          order: ned_mean, ned_median, ned_p90, precision_mean,
          precision_median, precision_p10, recall_mean, recall_median,
          recall_p10, f1_mean, f1_median, and f1_p10. The percentile is
          the 90th for ned, where higher is worse, and the 10th for the
          others, where lower is worse.

    Raises:
        ValueError: by names a text or score column, scores lacks the by
            column, a text column, or a score column, has no rows, has a
            missing value in its by column, its by column contains
            "overall", the label of the overall row, or a text column
            contains a missing or non-string value.
    """
    if by in (*_TEXT_COLUMNS, *_SCORE_COLUMNS):
        raise ValueError(f"{by!r}: is a text or score column")
    missing = [
        column
        for column in (by, *_TEXT_COLUMNS, *_SCORE_COLUMNS)
        if column not in scores
    ]
    if missing:
        raise ValueError(f"scores is missing columns {missing}")
    if scores.empty:
        raise ValueError("scores is empty")
    if scores[by].isna().any():
        raise ValueError(f"{by!r}: column has missing values")
    if (scores[by] == "overall").any():
        raise ValueError(f"{by!r}: column contains 'overall'")
    with_nulls = _text_columns_with_missing_values(scores)
    if with_nulls:
        raise ValueError(f"scores has missing values in columns {with_nulls}")
    not_text = _non_text_columns(scores)
    if not_text:
        raise ValueError(f"scores has non-text values in columns {not_text}")

    candidate_is_blank = scores["candidate"].str.strip().eq("")
    reference_has_text = scores["reference"].str.strip().ne("")
    pairs = scores[list(_SCORE_COLUMNS)].assign(
        empty_candidate=candidate_is_blank & reference_has_text
    )

    aggregations: dict[str, tuple[str, _Statistic]] = {
        "count": ("empty_candidate", "size"),
        "empty_candidates": ("empty_candidate", "sum"),
    }
    for column in _SCORE_COLUMNS:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_median"] = (column, "median")
        # A higher ned is worse and lower word scores are worse.
        if column == "ned":
            aggregations["ned_p90"] = ("ned", lambda s: s.quantile(0.9))
        else:
            aggregations[f"{column}_p10"] = (column, lambda s: s.quantile(0.1))

    by_group = pairs.groupby(scores[by]).agg(**aggregations)
    overall = (
        pairs.reset_index(drop=True)
        .assign(group="overall")
        .groupby("group")
        .agg(**aggregations)
    )
    return pd.concat([by_group, overall]).rename_axis(by)
