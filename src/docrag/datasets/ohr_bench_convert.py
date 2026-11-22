"""Conversion of OHR-Bench data to the forms shared across datasets."""

import pandas as pd

_GROUND_TRUTH_COLUMNS = ("domain", "doc_name", "page_idx", "gt_text")


def to_reference_pages(ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Converts OHR-Bench ground-truth pages to reference pages.

    ground_truth is not changed.

    Args:
        ground_truth: One row per page, with the columns domain, doc_name,
            page_idx, and gt_text. Other columns are ignored.

    Returns:
        One row per page, with the columns domain, doc_name, page_idx, and
        text, in the order and with the index of ground_truth.

    Raises:
        ValueError: ground_truth lacks a required column.
    """
    missing = [
        column for column in _GROUND_TRUTH_COLUMNS if column not in ground_truth
    ]
    if missing:
        raise ValueError(f"ground_truth is missing columns {missing}")

    return ground_truth[list(_GROUND_TRUTH_COLUMNS)].rename(
        columns={"gt_text": "text"}
    )
