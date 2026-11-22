"""Tests for docrag.datasets.ohr_bench_convert."""

import pandas as pd
import pytest

from docrag.datasets import ohr_bench_convert

_GROUND_TRUTH_COLUMNS = ["domain", "doc_name", "page_idx", "gt_text"]


def _ground_truth() -> pd.DataFrame:
    """Returns ground-truth pages with an extra column and a custom index."""
    return pd.DataFrame(
        {
            "domain": ["law", "news"],
            "doc_name": ["law/a", "news/b"],
            "page_idx": [0, 0],
            "gt_text": ["one", "two"],
            "formatting_noise_mild": ["x", "y"],
        },
        index=[7, 5],
    )


def test_to_reference_pages_rows() -> None:
    reference = ohr_bench_convert.to_reference_pages(_ground_truth())

    expected = pd.DataFrame(
        {
            "domain": ["law", "news"],
            "doc_name": ["law/a", "news/b"],
            "page_idx": [0, 0],
            "text": ["one", "two"],
        },
        index=[7, 5],
    )
    pd.testing.assert_frame_equal(reference, expected)


@pytest.mark.parametrize("column", _GROUND_TRUTH_COLUMNS)
def test_to_reference_pages_missing_column(column: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        ohr_bench_convert.to_reference_pages(
            _ground_truth().drop(columns=[column])
        )

    assert str(excinfo.value) == f"ground_truth is missing columns ['{column}']"


def test_to_reference_pages_input_unchanged() -> None:
    ground_truth = _ground_truth()

    ohr_bench_convert.to_reference_pages(ground_truth)

    pd.testing.assert_frame_equal(ground_truth, _ground_truth())
