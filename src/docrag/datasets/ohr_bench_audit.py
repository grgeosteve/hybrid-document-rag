"""Per-domain statistics of the OHR-Bench dataset for its audit."""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
import functools
import json
import pathlib

import lingua
import pandas as pd
import pyarrow.parquet as pq
import tqdm

_PAGE_COLUMNS = ("domain", "doc_name", "page_idx", "gt_text")
_QA_COLUMNS = (
    "doc_name",
    "ID",
    "questions",
    "answers",
    "doc_type",
    "answer_form",
    "evidence_source",
    "evidence_context",
    "evidence_page_no",
)


@functools.cache
def _detector() -> lingua.LanguageDetector:
    """Returns a detector for all languages, built on the first call."""
    return (
        lingua.LanguageDetectorBuilder.from_all_languages()
        .with_low_accuracy_mode()
        .build()
    )


def detect_languages(
    texts: Sequence[str], *, batch_size: int = 256
) -> list[tuple[str, float]]:
    """Detects the most likely language of each text.

    Shows a progress bar on standard error.

    Args:
        texts: The texts to classify.
        batch_size: The number of texts classified together.

    Returns:
        One (language, confidence) pair per text, in the order of texts. The
        language is a name such as "ENGLISH" and the confidence is between 0
        and 1. A blank text gives ("UNKNOWN", 0.0). A confidence of 0 means
        that no language was detected.

    Raises:
        ValueError: batch_size is less than 1.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    results: list[tuple[str, float]] = [("UNKNOWN", 0.0)] * len(texts)
    positions = [i for i, text in enumerate(texts) if not _is_blank(text)]
    with tqdm.tqdm(total=len(texts), unit="text") as progress:
        progress.update(len(texts) - len(positions))
        for start in range(0, len(positions), batch_size):
            batch = positions[start : start + batch_size]
            values = _detector().compute_language_confidence_values_in_parallel(
                [texts[i] for i in batch]
            )
            for i, confidences in zip(batch, values, strict=True):
                top = confidences[0]
                results[i] = (top.language.name, top.value)
            progress.update(len(batch))
    return results


def _detect_missing_columns(
    existing: Iterable[str], required: tuple[str, ...]
) -> list[str]:
    """Returns the required column names that are not in existing."""
    existing_cols = list(existing)
    required_cols = required

    missing = list(set(required_cols).difference(set(existing_cols)))
    return missing


def _is_blank(value: object) -> bool:
    """Returns whether a value is a string that is blank after stripping."""
    return isinstance(value, str) and not value.strip()


class OHRBenchAudit:
    """Statistics over the OHR-Bench pages and question-answer pairs.

    Language detections are computed on first use and kept for the life of
    the instance.
    """

    def __init__(
        self,
        pages: pd.DataFrame,
        qa: pd.DataFrame,
        *,
        detect: Callable[
            [Sequence[str]], list[tuple[str, float]]
        ] = detect_languages,
    ) -> None:
        """Checks and stores the two sources.

        Only the required columns are kept. The QA column doc_type is
        renamed to domain.

        Args:
            pages: One row per page, with the columns domain, doc_name,
                page_idx, and gt_text.
            qa: One row per question-answer pair, with the nine fields of
                the QA file.
            detect: The function that returns the language and confidence
                of each text in a sequence.

        Raises:
            ValueError: A required column is missing from pages or qa. The
                message names the source and the columns.
        """
        OHRBenchAudit._require_columns(
            name="pages", existing=pages.columns, columns=_PAGE_COLUMNS
        )
        OHRBenchAudit._require_columns(
            name="qa", existing=qa.columns, columns=_QA_COLUMNS
        )
        self.pages = pages[list(_PAGE_COLUMNS)]
        self.qa = qa[list(_QA_COLUMNS)]
        self.qa = self.qa.rename(columns={"doc_type": "domain"})
        self._detect = detect

    @classmethod
    def from_files(
        cls,
        parquet_path: pathlib.Path,
        qa_path: pathlib.Path,
        *,
        detect: Callable[
            [Sequence[str]], list[tuple[str, float]]
        ] = detect_languages,
    ) -> "OHRBenchAudit":
        """Creates an audit from the ground-truth parquet and the QA file.

        QA values keep the types they have in the JSON file.

        Args:
            parquet_path: The ground-truth parquet file.
            qa_path: The QA JSON file.
            detect: The function that returns the language and confidence
                of each text in a sequence.

        Returns:
            An audit over the two files.

        Raises:
            FileNotFoundError: A file does not exist.
            OSError: A file cannot be read.
            json.JSONDecodeError: The QA file is not JSON.
            ValueError: The parquet file is not valid parquet, the QA file's
                top level is not a list, or a required column is missing
                from either file.
        """
        pages_columns = pq.read_schema(parquet_path).names
        OHRBenchAudit._require_columns(
            name="pages", existing=pages_columns, columns=_PAGE_COLUMNS
        )
        pages = pd.read_parquet(parquet_path, columns=list(_PAGE_COLUMNS))

        with qa_path.open(encoding="utf-8") as qa_file:
            records = json.load(qa_file)
        if not isinstance(records, list):
            raise ValueError(f"{qa_path}: top level is not a list")
        qa = pd.DataFrame.from_records(records)

        return OHRBenchAudit(pages, qa, detect=detect)

    @classmethod
    def _require_columns(
        cls, name: str, existing: Iterable[str], columns: tuple[str, ...]
    ) -> None:
        """Raises an error if a source lacks any of the given columns.

        Args:
            name: The source's name, used in the message.
            existing: The column names the source has.
            columns: The column names the source must have.

        Raises:
            ValueError: One or more columns are missing.
        """
        missing_cols = _detect_missing_columns(existing, columns)
        if len(missing_cols) > 0:
            raise ValueError(
                f"{name} data is missing the following "
                f"required columns: {missing_cols}"
            )

    @functools.cached_property
    def _page_detections(self) -> pd.DataFrame:
        """Domain, language, and confidence of each page.

        A blank page has the language UNKNOWN and the confidence 0.
        """
        texts = self.pages["gt_text"]
        frame = self.pages
        detections = pd.DataFrame(
            self._detect(texts.to_list()),
            index=texts.index,
            columns=["language", "confidence"],
        )
        result = frame[["domain"]].join(detections)

        return result

    @functools.cached_property
    def _question_detections(self) -> pd.DataFrame:
        """Domain, language, and confidence of each question."""
        texts = self.qa["questions"]
        frame = self.qa
        detections = pd.DataFrame(
            self._detect(texts.to_list()),
            index=texts.index,
            columns=["language", "confidence"],
        )
        result = frame[["domain"]].join(detections)

        return result

    def missing_values(self) -> pd.DataFrame:
        """Counts null and empty values in each column of both sources.

        Returns:
            A frame indexed by (source, column) with the columns null and
            empty. A value is empty if it is a string that is blank after
            stripping whitespace.
        """
        counts = {
            name: pd.DataFrame(
                {
                    "null": frame.isna().sum(),
                    "empty": frame.map(_is_blank).sum(),
                }
            )
            for name, frame in (("pages", self.pages), ("qa", self.qa))
        }
        return pd.concat(counts, names=["source", "column"])

    def empty_pages(self) -> pd.DataFrame:
        """Counts pages with empty ground-truth text per domain.

        A page is empty if its ground-truth text is blank after stripping
        whitespace.

        Returns:
            A frame indexed by domain with the columns pages and empty.
        """
        page_stats = (
            self.pages.assign(
                empty=self.pages["gt_text"].map(lambda t: len(t.strip())) == 0
            )
            .groupby("domain")
            .agg(pages=("page_idx", "count"), empty=("empty", "sum"))
        )

        return page_stats

    def page_detections(self) -> pd.DataFrame:
        """Returns the detected language and confidence of non-empty pages.

        A page is empty if its ground-truth text is blank after stripping
        whitespace.

        Returns:
            A frame indexed by the page row index, with the columns domain,
            language, and confidence. Mutating the returned frame does not
            affect later calls.
        """
        return self._page_detections[
            ~self.pages["gt_text"].map(_is_blank)
        ].copy()

    def question_detections(self) -> pd.DataFrame:
        """Returns the detected language and confidence of each question.

        Returns:
            A frame indexed by the question row index, with the columns domain,
            language, and confidence. Mutating the returned frame does not
            affect later calls.
        """
        return self._question_detections.copy()

    def page_languages(self) -> pd.DataFrame:
        """Summarises the detected languages of non-empty pages.

        Returns:
            A frame indexed by (domain, language) with the columns texts,
            confidence_median, confidence_p10, and confidence_min.
        """
        language_df = self._page_detections[
            ~self.pages["gt_text"].map(_is_blank)
        ]
        result = language_df.groupby(["domain", "language"]).agg(
            texts=("confidence", "count"),
            confidence_median=("confidence", "median"),
            confidence_p10=("confidence", lambda s: s.quantile(0.1)),
            confidence_min=("confidence", "min"),
        )

        return result

    def question_languages(self) -> pd.DataFrame:
        """Summarises the detected languages of the questions.

        Returns:
            A frame indexed by (domain, language) with the columns texts,
            confidence_median, confidence_p10, and confidence_min.
        """
        language_df = self._question_detections
        result = language_df.groupby(["domain", "language"]).agg(
            texts=("confidence", "count"),
            confidence_median=("confidence", "median"),
            confidence_p10=("confidence", lambda s: s.quantile(0.1)),
            confidence_min=("confidence", "min"),
        )

        return result

    def english_share(self, threshold: float) -> pd.DataFrame:
        """Computes the share of English texts per domain.

        A text counts as English if its detected language is ENGLISH with a
        confidence of at least threshold. Empty pages are not counted.

        Args:
            threshold: The minimum confidence, between 0 and 1.

        Returns:
            A frame indexed by the domains that have non-empty pages, with
            the columns pages and questions. Each column is the share of
            that domain's texts that count as English.

        Raises:
            ValueError: threshold is outside (0, 1].
        """
        if threshold <= 0 or threshold > 1:
            raise ValueError(
                f"threshold value ({threshold}) is outside (0, 1]."
            )

        plang_df = self._page_detections[~self.pages["gt_text"].map(_is_blank)]
        qlang_df = self._question_detections

        plang_frame = plang_df.assign(
            english=(
                plang_df["confidence"].ge(threshold)
                & (plang_df["language"] == "ENGLISH")
            )
        )
        qlang_frame = qlang_df.assign(
            english=(
                qlang_df["confidence"].ge(threshold)
                & (qlang_df["language"] == "ENGLISH")
            )
        )

        plang_english_share = plang_frame.groupby("domain").agg(
            pages=("english", "mean")
        )
        qlang_english_share = qlang_frame.groupby("domain").agg(
            questions=("english", "mean")
        )
        results = plang_english_share.join(qlang_english_share)

        return results

    def answer_lengths_by_form(
        self, *, by_domain: bool = False
    ) -> pd.DataFrame:
        """Summarises answer word counts per answer form.

        An answer's word count is its number of whitespace-separated words.

        Args:
            by_domain: Whether to group by domain before answer form.

        Returns:
            A frame indexed by the groups with the columns count, median,
            p90, and max.
        """
        keys = ["domain", "answer_form"] if by_domain else ["answer_form"]

        text_series = self.qa["answers"]
        answer_level_stats = (
            self.qa.assign(
                word_count=text_series.map(lambda t: len(t.strip().split()))
            )
            .groupby(keys)
            .agg(
                count=("word_count", "count"),
                median=("word_count", "median"),
                p90=("word_count", lambda s: s.quantile(0.9)),
                max=("word_count", "max"),
            )
        )

        results = answer_level_stats

        return results

    def answer_lengths_by_evidence(
        self, *, by_domain: bool = False
    ) -> pd.DataFrame:
        """Summarises answer word counts per evidence source.

        An answer's word count is its number of whitespace-separated words.

        Args:
            by_domain: Whether to group by domain before evidence source.

        Returns:
            A frame indexed by the groups with the columns count, median,
            p90, and max.
        """
        keys = (
            ["domain", "evidence_source"] if by_domain else ["evidence_source"]
        )

        text_series = self.qa["answers"]
        answer_level_stats = (
            self.qa.assign(
                word_count=text_series.map(lambda t: len(t.strip().split()))
            )
            .groupby(keys)
            .agg(
                count=("word_count", "count"),
                median=("word_count", "median"),
                p90=("word_count", lambda s: s.quantile(0.9)),
                max=("word_count", "max"),
            )
        )

        results = answer_level_stats

        return results

    def document_lengths(self) -> pd.DataFrame:
        """Summarises document length per domain and over all domains.

        A document's length is its number of pages, and the number of
        characters and of whitespace-separated words in its ground-truth
        text. Empty pages count as pages.

        Returns:
            A frame indexed by domain, followed by a row named overall that
            covers every document, with the columns documents, pages_total,
            pages_mean, pages_median, pages_p90, pages_max,
            characters_median, characters_p90, characters_max,
            words_median, words_p90, and words_max.
        """
        text_series = self.pages["gt_text"]
        doc_level_stats = (
            self.pages.assign(
                char_count=text_series.str.len(),
                word_count=text_series.map(lambda t: len(t.strip().split())),
            )
            .groupby(["domain", "doc_name"])
            .agg(
                page_count=("page_idx", "count"),
                char_count=("char_count", "sum"),
                word_count=("word_count", "sum"),
            )
        )

        agg_spec = dict(
            documents=("doc_name", "count"),
            pages_total=("page_count", "sum"),
            pages_mean=("page_count", "mean"),
            pages_median=("page_count", "median"),
            pages_p90=("page_count", lambda s: s.quantile(0.9)),
            pages_max=("page_count", "max"),
            characters_median=("char_count", "median"),
            characters_p90=("char_count", lambda s: s.quantile(0.9)),
            characters_max=("char_count", "max"),
            words_median=("word_count", "median"),
            words_p90=("word_count", lambda s: s.quantile(0.9)),
            words_max=("word_count", "max"),
        )

        by_domain = (
            doc_level_stats.reset_index().groupby("domain").agg(**agg_spec)
        )
        overall = (
            doc_level_stats.reset_index()
            .assign(domain="overall")
            .groupby("domain")
            .agg(**agg_spec)
        )

        results = pd.concat([by_domain, overall])
        return results
