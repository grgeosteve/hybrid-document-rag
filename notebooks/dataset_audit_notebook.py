import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import marimo as mo
    import pandas as pd

    from docrag.datasets import ohr_bench
    from docrag.datasets import ohr_bench_audit
    from docrag.datasets import ohr_bench_convert
    from docrag.datasets import parsed_pages
    from docrag.evaluation import text_comparison
    from docrag.ingestion import factory

    return (
        factory,
        mo,
        ohr_bench,
        ohr_bench_audit,
        ohr_bench_convert,
        parsed_pages,
        pathlib,
        pd,
        text_comparison,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # OHR-Bench dataset audit

    Measurements over the OHR-Bench pages, question-answer pairs, and
    `pypdfium2` text-layer extraction, by domain.
    """)
    return


@app.cell
def _(ohr_bench, ohr_bench_audit, pathlib):
    DATA_ROOT = (
        pathlib.Path(__file__).resolve().parents[1]
        / "data"
        / ohr_bench.DATASET.name
    )

    audit = ohr_bench_audit.OHRBenchAudit.from_files(
        parquet_path=DATA_ROOT / ohr_bench.GROUND_TRUTH_FILE,
        qa_path=DATA_ROOT / ohr_bench.QA_FILE,
    )
    return DATA_ROOT, audit


@app.cell
def _(mo):
    mo.md(r"""
    ## Data integrity and language
    """)
    return


@app.cell(expand_output=True)
def _(audit, mo):
    missing_values = audit.missing_values()
    empty_pages = audit.empty_pages()
    page_languages = audit.page_languages()
    question_languages = audit.question_languages()

    page_languages_sorted = page_languages.reset_index().sort_values(
        ["domain", "texts"], ascending=[True, False]
    )
    question_languages_sorted = question_languages.reset_index().sort_values(
        ["domain", "texts"], ascending=[True, False]
    )

    mo.vstack(
        [
            mo.md("### Missing and empty values"),
            mo.ui.table(
                missing_values.reset_index(),
                pagination=False,
                max_columns=None,
                selection=None,
            ),
            mo.md("### Empty ground-truth pages by domain"),
            mo.ui.table(
                empty_pages.reset_index(),
                pagination=False,
                max_columns=None,
                selection=None,
            ),
            mo.accordion(
                {
                    "Full page language counts": mo.ui.table(
                        page_languages_sorted,
                        pagination=True,
                        max_columns=None,
                        selection=None,
                    ),
                    "Full question language counts": mo.ui.table(
                        question_languages_sorted,
                        pagination=True,
                        max_columns=None,
                        selection=None,
                    ),
                }
            ),
        ]
    )
    return page_languages, question_languages


@app.cell
def _(page_languages):
    def _():
        counts = page_languages["texts"].unstack(fill_value=0)
        top = counts.sum().nlargest(4).index
        grouped = counts[top].assign(OTHER=counts.drop(columns=top).sum(axis=1))
        shares = grouped.div(grouped.sum(axis=1), axis=0)
        ax = shares.plot.barh(stacked=True, xlim=(0, 1), figsize=(10, 4))
        ax.set_title("Page language share by domain")
        ax.set_xlabel("Share of non-empty pages")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig = ax.get_figure()
        fig.tight_layout()
        return fig

    _()
    return


@app.cell
def _(question_languages):
    def _():
        counts = question_languages["texts"].unstack(fill_value=0)
        top = counts.sum().nlargest(4).index
        grouped = counts[top].assign(OTHER=counts.drop(columns=top).sum(axis=1))
        shares = grouped.div(grouped.sum(axis=1), axis=0)
        ax = shares.plot.barh(stacked=True, xlim=(0, 1), figsize=(10, 4))
        ax.set_title("Question language share by domain")
        ax.set_xlabel("Share of questions")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig = ax.get_figure()
        fig.tight_layout()
        return fig

    _()
    return


@app.cell(expand_output=True)
def _(audit):
    def _():
        page_detections = audit.page_detections()
        english_pages = page_detections[
            page_detections["language"] == "ENGLISH"
        ]
        axes = english_pages.hist(
            column="confidence",
            by="domain",
            bins=50,
            range=(0, 1),
            figsize=(14, 16),
            log=True,
            layout=(4, 2),
        )
        fig = axes.flat[0].get_figure()
        fig.suptitle("English page confidence by domain (log y)")
        fig.tight_layout()
        return fig

    _()
    return


@app.cell(expand_output=True)
def _(audit):
    def _():
        question_detections = audit.question_detections()
        english_questions = question_detections[
            question_detections["language"] == "ENGLISH"
        ]
        axes = english_questions.hist(
            column="confidence",
            by="domain",
            bins=50,
            range=(0, 1),
            figsize=(14, 16),
            log=True,
            layout=(4, 2),
        )
        fig = axes.flat[0].get_figure()
        fig.suptitle("English question confidence by domain (log y)")
        fig.tight_layout()
        return fig

    _()
    return


@app.cell
def _(audit, mo, pd):
    language_threshold = 0.80
    thresholds = (0.50, language_threshold, 0.90)
    english_at_threshold = audit.english_share(threshold=language_threshold)
    english_by_threshold = (
        pd.concat(
            {f"{t:.2f}": audit.english_share(threshold=t) for t in thresholds},
            axis=1,
        )
        .swaplevel(axis=1)
        .sort_index(axis=1)
        .round(3)
    )
    english_by_threshold.columns = [
        f"{kind} @ {t}" for kind, t in english_by_threshold.columns
    ]
    mo.vstack(
        [
            mo.md("### English share across confidence thresholds"),
            mo.ui.table(
                english_by_threshold.reset_index(),
                pagination=False,
                max_columns=None,
                selection=None,
            ),
            mo.md(
                "No column has null values. 302 of 8,561 pages have blank "
                "ground-truth text. English shares vary by less than 1 "
                "percentage point between `0.50` and `0.90`. Threshold "
                f"`{language_threshold:.2f}` is used in the domain summary."
            ),
        ]
    )
    return english_at_threshold, language_threshold


@app.cell
def _(
    DATA_ROOT,
    audit,
    factory,
    mo,
    ohr_bench_convert,
    parsed_pages,
    text_comparison,
):
    reference_pages = ohr_bench_convert.to_reference_pages(audit.pages)
    backend = factory.ParserBackend.PYPDFIUM2
    parsed_path = (
        DATA_ROOT / parsed_pages.PARSED_PAGES_DIR / f"{backend.value}.parquet"
    )
    parsed = parsed_pages.load_parsed_pages(parsed_path)
    assert set(parsed["doc_name"].unique()) == set(
        reference_pages["doc_name"].unique()
    )

    parsed_reference_pair = parsed_pages.pair_with_reference(
        parsed=parsed, reference=reference_pages
    )
    pair_scores = text_comparison.score_text_pairs(pairs=parsed_reference_pair)
    parser_summary = text_comparison.summarise_text_scores(
        scores=pair_scores, by="domain"
    )
    mo.vstack(
        [
            mo.md(f"## Parser baseline (`{backend.value}`)"),
            mo.ui.table(
                parser_summary.round(3).reset_index(),
                pagination=False,
                max_columns=None,
                selection=None,
            ),
        ]
    )
    return parsed, parser_summary


@app.cell(expand_output=True)
def _(audit, mo):
    answer_lengths_by_form = audit.answer_lengths_by_form()
    domain_answer_lengths_by_form = audit.answer_lengths_by_form(by_domain=True)
    answer_lengths_by_evidence = audit.answer_lengths_by_evidence()
    answer_lengths_by_evidence_domain = audit.answer_lengths_by_evidence(
        by_domain=True
    )
    mo.vstack(
        [
            mo.md("## Answer lengths"),
            mo.md("### Word counts by answer form"),
            mo.ui.table(
                answer_lengths_by_form.reset_index(),
                pagination=False,
                max_columns=None,
                selection=None,
            ),
            mo.md("### Word counts by evidence source"),
            mo.ui.table(
                answer_lengths_by_evidence.reset_index(),
                pagination=False,
                max_columns=None,
                selection=None,
            ),
            mo.accordion(
                {
                    "By domain and answer form": mo.ui.table(
                        domain_answer_lengths_by_form.reset_index(),
                        pagination=False,
                        max_columns=None,
                        selection=None,
                    ),
                    "By domain and evidence source": mo.ui.table(
                        answer_lengths_by_evidence_domain.reset_index(),
                        pagination=False,
                        max_columns=None,
                        selection=None,
                    ),
                }
            ),
        ]
    )
    return (answer_lengths_by_evidence_domain,)


@app.cell
def _(audit, mo):
    doc_length = audit.document_lengths()
    mo.vstack(
        [
            mo.md("## Document lengths"),
            mo.ui.table(
                doc_length.round(1).reset_index(),
                pagination=False,
                max_columns=None,
                selection=None,
            ),
        ]
    )
    return (doc_length,)


@app.cell
def _(audit, mo, parsed):
    chart_all = (
        audit.qa[audit.qa["evidence_source"] == "chart"]
        .sort_values("ID")
        .assign(
            evidence_page_no=lambda df: df["evidence_page_no"].astype("int64")
        )
        .merge(
            audit.pages[["doc_name", "page_idx", "gt_text"]],
            left_on=["doc_name", "evidence_page_no"],
            right_on=["doc_name", "page_idx"],
            how="left",
        )
        .merge(
            parsed[["doc_name", "page_idx", "text"]].rename(
                columns={"text": "parsed_text"}
            ),
            on=["doc_name", "page_idx"],
            how="left",
        )
    )
    chart_summary = (
        chart_all.assign(
            has_chart_tag=chart_all["gt_text"].str.contains(
                "<chart>", regex=False, na=False
            ),
            has_latex_tabular=chart_all["gt_text"].str.contains(
                r"\begin{tabular}", regex=False, na=False
            ),
        )
        .groupby("domain")
        .agg(
            chart_qs=("ID", "count"),
            docs=("doc_name", "nunique"),
            with_chart_tag=("has_chart_tag", "sum"),
            with_latex_tabular=("has_latex_tabular", "sum"),
        )
    )
    chart_examples = chart_all.groupby("domain").head(2)[
        [
            "ID",
            "domain",
            "doc_name",
            "evidence_page_no",
            "questions",
            "answers",
            "evidence_context",
            "gt_text",
            "parsed_text",
        ]
    ]
    mo.vstack(
        [
            mo.md("## Charts in the ground truth"),
            mo.md("### Chart syntax across all chart questions"),
            mo.ui.table(
                chart_summary.reset_index(),
                pagination=False,
                max_columns=None,
                selection=None,
            ),
            mo.md(
                "### Sample chart questions (2 per domain): "
                "ground truth vs parsed"
            ),
            mo.ui.table(
                chart_examples,
                pagination=False,
                max_columns=None,
                selection=None,
                wrapped_columns=[
                    "questions",
                    "evidence_context",
                    "gt_text",
                    "parsed_text",
                ],
            ),
        ]
    )
    return


@app.cell
def _(
    answer_lengths_by_evidence_domain,
    doc_length,
    english_at_threshold,
    language_threshold,
    mo,
    parser_summary,
    pd,
):
    domains = [domain for domain in doc_length.index if domain != "overall"]
    evidence_counts = answer_lengths_by_evidence_domain["count"].unstack(
        fill_value=0
    )
    text_q = evidence_counts.get("text", pd.Series(0, index=domains))
    table_q = evidence_counts.get("table", pd.Series(0, index=domains))

    domain_summary = pd.DataFrame(
        {
            "documents": doc_length.loc[domains, "documents"],
            "pages_median": doc_length.loc[domains, "pages_median"],
            "pages_p90": doc_length.loc[domains, "pages_p90"].round(1),
            f"eng_pages@{language_threshold:.2f}": english_at_threshold.loc[
                domains, "pages"
            ].round(3),
            f"eng_questions@{language_threshold:.2f}": english_at_threshold.loc[
                domains, "questions"
            ].round(3),
            "empty_cand_pct": (
                100.0
                * parser_summary.loc[domains, "empty_candidates"]
                / parser_summary.loc[domains, "count"]
            ).round(1),
            "f1_median": parser_summary.loc[domains, "f1_median"].round(3),
            "f1_p10": parser_summary.loc[domains, "f1_p10"].round(3),
            "ned_median": parser_summary.loc[domains, "ned_median"].round(3),
            "text_qs": text_q.reindex(domains, fill_value=0),
            "table_qs": table_q.reindex(domains, fill_value=0),
        },
        index=pd.Index(domains, name="domain"),
    )
    domain_summary["text_and_table_qs"] = (
        domain_summary["text_qs"] + domain_summary["table_qs"]
    )
    domain_summary = domain_summary.sort_values("empty_cand_pct")

    mo.vstack(
        [
            mo.md("## Domain summary"),
            mo.ui.table(
                domain_summary.reset_index(),
                pagination=False,
                max_columns=None,
                selection=None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Data integrity:** 8,561 pages (302 blank `gt_text`) and 8,498 QA pairs,
      with 0 null values across all columns.
    - **Language:** At confidence threshold `0.80`, `finance` (87.0% pages, 98.3%
      questions), `law` (88.9% / 97.1%), `administration` (88.3% / 97.2%),
      `manual` (86.4% / 93.2%), and `academic` (84.1% / 87.8%) are predominantly
      English. `news` (75.1% / 44.5%) and `textbook` (68.0% / 71.1%) contain
      large Chinese shares. Domain shares change by less than 1 percentage point
      between `0.50` and `0.90`.
    - **Parser baseline (`pypdfium2`):** All 8,561 pages pair with 0 page
      mismatches. Empty-candidate rates range from `finance` (0.9%) and `law`
      (3.5%) to `academic` (4.8%), `news` (10.7%), `manual` (11.2%),
      `administration` (19.2%), and `textbook` (62.7%). Only `law` (0.672),
      `finance` (0.553), and `academic` (0.213) remain above `0.000` at p10 F1.
    - **Answer lengths:** `Numeric` (2,980 questions, median 1, p90 2 words) and
      `Yes/No` (596 questions, median 1, p90 1 word) make up 42.1% of questions.
      `String` (4,190 questions, median 3, p90 13 words) and `List` (583 questions, median 4,
       p90 12 words) run up to 20 words.
    - **Document lengths in pages:** `finance` (median 8, p90 108.8, max 382 pages),
      `manual` (median 11, p90 43.2), `law` (median 6, p90 29.6), `academic`
      (median 11, p90 20.0), and `administration` (median 5, p90 20.0) are
      multi-page. `news` (median 1, p90 2) and `textbook` (median 1, p90 1) are
      mostly single-page.
    - **Charts:** All 765 `chart` questions across 5 domains store chart values
      in `gt_text` inside `<chart>` tags as LaTeX `\begin{tabular}` tables.
      Plain text extraction (`pypdfium2`) loses chart structure, drops rasterised
      charts, or emits garbled bytes where PDF font mappings are missing.
      This is now a documented limitation which is expected to affect retrieval
      performance, and will be measured and addressed at a later stage.
    """)
    return


if __name__ == "__main__":
    app.run()
