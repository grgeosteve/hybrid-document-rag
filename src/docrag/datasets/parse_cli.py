"""Command line that parses a named dataset's PDFs into a parquet file."""

import argparse
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import logging
import pathlib
import types

from docrag.datasets import dataset
from docrag.datasets import ohr_bench
from docrag.datasets import parsed_pages
from docrag.file_ops import paths
from docrag.ingestion import factory
from docrag.ingestion import protocol

_DATASETS = types.MappingProxyType({ohr_bench.DATASET.name: ohr_bench.DATASET})

_logger = logging.getLogger(__name__)


def parse_dataset(
    data_root: pathlib.Path,
    dataset_to_parse: dataset.Dataset,
    backend: factory.ParserBackend,
    *,
    create_parser: Callable[
        [factory.ParserBackend], protocol.PdfParser
    ] = factory.create_pdf_parser,
    overwrite: bool = False,
) -> pathlib.Path:
    """Parses every PDF of a dataset and writes the pages to parquet.

    Reads all PDF files under the dataset's extracted PDF directory in
    sorted path order and writes the result to
    ``<data_root>/<dataset>/parsed_pages/<backend>.parquet``. Validates
    paths and checks whether the output file already exists before creating
    the parser or starting to parse.

    Args:
        data_root: The directory that holds every dataset.
        dataset_to_parse: The dataset whose PDFs are parsed.
        backend: The parser backend, used to build the parser and name the
            output parquet file.
        create_parser: The function that returns a PDF parser for a backend.
        overwrite: Whether to replace an existing output parquet file.

    Returns:
        The path of the written parquet file.

    Raises:
        FileNotFoundError: The dataset's PDF directory does not exist, or a
            PDF disappears before it is parsed.
        FileExistsError: The dataset directory, PDF directory, or output
            directory exists and is not a directory, the output path or a
            matched PDF path exists and is not a regular file, or the
            output file already exists and overwrite is false.
        OSError: The dataset directory, PDF directory, output directory,
            output path, or a matched PDF path is a symbolic link, or a
            file operation fails.
        ValueError: The dataset's PDF directory contains no PDF files.
        protocol.UnreadablePdfError: A PDF cannot be opened.
    """
    dataset_dir = data_root / dataset_to_parse.name
    pdf_dir = dataset_dir / dataset_to_parse.extract_dir_name
    output_dir = dataset_dir / parsed_pages.PARSED_PAGES_DIR
    output_path = output_dir / f"{backend.value}.parquet"

    paths.reject_symlink(dataset_dir)
    paths.reject_non_directory(dataset_dir)
    paths.reject_symlink(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"{pdf_dir}: does not exist")
    paths.reject_non_directory(pdf_dir)

    paths.reject_symlink(output_dir)
    paths.reject_non_directory(output_dir)
    paths.reject_symlink(output_path)
    paths.reject_non_file(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path}: already exists")

    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    for pdf_path in pdf_paths:
        paths.reject_symlink(pdf_path)
        paths.reject_non_file(pdf_path)
    if not pdf_paths:
        raise ValueError(f"{pdf_dir}: contains no PDF files")

    doc_names = [
        pdf_path.relative_to(pdf_dir).with_suffix("").as_posix()
        for pdf_path in pdf_paths
    ]
    parser = create_parser(backend)
    pages = parsed_pages.parse_documents(pdf_dir, doc_names, parser=parser)
    output_dir.mkdir(exist_ok=True)
    parsed_pages.save_parsed_pages(pages, output_path, overwrite=overwrite)
    _logger.info(
        "Parsed %d documents (%d pages) to %s",
        len(doc_names),
        len(pages),
        output_path,
    )
    return output_path


def main(
    argv: Sequence[str] | None = None,
    *,
    data_root: pathlib.Path,
    datasets: Mapping[str, dataset.Dataset] = _DATASETS,
    create_parser: Callable[
        [factory.ParserBackend], protocol.PdfParser
    ] = factory.create_pdf_parser,
) -> int:
    """Parses the dataset named on the command line.

    An expected failure, such as a missing PDF directory, an existing
    output file without overwrite, or an unreadable PDF, is logged as one
    line and gives exit status 1.

    Args:
        argv: The command-line arguments without the program name, or None
            to read them from sys.argv.
        data_root: The directory that holds every dataset.
        datasets: The datasets that can be parsed, by name.
        create_parser: The function that returns a PDF parser for a backend.

    Returns:
        The exit status: 0 if the dataset was parsed, 1 if it was not.

    Raises:
        SystemExit: The arguments are not valid, with exit status 2, or
            help was requested, with exit status 0.
    """
    arg_parser = argparse.ArgumentParser(
        description="Parses a dataset's PDFs into a parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument(
        "dataset", choices=sorted(datasets), help="The dataset to parse."
    )
    arg_parser.add_argument(
        "--backend",
        required=True,
        choices=sorted(backend.value for backend in factory.ParserBackend),
        help="The PDF parser backend.",
    )
    arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing parsed-pages file.",
    )
    args = arg_parser.parse_args(argv)
    backend = factory.ParserBackend(args.backend)
    try:
        parse_dataset(
            data_root,
            datasets[args.dataset],
            backend,
            create_parser=create_parser,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, protocol.UnreadablePdfError) as error:
        _logger.error("%s", error)
        return 1
    return 0
