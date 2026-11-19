"""Pinned sources and file locations of the OHR-Bench dataset."""

from docrag.datasets import dataset
from docrag.file_ops import checksum
from docrag.file_ops import download

_HF_REVISION = "7f833e3eda9a571a9ea545a8f6d476fa1685033d"
_GITHUB_REVISION = "798dd3db7c27bb07019ff6b3661abd37cedae145"
_HF_BASE_URL = (
    "https://huggingface.co/datasets/opendatalab/OHR-Bench/resolve/"
    f"{_HF_REVISION}"
)
_GITHUB_BASE_URL = (
    "https://raw.githubusercontent.com/opendatalab/OHR-Bench/"
    f"{_GITHUB_REVISION}"
)

PDFS_ARCHIVE = "pdfs.zip"
PDFS_DIR = "pdfs"
GROUND_TRUTH_FILE = "OHR-Bench_v2.parquet"
QA_FILE = "qas_v2.json"

_PDFS_SOURCE = download.Source(
    url=f"{_HF_BASE_URL}/{PDFS_ARCHIVE}",
    filename=PDFS_ARCHIVE,
    checksum=checksum.Checksum(
        algorithm=checksum.Algorithm.SHA256,
        hexdigest="f9bc65f383172c4ea47940c47dfab01dd36c03a120bc0450d7a962917098c783",  # pylint: disable=line-too-long
    ),
)

DATASET = dataset.Dataset(
    name="ohr_bench",
    sources=(
        _PDFS_SOURCE,
        download.Source(
            url=f"{_HF_BASE_URL}/{GROUND_TRUTH_FILE}",
            filename=GROUND_TRUTH_FILE,
            checksum=checksum.Checksum(
                algorithm=checksum.Algorithm.SHA256,
                hexdigest="e7a1a8ac1f72c84a2fcdc0faf3af09190654c41a1dd33a8e3cf52393e4b53c82",  # pylint: disable=line-too-long
            ),
        ),
        # GitHub publishes a git blob SHA-1 for this file, not a SHA-256.
        download.Source(
            url=f"{_GITHUB_BASE_URL}/data/{QA_FILE}",
            filename=QA_FILE,
            checksum=checksum.Checksum(
                algorithm=checksum.Algorithm.GIT_BLOB_SHA1,
                hexdigest="6fc22e117f6acc5284f298defdec835af35722f9",
            ),
        ),
    ),
    archive_source=_PDFS_SOURCE,
    extract_dir_name=PDFS_DIR,
)
