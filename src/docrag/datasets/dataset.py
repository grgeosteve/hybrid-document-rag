"""The definition of a dataset: its name, its files, and its archive."""

import dataclasses

from docrag.file_ops import download
from docrag.file_ops import paths


@dataclasses.dataclass(frozen=True)
class Dataset:
    """A dataset's files and the archive extracted from them.

    Attributes:
        name: The name of the dataset, used as its directory under the data
            root.
        sources: The files to download.
        archive_source: The ZIP archive among sources to extract.
        extract_dir_name: The name of the directory the archive is extracted
            into, inside the dataset's directory.
    """

    name: str
    sources: tuple[download.Source, ...]
    archive_source: download.Source
    extract_dir_name: str

    def __post_init__(self) -> None:
        """Checks that the fields describe a consistent dataset.

        Raises:
            ValueError: name or extract_dir_name is not a plain file name,
                two sources have the same file name, archive_source is not
                in sources, or extract_dir_name is the file name of a
                source.
        """
        paths.reject_non_plain_name(self.name)
        paths.reject_non_plain_name(self.extract_dir_name)
        filenames: set[str] = set()
        for source in self.sources:
            if source.filename in filenames:
                raise ValueError(
                    f"{source.filename!r}: names more than one source"
                )
            filenames.add(source.filename)
        if self.archive_source not in self.sources:
            raise ValueError(
                f"{self.archive_source.filename!r}: archive source is not in "
                "sources"
            )
        if self.extract_dir_name in filenames:
            raise ValueError(
                f"{self.extract_dir_name!r}: also names a source file"
            )
