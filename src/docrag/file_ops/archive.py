"""Extracts ZIP archives, rejecting path traversal and symbolic links."""

import pathlib
import shutil
import stat
import tempfile
import zipfile

from docrag.file_ops import checksum
from docrag.file_ops import paths

_STAGING_INFIX = ".extract-staging-"
_MARKER_NAME = ".archive-checksum"


class UnsafeArchiveError(Exception):
    """An archive member cannot be extracted safely.

    Attributes:
        archive_path: The archive that holds the member.
        member_name: The member's name as stored in the archive.
        reason: What makes the member unsafe.
    """

    def __init__(
        self, archive_path: pathlib.Path, member_name: str, reason: str
    ) -> None:
        """Initialises the error with the archive and the offending member.

        Args:
            archive_path: The archive that holds the member.
            member_name: The member's name as stored in the archive.
            reason: What makes the member unsafe, completing the sentence
                "member ... ".
        """
        self.archive_path = archive_path
        self.member_name = member_name
        self.reason = reason

        super().__init__(f"{archive_path}: member {member_name!r} {reason}")


def secure_zip_extract(
    archive_path: pathlib.Path, dest_dir: pathlib.Path
) -> None:
    """Extracts a ZIP archive, rejecting unsafe members.

    A member is unsafe if it is a symbolic link or resolves to a path
    outside dest_dir. Every member is checked before anything is written,
    so an archive with an unsafe member is rejected as a whole and leaves
    no files behind.

    Args:
        archive_path: The ZIP archive to extract.
        dest_dir: The directory to extract into.

    Raises:
        FileNotFoundError: archive_path does not exist.
        zipfile.BadZipFile: archive_path is not a ZIP archive.
        UnsafeArchiveError: A member of the archive is a symbolic link or
            resolves to a path outside dest_dir.
        OSError: The archive cannot be read, or its members cannot be
            written under dest_dir.
    """
    dest_dir = dest_dir.resolve()

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            if stat.S_ISLNK(member.external_attr >> 16):
                raise UnsafeArchiveError(
                    archive_path, member.filename, "is a symbolic link"
                )

            member_target = dest_dir / member.filename

            if not paths.is_safe_path(dest_dir, member_target):
                raise UnsafeArchiveError(
                    archive_path,
                    member.filename,
                    "resolves outside the extraction directory",
                )

        zip_ref.extractall(dest_dir)


def _read_marker(directory: pathlib.Path) -> str | None:
    """Returns the marker that extract_archive wrote into a directory.

    Args:
        directory: An existing directory.

    Returns:
        The marker's text, or None if the directory has no marker file.

    Raises:
        OSError: The marker is a symbolic link or cannot be read.
    """
    marker_path = directory / _MARKER_NAME
    paths.reject_symlink(marker_path)
    if not marker_path.is_file():
        return None
    return marker_path.read_text()


def _check_removable(
    directory: pathlib.Path, marker: str | None, overwrite: bool
) -> None:
    """Raises an error unless an existing directory may be removed.

    A directory may be removed only if it holds a marker, which shows that
    extract_archive created it, and overwrite is true.

    Args:
        directory: The existing directory.
        marker: The directory's marker text, or None if it has none.
        overwrite: Whether the caller allows removal.

    Raises:
        FileExistsError: The directory has no marker, or overwrite is false.
    """
    if marker is None:
        raise FileExistsError(
            f"{directory}: exists and has no {_MARKER_NAME} marker"
        )
    if not overwrite:
        raise FileExistsError(
            f"{directory}: holds an earlier extraction and overwrite is false"
        )


def _remove_staging_dirs(target_dir: pathlib.Path) -> None:
    """Removes the staging directories that earlier calls left for a target.

    Args:
        target_dir: The target directory. Its parent does not have to exist.

    Raises:
        FileExistsError: A path named as a staging directory is not a
            directory.
        OSError: A path named as a staging directory is a symbolic link, or
            a staging directory cannot be removed.
    """
    if not target_dir.parent.is_dir():
        return
    prefix = target_dir.name + _STAGING_INFIX
    for path in sorted(target_dir.parent.iterdir()):
        if path.name.startswith(prefix):
            paths.reject_symlink(path)
            paths.reject_non_directory(path)
            shutil.rmtree(path)


def extract_archive(
    archive_path: pathlib.Path,
    target_dir: pathlib.Path,
    archive_checksum: checksum.Checksum,
    *,
    overwrite: bool = False,
) -> bool:
    """Extracts a ZIP archive into a directory, unless that was done before.

    The target holds a marker file recording the archive's checksum. A
    target with this archive's marker is kept. An existing target is
    replaced only if it holds a marker and overwrite is true.

    The archive is extracted into a staging directory next to the target
    and then moved into place. An earlier target is removed only after the
    new one is in place. A failed call can leave a staging directory
    behind, and the next call removes it.

    Args:
        archive_path: The ZIP archive to extract.
        target_dir: The directory the contents end up in.
        archive_checksum: The checksum of the archive, recorded in the
            marker.
        overwrite: Whether a target that holds another archive's marker
            may be replaced.

    Returns:
        True if the archive was extracted, False if target_dir already
        held this archive's marker.

    Raises:
        FileExistsError: target_dir or a path named as a staging directory
            is not a directory, or target_dir has no marker, or holds
            another archive's marker and overwrite is false.
        FileNotFoundError: archive_path does not exist.
        zipfile.BadZipFile: archive_path is not a ZIP archive.
        UnsafeArchiveError: A member of the archive is a symbolic link or
            resolves to a path outside the directory it is extracted into.
        OSError: target_dir, its marker, or a path named as a staging
            directory is a symbolic link, or a file operation fails.
    """
    marker_text = (
        f"{archive_checksum.algorithm.value}:{archive_checksum.hexdigest}"
    )
    paths.reject_symlink(target_dir)
    paths.reject_non_directory(target_dir)
    _remove_staging_dirs(target_dir)
    replace_target = False
    if target_dir.is_dir():
        target_marker = _read_marker(target_dir)
        if target_marker == marker_text:
            return False
        _check_removable(target_dir, target_marker, overwrite)
        replace_target = True

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = pathlib.Path(
        tempfile.mkdtemp(
            prefix=target_dir.name + _STAGING_INFIX, dir=target_dir.parent
        )
    )
    extracted_dir = staging_dir / "extracted"
    previous_dir = staging_dir / "previous"
    target_moved = False
    try:
        extracted_dir.mkdir()
        secure_zip_extract(archive_path, extracted_dir)
        # Write marker with algorithm and checksum of the original archive
        (extracted_dir / _MARKER_NAME).write_text(marker_text)
        if replace_target:
            target_dir.rename(previous_dir)
            target_moved = True
        extracted_dir.rename(target_dir)
    except (OSError, zipfile.BadZipFile, UnsafeArchiveError):
        if target_moved:
            previous_dir.rename(target_dir)
        shutil.rmtree(staging_dir)
        raise

    shutil.rmtree(staging_dir)
    return True
