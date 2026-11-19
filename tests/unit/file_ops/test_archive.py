"""Tests for docrag.file_ops.archive."""

from collections.abc import Mapping
import pathlib
import shutil
import stat
import zipfile

import pytest

from docrag.file_ops import archive
from docrag.file_ops import checksum

_CHECKSUM = checksum.Checksum(
    algorithm=checksum.Algorithm.SHA256, hexdigest="a" * 64
)
_OTHER_CHECKSUM = checksum.Checksum(
    algorithm=checksum.Algorithm.SHA256, hexdigest="b" * 64
)
_MARKER_NAME = ".archive-checksum"
_MARKER_TEXT = "sha256:" + "a" * 64
_OTHER_MARKER_TEXT = "sha256:" + "b" * 64
_STAGING_INFIX = ".extract-staging-"
_OVERWRITE = [
    pytest.param(False, id="no-overwrite"),
    pytest.param(True, id="overwrite"),
]


def _write_zip(path: pathlib.Path, members: Mapping[str, bytes]) -> None:
    """Writes a ZIP archive.

    Args:
        path: The archive to create.
        members: Member names mapped to their contents. Names are stored
            as given, so a name may contain `..`.
    """
    with zipfile.ZipFile(path, "w") as zip_file:
        for name, content in members.items():
            zip_file.writestr(name, content)


def _write_zip_with_symlink(
    path: pathlib.Path, link_name: str, link_target: str
) -> None:
    """Writes a ZIP archive with a regular member and a symbolic link.

    Args:
        path: The archive to create.
        link_name: The name of the symbolic link member.
        link_target: The path the link points to.
    """
    link_info = zipfile.ZipInfo(link_name)
    link_info.create_system = 3  # Unix, so the mode bits are read.
    link_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr("ok.txt", b"ok")
        zip_file.writestr(link_info, link_target)


def _names(directory: pathlib.Path) -> list[str]:
    """Returns the sorted names of the entries in a directory.

    Args:
        directory: The directory to list.
    """
    return sorted(path.name for path in directory.iterdir())


def _staging_dirs(target_dir: pathlib.Path) -> list[pathlib.Path]:
    """Returns the paths named as staging directories next to a target.

    Args:
        target_dir: The target directory. Its parent must exist.
    """
    prefix = target_dir.name + _STAGING_INFIX
    return sorted(
        path
        for path in target_dir.parent.iterdir()
        if path.name.startswith(prefix)
    )


def _write_leftover_staging_dir(target_dir: pathlib.Path) -> pathlib.Path:
    """Writes a staging directory as an interrupted call leaves it.

    Args:
        target_dir: The target directory.

    Returns:
        The staging directory.
    """
    staging_dir = target_dir.with_name(target_dir.name + _STAGING_INFIX + "x")
    (staging_dir / "extracted").mkdir(parents=True)
    (staging_dir / "extracted" / "partial.txt").write_bytes(b"partial")
    (staging_dir / "previous").mkdir()
    (staging_dir / "previous" / "previous.txt").write_bytes(b"previous")
    return staging_dir


def test_is_safe_path_inside_base_dir(tmp_path: pathlib.Path) -> None:
    assert archive.is_safe_path(tmp_path, tmp_path / "nested" / "file.txt")


def test_is_safe_path_base_dir_itself(tmp_path: pathlib.Path) -> None:
    assert archive.is_safe_path(tmp_path, tmp_path)


def test_is_safe_path_outside_base_dir(tmp_path: pathlib.Path) -> None:
    base_dir = tmp_path / "base"

    assert not archive.is_safe_path(base_dir, base_dir / ".." / "file.txt")


def test_is_safe_path_symlink_to_outside(tmp_path: pathlib.Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    link = base_dir / "link"
    link.symlink_to(outside_dir, target_is_directory=True)

    assert not archive.is_safe_path(base_dir, link / "file.txt")


def test_secure_zip_extract_safe_archive(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha", "nested/b.txt": b"beta"})
    dest_dir = tmp_path / "out"

    archive.secure_zip_extract(archive_path, dest_dir)

    assert (dest_dir / "a.txt").read_bytes() == b"alpha"
    assert (dest_dir / "nested" / "b.txt").read_bytes() == b"beta"


@pytest.mark.parametrize(
    "member_name",
    [
        pytest.param("../evil.txt", id="parent"),
        pytest.param("/abs/evil.txt", id="absolute"),
        pytest.param("/evil.txt", id="root-directory"),
        pytest.param("nested/../../evil.txt", id="nested-parent"),
    ],
)
def test_secure_zip_extract_unsafe_member(
    tmp_path: pathlib.Path, member_name: str
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"ok.txt": b"ok", member_name: b"evil"})
    dest_dir = tmp_path / "out"

    with pytest.raises(archive.UnsafeArchiveError) as excinfo:
        archive.secure_zip_extract(archive_path, dest_dir)

    assert excinfo.value.archive_path == archive_path
    assert excinfo.value.member_name == member_name
    assert excinfo.value.reason == "resolves outside the extraction directory"
    assert str(excinfo.value) == (
        f"{archive_path}: member {member_name!r} resolves outside the "
        "extraction directory"
    )
    assert not (dest_dir / "ok.txt").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_secure_zip_extract_symlink_member(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip_with_symlink(archive_path, "link", "passwd.txt")
    dest_dir = tmp_path / "out"

    with pytest.raises(archive.UnsafeArchiveError) as excinfo:
        archive.secure_zip_extract(archive_path, dest_dir)

    assert excinfo.value.archive_path == archive_path
    assert excinfo.value.member_name == "link"
    assert excinfo.value.reason == "is a symbolic link"
    assert str(excinfo.value) == (
        f"{archive_path}: member 'link' is a symbolic link"
    )
    assert not (dest_dir / "ok.txt").exists()
    assert not (dest_dir / "link").exists()


def test_secure_zip_extract_missing_archive(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        archive.secure_zip_extract(tmp_path / "missing.zip", tmp_path / "out")


def test_secure_zip_extract_not_a_zip(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(b"not a ZIP archive")

    with pytest.raises(zipfile.BadZipFile):
        archive.secure_zip_extract(archive_path, tmp_path / "out")


def test_secure_zip_extract_parent_is_file(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})

    with pytest.raises(OSError):
        archive.secure_zip_extract(archive_path, archive_path / "out")


def test_extract_archive_new_target(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"

    assert archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert _names(target_dir) == [_MARKER_NAME, "a.txt"]
    assert (target_dir / "a.txt").read_bytes() == b"alpha"
    assert (target_dir / _MARKER_NAME).read_text() == _MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_empty_archive(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {})
    target_dir = tmp_path / "data"

    assert archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    assert not archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert _names(target_dir) == [_MARKER_NAME]
    assert (target_dir / _MARKER_NAME).read_text() == _MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_missing_parent_dir(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "missing" / "data"

    assert archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert _names(target_dir) == [_MARKER_NAME, "a.txt"]
    assert (target_dir / "a.txt").read_bytes() == b"alpha"
    assert (target_dir / _MARKER_NAME).read_text() == _MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert list(target_dir.parent.iterdir()) == [target_dir]


def test_extract_archive_same_checksum(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"
    archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    (target_dir / "a.txt").write_bytes(b"changed")
    added = target_dir / "added.txt"
    added.write_bytes(b"added")

    assert not archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert _names(target_dir) == [_MARKER_NAME, "a.txt", "added.txt"]
    assert (target_dir / "a.txt").read_bytes() == b"changed"
    assert added.read_bytes() == b"added"
    assert (target_dir / _MARKER_NAME).read_text() == _MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_other_archive_overwrite(
    tmp_path: pathlib.Path,
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"old.txt": b"old"})
    target_dir = tmp_path / "data"
    archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    _write_zip(archive_path, {"new.txt": b"new"})

    assert archive.extract_archive(
        archive_path, target_dir, _OTHER_CHECKSUM, overwrite=True
    )

    assert _names(target_dir) == [_MARKER_NAME, "new.txt"]
    assert (target_dir / "new.txt").read_bytes() == b"new"
    assert (target_dir / _MARKER_NAME).read_text() == _OTHER_MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_other_archive_no_overwrite(
    tmp_path: pathlib.Path,
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"old.txt": b"old"})
    target_dir = tmp_path / "data"
    archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    _write_zip(archive_path, {"new.txt": b"new"})

    with pytest.raises(FileExistsError) as excinfo:
        archive.extract_archive(archive_path, target_dir, _OTHER_CHECKSUM)

    assert str(excinfo.value) == (
        f"{target_dir}: holds an earlier extraction and overwrite is false"
    )
    assert _names(target_dir) == [_MARKER_NAME, "old.txt"]
    assert (target_dir / "old.txt").read_bytes() == b"old"
    assert (target_dir / _MARKER_NAME).read_text() == _MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_overwrite_extraction_fails(
    tmp_path: pathlib.Path,
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"old.txt": b"old"})
    target_dir = tmp_path / "data"
    archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    archive_path.write_bytes(b"not a ZIP archive")

    with pytest.raises(zipfile.BadZipFile):
        archive.extract_archive(
            archive_path, target_dir, _OTHER_CHECKSUM, overwrite=True
        )

    assert _names(target_dir) == [_MARKER_NAME, "old.txt"]
    assert (target_dir / "old.txt").read_bytes() == b"old"
    assert (target_dir / _MARKER_NAME).read_text() == _MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_overwrite_move_into_place_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"old.txt": b"old"})
    target_dir = tmp_path / "data"
    archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    _write_zip(archive_path, {"new.txt": b"new"})
    rename = pathlib.Path.rename
    failed_sources: list[pathlib.Path] = []

    def fail_first_rename_onto_target(
        source: pathlib.Path, destination: pathlib.Path
    ) -> pathlib.Path:
        """Fails the first rename onto target_dir and performs the rest.

        Args:
            source: The path to rename.
            destination: The new path.

        Returns:
            The new path.

        Raises:
            OSError: This is the first rename onto target_dir.
        """
        if destination == target_dir and not failed_sources:
            failed_sources.append(source)
            raise OSError(f"{destination}: rename failed")
        return rename(source, destination)

    monkeypatch.setattr(pathlib.Path, "rename", fail_first_rename_onto_target)

    with pytest.raises(OSError) as excinfo:
        archive.extract_archive(
            archive_path, target_dir, _OTHER_CHECKSUM, overwrite=True
        )

    assert str(excinfo.value) == f"{target_dir}: rename failed"
    assert len(failed_sources) == 1
    assert _names(target_dir) == [_MARKER_NAME, "old.txt"]
    assert (target_dir / "old.txt").read_bytes() == b"old"
    assert (target_dir / _MARKER_NAME).read_text() == _MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_overwrite_restore_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"old.txt": b"old"})
    target_dir = tmp_path / "data"
    archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    _write_zip(archive_path, {"new.txt": b"new"})
    rename = pathlib.Path.rename

    def fail_rename_onto_target(
        source: pathlib.Path, destination: pathlib.Path
    ) -> pathlib.Path:
        """Fails every rename onto target_dir and performs the rest.

        Args:
            source: The path to rename.
            destination: The new path.

        Returns:
            The new path.

        Raises:
            OSError: The rename is onto target_dir.
        """
        if destination == target_dir:
            raise OSError(f"{source.name}: rename failed")
        return rename(source, destination)

    monkeypatch.setattr(pathlib.Path, "rename", fail_rename_onto_target)

    with pytest.raises(OSError) as excinfo:
        archive.extract_archive(
            archive_path, target_dir, _OTHER_CHECKSUM, overwrite=True
        )

    assert str(excinfo.value) == "previous: rename failed"
    assert str(excinfo.value.__context__) == "extracted: rename failed"
    assert not target_dir.exists()
    staging_dirs = _staging_dirs(target_dir)
    assert len(staging_dirs) == 1
    assert _names(staging_dirs[0]) == ["extracted", "previous"]
    assert _names(staging_dirs[0] / "previous") == [_MARKER_NAME, "old.txt"]
    assert _names(staging_dirs[0] / "extracted") == [_MARKER_NAME, "new.txt"]
    assert set(tmp_path.iterdir()) == {archive_path, staging_dirs[0]}


def test_extract_archive_staging_removal_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"old.txt": b"old"})
    target_dir = tmp_path / "data"
    archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    _write_zip(archive_path, {"new.txt": b"new"})

    def fail_rmtree(path: pathlib.Path) -> None:
        """Fails as a directory that cannot be removed does.

        Args:
            path: The directory that would be removed.

        Raises:
            OSError: Always.
        """
        raise OSError(f"{path}: removal failed")

    monkeypatch.setattr(shutil, "rmtree", fail_rmtree)

    with pytest.raises(OSError) as excinfo:
        archive.extract_archive(
            archive_path, target_dir, _OTHER_CHECKSUM, overwrite=True
        )

    staging_dirs = _staging_dirs(target_dir)
    assert len(staging_dirs) == 1
    assert str(excinfo.value) == f"{staging_dirs[0]}: removal failed"
    assert _names(target_dir) == [_MARKER_NAME, "new.txt"]
    assert (target_dir / "new.txt").read_bytes() == b"new"
    assert (target_dir / _MARKER_NAME).read_text() == _OTHER_MARKER_TEXT
    assert _names(staging_dirs[0]) == ["previous"]
    assert _names(staging_dirs[0] / "previous") == [_MARKER_NAME, "old.txt"]
    assert set(tmp_path.iterdir()) == {
        archive_path,
        target_dir,
        staging_dirs[0],
    }


@pytest.mark.parametrize("overwrite", _OVERWRITE)
def test_extract_archive_target_without_marker(
    tmp_path: pathlib.Path, overwrite: bool
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"
    target_dir.mkdir()
    (target_dir / "stale.txt").write_bytes(b"stale")

    with pytest.raises(FileExistsError) as excinfo:
        archive.extract_archive(
            archive_path, target_dir, _CHECKSUM, overwrite=overwrite
        )

    assert str(excinfo.value) == (
        f"{target_dir}: exists and has no {_MARKER_NAME} marker"
    )
    assert _names(target_dir) == ["stale.txt"]
    assert (target_dir / "stale.txt").read_bytes() == b"stale"
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


@pytest.mark.parametrize("overwrite", _OVERWRITE)
def test_extract_archive_leftover_staging_dir(
    tmp_path: pathlib.Path, overwrite: bool
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"
    _write_leftover_staging_dir(target_dir)

    assert archive.extract_archive(
        archive_path, target_dir, _CHECKSUM, overwrite=overwrite
    )

    assert _names(target_dir) == [_MARKER_NAME, "a.txt"]
    assert (target_dir / "a.txt").read_bytes() == b"alpha"
    assert (target_dir / _MARKER_NAME).read_text() == _MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_leftover_staging_dir_same_checksum(
    tmp_path: pathlib.Path,
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"
    archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    _write_leftover_staging_dir(target_dir)

    assert not archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert _names(target_dir) == [_MARKER_NAME, "a.txt"]
    assert (target_dir / _MARKER_NAME).read_text() == _MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_leftover_staging_dir_other_target(
    tmp_path: pathlib.Path,
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"
    other_staging_dir = _write_leftover_staging_dir(tmp_path / "other")

    assert archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert _names(other_staging_dir) == ["extracted", "previous"]
    assert set(tmp_path.iterdir()) == {
        archive_path,
        target_dir,
        other_staging_dir,
    }


def test_extract_archive_staging_path_is_symlink(
    tmp_path: pathlib.Path,
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "keep.txt").write_bytes(b"keep")
    staging_link = target_dir.with_name(target_dir.name + _STAGING_INFIX + "x")
    staging_link.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(OSError) as excinfo:
        archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert str(excinfo.value) == (
        f"{staging_link}: symbolic links are not allowed"
    )
    assert staging_link.is_symlink()
    assert _names(real_dir) == ["keep.txt"]
    assert not target_dir.exists()
    assert set(tmp_path.iterdir()) == {archive_path, real_dir, staging_link}


def test_extract_archive_staging_path_is_file(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"
    staging_file = target_dir.with_name(target_dir.name + _STAGING_INFIX + "x")
    staging_file.write_bytes(b"a file")

    with pytest.raises(FileExistsError) as excinfo:
        archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert str(excinfo.value) == (
        f"{staging_file}: exists and is not a directory"
    )
    assert staging_file.read_bytes() == b"a file"
    assert not target_dir.exists()
    assert set(tmp_path.iterdir()) == {archive_path, staging_file}


def test_extract_archive_other_exception_leaves_staging_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"

    def fail_extract(
        archive_path: pathlib.Path, dest_dir: pathlib.Path
    ) -> None:
        """Fails as an extraction stopped by an unlisted error does.

        Args:
            archive_path: The ZIP archive that would be extracted.
            dest_dir: The directory that would be extracted into.

        Raises:
            RuntimeError: Always.
        """
        del archive_path, dest_dir  # Unused.
        raise RuntimeError("extraction stopped")

    monkeypatch.setattr(archive, "secure_zip_extract", fail_extract)
    with pytest.raises(RuntimeError):
        archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert not target_dir.exists()
    assert len(_staging_dirs(target_dir)) == 1

    monkeypatch.undo()
    assert archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert _names(target_dir) == [_MARKER_NAME, "a.txt"]
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_other_exception_between_moves(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"old.txt": b"old"})
    target_dir = tmp_path / "data"
    archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    _write_zip(archive_path, {"new.txt": b"new"})
    rename = pathlib.Path.rename

    def fail_rename_onto_target(
        source: pathlib.Path, destination: pathlib.Path
    ) -> pathlib.Path:
        """Fails every rename onto target_dir and performs the rest.

        Args:
            source: The path to rename.
            destination: The new path.

        Returns:
            The new path.

        Raises:
            RuntimeError: The rename is onto target_dir.
        """
        if destination == target_dir:
            raise RuntimeError("interrupted")
        return rename(source, destination)

    monkeypatch.setattr(pathlib.Path, "rename", fail_rename_onto_target)
    with pytest.raises(RuntimeError):
        archive.extract_archive(
            archive_path, target_dir, _OTHER_CHECKSUM, overwrite=True
        )

    assert not target_dir.exists()
    staging_dirs = _staging_dirs(target_dir)
    assert len(staging_dirs) == 1
    assert _names(staging_dirs[0] / "previous") == [_MARKER_NAME, "old.txt"]

    monkeypatch.undo()
    assert archive.extract_archive(archive_path, target_dir, _OTHER_CHECKSUM)

    assert _names(target_dir) == [_MARKER_NAME, "new.txt"]
    assert (target_dir / _MARKER_NAME).read_text() == _OTHER_MARKER_TEXT
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


@pytest.mark.parametrize("overwrite", _OVERWRITE)
def test_extract_archive_target_is_symlink(
    tmp_path: pathlib.Path, overwrite: bool
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "keep.txt").write_bytes(b"keep")
    target_dir = tmp_path / "data"
    target_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(OSError) as excinfo:
        archive.extract_archive(
            archive_path, target_dir, _CHECKSUM, overwrite=overwrite
        )

    assert str(excinfo.value) == (
        f"{target_dir}: symbolic links are not allowed"
    )
    assert target_dir.is_symlink()
    assert _names(real_dir) == ["keep.txt"]
    assert (real_dir / "keep.txt").read_bytes() == b"keep"
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, real_dir, target_dir}


def test_extract_archive_marker_is_symlink(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"
    archive.extract_archive(archive_path, target_dir, _CHECKSUM)
    marker_path = target_dir / _MARKER_NAME
    marker_path.unlink()
    real_marker = tmp_path / "real-marker"
    real_marker.write_text(_MARKER_TEXT)
    marker_path.symlink_to(real_marker)

    with pytest.raises(OSError) as excinfo:
        archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert str(excinfo.value) == (
        f"{marker_path}: symbolic links are not allowed"
    )
    assert marker_path.is_symlink()
    assert real_marker.read_text() == _MARKER_TEXT
    assert _names(target_dir) == [_MARKER_NAME, "a.txt"]
    assert (target_dir / "a.txt").read_bytes() == b"alpha"
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir, real_marker}


@pytest.mark.parametrize("overwrite", _OVERWRITE)
def test_extract_archive_target_is_file(
    tmp_path: pathlib.Path, overwrite: bool
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})
    target_dir = tmp_path / "data"
    target_dir.write_bytes(b"a file")

    with pytest.raises(FileExistsError) as excinfo:
        archive.extract_archive(
            archive_path, target_dir, _CHECKSUM, overwrite=overwrite
        )

    assert str(excinfo.value) == (
        f"{target_dir}: exists and is not a directory"
    )
    assert target_dir.read_bytes() == b"a file"
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path, target_dir}


def test_extract_archive_unsafe_archive(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"../evil.txt": b"evil"})
    target_dir = tmp_path / "data"

    with pytest.raises(archive.UnsafeArchiveError):
        archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert not target_dir.exists()
    assert not _staging_dirs(target_dir)
    assert not (tmp_path / "evil.txt").exists()
    assert set(tmp_path.iterdir()) == {archive_path}


def test_extract_archive_symlink_member(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip_with_symlink(archive_path, "link", "passwd.txt")
    target_dir = tmp_path / "data"

    with pytest.raises(archive.UnsafeArchiveError):
        archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert not target_dir.exists()
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path}


def test_extract_archive_missing_archive(tmp_path: pathlib.Path) -> None:
    target_dir = tmp_path / "data"

    with pytest.raises(FileNotFoundError):
        archive.extract_archive(tmp_path / "missing.zip", target_dir, _CHECKSUM)

    assert not target_dir.exists()
    assert not _staging_dirs(target_dir)
    assert not list(tmp_path.iterdir())


def test_extract_archive_not_a_zip(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(b"not a ZIP archive")
    target_dir = tmp_path / "data"

    with pytest.raises(zipfile.BadZipFile):
        archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert not target_dir.exists()
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path}


def test_extract_archive_failure_during_extraction(
    tmp_path: pathlib.Path,
) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha", "a.txt/b.txt": b"beta"})
    target_dir = tmp_path / "data"

    with pytest.raises(OSError):
        archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert not target_dir.exists()
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path}


def test_extract_archive_marker_write_fails(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {f"{_MARKER_NAME}/a.txt": b"alpha"})
    target_dir = tmp_path / "data"

    with pytest.raises(IsADirectoryError):
        archive.extract_archive(archive_path, target_dir, _CHECKSUM)

    assert not target_dir.exists()
    assert not _staging_dirs(target_dir)
    assert set(tmp_path.iterdir()) == {archive_path}


def test_extract_archive_parent_is_file(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "data.zip"
    _write_zip(archive_path, {"a.txt": b"alpha"})

    with pytest.raises(OSError):
        archive.extract_archive(archive_path, archive_path / "data", _CHECKSUM)

    assert archive_path.is_file()
    assert set(tmp_path.iterdir()) == {archive_path}
