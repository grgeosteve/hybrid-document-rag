"""Tests for docrag.file_ops.paths."""

import pathlib

import pytest

from docrag.file_ops import paths


def test_is_safe_path_inside_base_dir(tmp_path: pathlib.Path) -> None:
    assert paths.is_safe_path(tmp_path, tmp_path / "nested" / "file.txt")


def test_is_safe_path_base_dir_itself(tmp_path: pathlib.Path) -> None:
    assert paths.is_safe_path(tmp_path, tmp_path)


def test_is_safe_path_outside_base_dir(tmp_path: pathlib.Path) -> None:
    base_dir = tmp_path / "base"

    assert not paths.is_safe_path(base_dir, base_dir / ".." / "file.txt")


def test_is_safe_path_symlink_to_outside(tmp_path: pathlib.Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    link = base_dir / "link"
    link.symlink_to(outside_dir, target_is_directory=True)

    assert not paths.is_safe_path(base_dir, link / "file.txt")


def test_reject_symlink_link_to_file(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"target")
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(OSError) as excinfo:
        paths.reject_symlink(link)

    assert str(excinfo.value) == f"{link}: symbolic links are not allowed"


def test_reject_symlink_dangling_link(tmp_path: pathlib.Path) -> None:
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing.txt")

    with pytest.raises(OSError) as excinfo:
        paths.reject_symlink(link)

    assert str(excinfo.value) == f"{link}: symbolic links are not allowed"


def test_reject_symlink_regular_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"file")

    paths.reject_symlink(path)


def test_reject_symlink_directory(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "directory"
    path.mkdir()

    paths.reject_symlink(path)


def test_reject_symlink_missing_path(tmp_path: pathlib.Path) -> None:
    paths.reject_symlink(tmp_path / "missing")


def test_reject_non_directory_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"file")

    with pytest.raises(FileExistsError) as excinfo:
        paths.reject_non_directory(path)

    assert str(excinfo.value) == f"{path}: exists and is not a directory"


def test_reject_non_directory_directory(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "directory"
    path.mkdir()

    paths.reject_non_directory(path)


def test_reject_non_directory_missing_path(tmp_path: pathlib.Path) -> None:
    paths.reject_non_directory(tmp_path / "missing")
