"""Checks paths for symbolic links, file types, and traversal."""

import pathlib


def is_safe_path(base_dir: pathlib.Path, target_path: pathlib.Path) -> bool:
    """Reports whether a path resolves to a location inside a directory.

    Both paths are resolved before they are compared, so `..` components
    and symlinks cannot be used to point outside the directory.

    Args:
        base_dir: The directory the target must stay within.
        target_path: The path to test.

    Returns:
        True if target_path is base_dir itself or lies inside it, False
        otherwise.
    """
    base_path = base_dir.resolve()
    target_path = target_path.resolve()

    return target_path.is_relative_to(base_path)


def reject_symlink(path: pathlib.Path) -> None:
    """Raises an error if a path is a symbolic link.

    Args:
        path: The path to check. It does not have to exist.

    Raises:
        OSError: path is a symbolic link, including one whose target does
            not exist.
    """
    if path.is_symlink():
        raise OSError(f"{path}: symbolic links are not allowed")


def reject_non_directory(path: pathlib.Path) -> None:
    """Raises an error if a path exists and is not a directory.

    Args:
        path: The path to check. It does not have to exist.

    Raises:
        FileExistsError: path exists and is not a directory.
    """
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"{path}: exists and is not a directory")


def reject_non_file(path: pathlib.Path) -> None:
    """Raises an error if a path exists and is not a regular file.

    Args:
        path: The path to check. It does not have to exist.

    Raises:
        FileExistsError: path exists and is not a regular file.
    """
    if path.exists() and not path.is_file():
        raise FileExistsError(f"{path}: exists and is not a regular file")
