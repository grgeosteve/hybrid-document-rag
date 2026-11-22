"""Smoke test for scripts/parse_dataset.py."""

import subprocess
import sys

import pytest


def test_parse_dataset_help(pytestconfig: pytest.Config) -> None:
    script = pytestconfig.rootpath / "scripts" / "parse_dataset.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        check=True,
        text=True,
    )

    assert "{ohr_bench}" in result.stdout
    assert "--backend" in result.stdout
    assert "{pypdfium2}" in result.stdout
    assert "--overwrite" in result.stdout
