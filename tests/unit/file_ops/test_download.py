"""Tests for docrag.file_ops.download."""

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import hashlib
import pathlib
import secrets
import stat

import pytest
import requests

from docrag.file_ops import checksum
from docrag.file_ops import download

_URL = "https://example.com/data.bin"
_FILENAME = "data.bin"
_CONTENT = b"contents of the file"
_OLD_CONTENT = b"contents of an earlier revision"
_SHA256_CHECKSUM = checksum.Checksum(
    algorithm=checksum.Algorithm.SHA256,
    hexdigest=hashlib.sha256(_CONTENT).hexdigest(),
)
_GIT_BLOB_SHA1_CHECKSUM = checksum.Checksum(
    algorithm=checksum.Algorithm.GIT_BLOB_SHA1,
    hexdigest=hashlib.sha1(b"blob %d\0" % len(_CONTENT) + _CONTENT).hexdigest(),
)
_CHECKSUMS = [
    pytest.param(_SHA256_CHECKSUM, id="sha256"),
    pytest.param(_GIT_BLOB_SHA1_CHECKSUM, id="git-blob-sha1"),
]
_OVERWRITE = [
    pytest.param(False, id="no-overwrite"),
    pytest.param(True, id="overwrite"),
]
_PARTIAL_INFIX = ".download-"


def _source(expected: checksum.Checksum = _SHA256_CHECKSUM) -> download.Source:
    """Returns the source of the test file.

    Args:
        expected: The checksum the source publishes.
    """
    return download.Source(url=_URL, filename=_FILENAME, checksum=expected)


def _partial_files(data_dir: pathlib.Path) -> list[pathlib.Path]:
    """Returns the paths named as partial files of the test file.

    Args:
        data_dir: The data directory. It must exist.
    """
    prefix = _FILENAME + _PARTIAL_INFIX
    return sorted(
        path for path in data_dir.iterdir() if path.name.startswith(prefix)
    )


def _leftover_partial_path(
    data_dir: pathlib.Path, filename: str = _FILENAME, token: str = "x"
) -> pathlib.Path:
    """Returns a path named as a partial file that an earlier call left.

    Args:
        data_dir: The data directory.
        filename: The name of the file the partial file belongs to.
        token: The random part of the name.
    """
    return data_dir / f"{filename}{_PARTIAL_INFIX}{token}"


class _FakeTokens:
    """Returns fixed random tokens, creating a file under the taken one.

    The partial file of the taken token is created when the token is
    returned, as if another process had just created it.
    """

    def __init__(
        self, data_dir: pathlib.Path, tokens: Sequence[str], taken: str
    ) -> None:
        """Initialises the tokens.

        Args:
            data_dir: The data directory the partial files are in.
            tokens: The tokens to return, in order.
            taken: The token whose partial file already exists.
        """
        self._data_dir = data_dir
        self._tokens = list(tokens)
        self._taken = taken

    def __call__(self, nbytes: int) -> str:
        """Returns the next token, creating its file if it is taken.

        Args:
            nbytes: The number of random bytes the caller asks for.

        Returns:
            The next token.
        """
        del nbytes  # The tokens are fixed.
        token = self._tokens.pop(0)
        if token == self._taken:
            _leftover_partial_path(self._data_dir, token=token).write_bytes(
                b"taken"
            )
        return token


class _FakeDownloader:
    """Writes fixed bytes to the destination and records each call.

    Attributes:
        calls: The URL and destination path of each call, in order.
    """

    def __init__(self, content: bytes) -> None:
        """Initialises the downloader.

        Args:
            content: The bytes written on each call.
        """
        self._content = content
        self.calls: list[tuple[str, pathlib.Path]] = []

    def __call__(self, url: str, destination_path: pathlib.Path) -> None:
        """Writes the content to a file.

        Args:
            url: The URL that would be downloaded.
            destination_path: The file to write.
        """
        self.calls.append((url, destination_path))
        destination_path.write_bytes(self._content)


class _FailingDownloader:
    """Writes part of the content, then fails as a lost connection does.

    Attributes:
        calls: The URL and destination path of each call, in order.
    """

    def __init__(self) -> None:
        """Initialises the downloader."""
        self.calls: list[tuple[str, pathlib.Path]] = []

    def __call__(self, url: str, destination_path: pathlib.Path) -> None:
        """Writes the first bytes of the content, then raises.

        Args:
            url: The URL that would be downloaded.
            destination_path: The file to write.

        Raises:
            requests.ConnectionError: Always.
        """
        self.calls.append((url, destination_path))
        destination_path.write_bytes(_CONTENT[:5])
        raise requests.ConnectionError("connection lost")


class _FakeRaw:
    """Serves fixed chunks in place of urllib3's streamed response.

    Attributes:
        released: Whether the response released its connection.
    """

    def __init__(
        self,
        chunks: Sequence[bytes],
        error: requests.RequestException | None = None,
    ) -> None:
        """Initialises the stream.

        Args:
            chunks: The pieces of the body, in the order they are served.
            error: The error raised after the last piece, or None if the
                transfer completes.
        """
        self._chunks = chunks
        self._error = error
        self.released = False

    def stream(self, amt: int, decode_content: bool) -> Iterator[bytes]:
        """Yields the pieces of the body, then raises the error if any.

        Args:
            amt: The size the caller asks for.
            decode_content: Whether the caller asks for decoded content.

        Yields:
            Each piece the stream was created with, whatever its size.

        Raises:
            requests.RequestException: The stream was created with an
                error.
        """
        del amt, decode_content  # The pieces are served as given.
        yield from self._chunks
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        """Does nothing. There is no connection to close."""

    def release_conn(self) -> None:
        """Records that the connection was released."""
        self.released = True


def _response(
    raw: _FakeRaw, headers: Mapping[str, str], status_code: int = 200
) -> requests.Response:
    """Builds a real response that reads its body from a fake stream.

    Args:
        raw: The stream the body is read from.
        headers: The response headers.
        status_code: The HTTP status.

    Returns:
        The response.
    """
    response = requests.Response()
    response.status_code = status_code
    response.headers.update(headers)
    response.raw = raw
    response.url = _URL
    return response


class _FakeGet:
    """Returns a fixed response and records each call.

    Attributes:
        calls: The positional and keyword arguments of each call.
    """

    def __init__(self, response: requests.Response) -> None:
        """Initialises the fake.

        Args:
            response: The response every call returns.
        """
        self._response = response
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> requests.Response:
        """Records the call and returns the response.

        Args:
            *args: The positional arguments of the call.
            **kwargs: The keyword arguments of the call.

        Returns:
            The fixed response.
        """
        self.calls.append((args, kwargs))
        return self._response


def test_source_plain_filename() -> None:
    source = _source()

    assert source.filename == _FILENAME


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param("", id="empty"),
        pytest.param(".", id="dot"),
        pytest.param("..", id="dot-dot"),
        pytest.param("nested/data.bin", id="nested"),
        pytest.param("../data.bin", id="parent"),
        pytest.param("/data.bin", id="absolute"),
    ],
)
def test_source_invalid_filename(filename: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        download.Source(url=_URL, filename=filename, checksum=_SHA256_CHECKSUM)

    assert str(excinfo.value) == f"{filename!r}: not a plain file name"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("12", 12, id="valid"),
        pytest.param("0", 0, id="zero"),
        pytest.param(None, None, id="missing"),
        pytest.param("", None, id="empty"),
        pytest.param("abc", None, id="not-a-number"),
        pytest.param("12, 12", None, id="repeated"),
        pytest.param("-5", None, id="negative"),
        pytest.param("+12", None, id="sign"),
        pytest.param("1_000", None, id="underscore"),
        pytest.param("١٢", None, id="non-ascii-digits"),
    ],
)
def test_parse_content_length_values(
    value: str | None, expected: int | None
) -> None:
    assert download.parse_content_length(value) == expected


def test_download_data_ok_response(tmp_path: pathlib.Path) -> None:
    raw = _FakeRaw([b"first ", b"second"])
    fake_get = _FakeGet(_response(raw, {"Content-Length": "12"}))
    destination_path = tmp_path / _FILENAME

    download.download_data(_URL, destination_path, http_get=fake_get)

    assert destination_path.read_bytes() == b"first second"
    assert raw.released


def test_download_data_no_content_length(tmp_path: pathlib.Path) -> None:
    fake_get = _FakeGet(_response(_FakeRaw([b"first ", b"second"]), {}))
    destination_path = tmp_path / _FILENAME

    download.download_data(_URL, destination_path, http_get=fake_get)

    assert destination_path.read_bytes() == b"first second"


def test_download_data_empty_chunk(tmp_path: pathlib.Path) -> None:
    fake_get = _FakeGet(_response(_FakeRaw([b"first ", b"", b"second"]), {}))
    destination_path = tmp_path / _FILENAME

    download.download_data(_URL, destination_path, http_get=fake_get)

    assert destination_path.read_bytes() == b"first second"


def test_download_data_existing_file(tmp_path: pathlib.Path) -> None:
    fake_get = _FakeGet(_response(_FakeRaw([b"new"]), {}))
    destination_path = tmp_path / _FILENAME
    destination_path.write_bytes(b"old and longer")

    download.download_data(_URL, destination_path, http_get=fake_get)

    assert destination_path.read_bytes() == b"new"


def test_download_data_request_arguments(tmp_path: pathlib.Path) -> None:
    fake_get = _FakeGet(_response(_FakeRaw([b"first"]), {}))

    download.download_data(_URL, tmp_path / _FILENAME, http_get=fake_get)

    assert fake_get.calls == [
        (
            (_URL,),
            {
                "stream": True,
                "timeout": 30,
                "headers": {"Accept-Encoding": "identity"},
            },
        )
    ]


def test_download_data_error_status(tmp_path: pathlib.Path) -> None:
    raw = _FakeRaw([])
    fake_get = _FakeGet(_response(raw, {}, status_code=404))
    destination_path = tmp_path / _FILENAME

    with pytest.raises(requests.HTTPError):
        download.download_data(_URL, destination_path, http_get=fake_get)

    assert not destination_path.exists()
    assert raw.released


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            requests.ConnectionError("connection refused"),
            id="connection-error",
        ),
        pytest.param(requests.Timeout("timed out"), id="timeout"),
    ],
)
def test_download_data_request_error(
    tmp_path: pathlib.Path, error: requests.RequestException
) -> None:
    def failing_get(*args: object, **kwargs: object) -> requests.Response:
        del args, kwargs  # Every request fails the same way.
        raise error

    destination_path = tmp_path / _FILENAME

    with pytest.raises(type(error)):
        download.download_data(_URL, destination_path, http_get=failing_get)

    assert not destination_path.exists()


def test_download_data_error_during_transfer(tmp_path: pathlib.Path) -> None:
    raw = _FakeRaw(
        [b"first "],
        requests.exceptions.ChunkedEncodingError("connection broken"),
    )
    fake_get = _FakeGet(_response(raw, {"Content-Length": "12"}))

    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        download.download_data(_URL, tmp_path / _FILENAME, http_get=fake_get)

    assert raw.released


def test_download_data_missing_directory(tmp_path: pathlib.Path) -> None:
    fake_get = _FakeGet(_response(_FakeRaw([b"first"]), {}))

    with pytest.raises(FileNotFoundError):
        download.download_data(
            _URL, tmp_path / "missing" / _FILENAME, http_get=fake_get
        )


@pytest.mark.parametrize("expected", _CHECKSUMS)
def test_ensure_file_new_file(
    tmp_path: pathlib.Path, expected: checksum.Checksum
) -> None:
    destination_path = tmp_path / _FILENAME
    downloader = _FakeDownloader(_CONTENT)

    assert download.ensure_file(_source(expected), tmp_path, downloader)

    assert len(downloader.calls) == 1
    assert destination_path.read_bytes() == _CONTENT
    assert not _partial_files(tmp_path)
    assert list(tmp_path.iterdir()) == [destination_path]


def test_ensure_file_downloads_into_partial_file(
    tmp_path: pathlib.Path,
) -> None:
    downloader = _FakeDownloader(_CONTENT)

    download.ensure_file(_source(), tmp_path, downloader)

    assert len(downloader.calls) == 1
    url, partial_path = downloader.calls[0]
    assert url == _URL
    assert partial_path.parent == tmp_path
    assert partial_path.name.startswith(_FILENAME + _PARTIAL_INFIX)


def test_ensure_file_default_permissions(tmp_path: pathlib.Path) -> None:
    reference_path = tmp_path / "reference.bin"
    reference_path.write_bytes(b"")
    data_dir = tmp_path / "data"

    download.ensure_file(_source(), data_dir, _FakeDownloader(_CONTENT))

    assert stat.S_IMODE((data_dir / _FILENAME).stat().st_mode) == (
        stat.S_IMODE(reference_path.stat().st_mode)
    )


def test_ensure_file_partial_name_taken(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination_path = tmp_path / _FILENAME
    taken_path = _leftover_partial_path(tmp_path, token="taken")
    monkeypatch.setattr(
        secrets, "token_hex", _FakeTokens(tmp_path, ["taken", "free"], "taken")
    )
    downloader = _FakeDownloader(_CONTENT)

    assert download.ensure_file(_source(), tmp_path, downloader)

    assert [path for _, path in downloader.calls] == [
        _leftover_partial_path(tmp_path, token="free")
    ]
    assert taken_path.read_bytes() == b"taken"
    assert destination_path.read_bytes() == _CONTENT
    assert set(tmp_path.iterdir()) == {destination_path, taken_path}


def test_ensure_file_no_free_partial_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination_path = tmp_path / _FILENAME
    taken_path = _leftover_partial_path(tmp_path, token="taken")
    monkeypatch.setattr(
        secrets, "token_hex", _FakeTokens(tmp_path, ["taken"] * 100, "taken")
    )
    downloader = _FakeDownloader(_CONTENT)

    with pytest.raises(FileExistsError) as excinfo:
        download.ensure_file(_source(), tmp_path, downloader)

    assert str(excinfo.value) == (
        f"{destination_path}: no free partial file name after 100 attempts"
    )
    assert not downloader.calls
    assert taken_path.read_bytes() == b"taken"
    assert list(tmp_path.iterdir()) == [taken_path]


def test_ensure_file_missing_data_dir(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "data" / "ohr_bench"

    assert download.ensure_file(_source(), data_dir, _FakeDownloader(_CONTENT))

    assert (data_dir / _FILENAME).read_bytes() == _CONTENT
    assert list(data_dir.iterdir()) == [data_dir / _FILENAME]


def test_ensure_file_data_dir_is_file(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.write_bytes(b"")
    downloader = _FakeDownloader(_CONTENT)

    with pytest.raises(FileExistsError):
        download.ensure_file(_source(), data_dir, downloader)

    assert not downloader.calls
    assert data_dir.read_bytes() == b""


@pytest.mark.parametrize("overwrite", _OVERWRITE)
def test_ensure_file_matching_file(
    tmp_path: pathlib.Path, overwrite: bool
) -> None:
    destination_path = tmp_path / _FILENAME
    destination_path.write_bytes(_CONTENT)
    downloader = _FakeDownloader(_CONTENT)

    assert not download.ensure_file(
        _source(), tmp_path, downloader, overwrite=overwrite
    )

    assert not downloader.calls
    assert destination_path.read_bytes() == _CONTENT
    assert list(tmp_path.iterdir()) == [destination_path]


def test_ensure_file_mismatching_file_no_overwrite(
    tmp_path: pathlib.Path,
) -> None:
    destination_path = tmp_path / _FILENAME
    destination_path.write_bytes(_OLD_CONTENT)
    downloader = _FakeDownloader(_CONTENT)

    with pytest.raises(FileExistsError) as excinfo:
        download.ensure_file(_source(), tmp_path, downloader)

    assert str(excinfo.value) == (
        f"{destination_path}: does not match its checksum and overwrite is "
        "false"
    )
    assert not downloader.calls
    assert destination_path.read_bytes() == _OLD_CONTENT
    assert list(tmp_path.iterdir()) == [destination_path]


@pytest.mark.parametrize("expected", _CHECKSUMS)
def test_ensure_file_mismatching_file_overwrite(
    tmp_path: pathlib.Path, expected: checksum.Checksum
) -> None:
    destination_path = tmp_path / _FILENAME
    destination_path.write_bytes(_OLD_CONTENT)
    downloader = _FakeDownloader(_CONTENT)

    assert download.ensure_file(
        _source(expected), tmp_path, downloader, overwrite=True
    )

    assert len(downloader.calls) == 1
    assert destination_path.read_bytes() == _CONTENT
    assert list(tmp_path.iterdir()) == [destination_path]


def test_ensure_file_overwrite_download_fails(tmp_path: pathlib.Path) -> None:
    destination_path = tmp_path / _FILENAME
    destination_path.write_bytes(_OLD_CONTENT)

    with pytest.raises(requests.ConnectionError):
        download.ensure_file(
            _source(), tmp_path, _FailingDownloader(), overwrite=True
        )

    assert destination_path.read_bytes() == _OLD_CONTENT
    assert not _partial_files(tmp_path)
    assert list(tmp_path.iterdir()) == [destination_path]


def test_ensure_file_overwrite_checksum_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    destination_path = tmp_path / _FILENAME
    destination_path.write_bytes(_OLD_CONTENT)

    with pytest.raises(download.ChecksumMismatchError):
        download.ensure_file(
            _source(),
            tmp_path,
            _FakeDownloader(b"also wrong"),
            overwrite=True,
        )

    assert destination_path.read_bytes() == _OLD_CONTENT
    assert not _partial_files(tmp_path)
    assert list(tmp_path.iterdir()) == [destination_path]


@pytest.mark.parametrize("correct", _CHECKSUMS)
def test_ensure_file_checksum_mismatch(
    tmp_path: pathlib.Path, correct: checksum.Checksum
) -> None:
    wrong = checksum.Checksum(
        algorithm=correct.algorithm, hexdigest="0" * len(correct.hexdigest)
    )
    destination_path = tmp_path / _FILENAME

    with pytest.raises(download.ChecksumMismatchError) as excinfo:
        download.ensure_file(
            _source(wrong), tmp_path, _FakeDownloader(_CONTENT)
        )

    assert excinfo.value.path == destination_path
    assert excinfo.value.expected == wrong
    assert excinfo.value.actual == correct.hexdigest
    assert str(excinfo.value) == (
        f"{destination_path}: {wrong.algorithm.value} checksum mismatch.\n"
        f"Expected: {wrong.hexdigest}\n"
        f"Actual:   {correct.hexdigest}"
    )
    assert not destination_path.exists()
    assert not _partial_files(tmp_path)
    assert not list(tmp_path.iterdir())


def test_ensure_file_download_failure(tmp_path: pathlib.Path) -> None:
    downloader = _FailingDownloader()

    with pytest.raises(requests.ConnectionError):
        download.ensure_file(_source(), tmp_path, downloader)

    assert len(downloader.calls) == 1
    assert not (tmp_path / _FILENAME).exists()
    assert not _partial_files(tmp_path)
    assert not list(tmp_path.iterdir())


def test_ensure_file_rename_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination_path = tmp_path / _FILENAME

    def fail_rename(
        source: pathlib.Path, destination: pathlib.Path
    ) -> pathlib.Path:
        """Fails as a rename that the file system refuses does.

        Args:
            source: The path to rename.
            destination: The new path.

        Raises:
            OSError: Always.
        """
        del source  # Unused.
        raise OSError(f"{destination}: rename failed")

    monkeypatch.setattr(pathlib.Path, "rename", fail_rename)

    with pytest.raises(OSError) as excinfo:
        download.ensure_file(_source(), tmp_path, _FakeDownloader(_CONTENT))

    assert str(excinfo.value) == f"{destination_path}: rename failed"
    assert not destination_path.exists()
    assert not _partial_files(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("overwrite", _OVERWRITE)
def test_ensure_file_destination_is_symlink(
    tmp_path: pathlib.Path, overwrite: bool
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    real_file = data_dir / "real.bin"
    real_file.write_bytes(_OLD_CONTENT)
    destination_path = data_dir / _FILENAME
    destination_path.symlink_to(real_file)
    downloader = _FakeDownloader(_CONTENT)

    with pytest.raises(OSError) as excinfo:
        download.ensure_file(
            _source(), data_dir, downloader, overwrite=overwrite
        )

    assert str(excinfo.value) == (
        f"{destination_path}: symbolic links are not allowed"
    )
    assert not downloader.calls
    assert destination_path.is_symlink()
    assert real_file.read_bytes() == _OLD_CONTENT
    assert set(data_dir.iterdir()) == {real_file, destination_path}


def test_ensure_file_destination_outside_data_dir(
    tmp_path: pathlib.Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(_OLD_CONTENT)
    destination_path = data_dir / _FILENAME
    destination_path.symlink_to(outside_file)
    downloader = _FakeDownloader(_CONTENT)

    with pytest.raises(download.UnsafeDestinationError) as excinfo:
        download.ensure_file(_source(), data_dir, downloader, overwrite=True)

    assert excinfo.value.destination_path == destination_path
    assert excinfo.value.data_dir == data_dir
    assert str(excinfo.value) == (
        f"{destination_path}: resolves outside {data_dir}"
    )
    assert not downloader.calls
    assert outside_file.read_bytes() == _OLD_CONTENT
    assert list(data_dir.iterdir()) == [destination_path]


@pytest.mark.parametrize("overwrite", _OVERWRITE)
def test_ensure_file_destination_is_directory(
    tmp_path: pathlib.Path, overwrite: bool
) -> None:
    destination_path = tmp_path / _FILENAME
    destination_path.mkdir()
    downloader = _FakeDownloader(_CONTENT)

    with pytest.raises(FileExistsError) as excinfo:
        download.ensure_file(
            _source(), tmp_path, downloader, overwrite=overwrite
        )

    assert str(excinfo.value) == (
        f"{destination_path}: exists and is not a regular file"
    )
    assert not downloader.calls
    assert destination_path.is_dir()
    assert list(tmp_path.iterdir()) == [destination_path]


def test_ensure_file_leftover_partial_file(tmp_path: pathlib.Path) -> None:
    destination_path = tmp_path / _FILENAME
    _leftover_partial_path(tmp_path).write_bytes(b"left by an interrupted run")

    assert download.ensure_file(_source(), tmp_path, _FakeDownloader(_CONTENT))

    assert destination_path.read_bytes() == _CONTENT
    assert not _partial_files(tmp_path)
    assert list(tmp_path.iterdir()) == [destination_path]


def test_ensure_file_leftover_partial_file_matching_file(
    tmp_path: pathlib.Path,
) -> None:
    destination_path = tmp_path / _FILENAME
    destination_path.write_bytes(_CONTENT)
    _leftover_partial_path(tmp_path).write_bytes(b"left by an interrupted run")
    downloader = _FakeDownloader(_CONTENT)

    assert not download.ensure_file(_source(), tmp_path, downloader)

    assert not downloader.calls
    assert not _partial_files(tmp_path)
    assert list(tmp_path.iterdir()) == [destination_path]


def test_ensure_file_leftover_partial_file_other_source(
    tmp_path: pathlib.Path,
) -> None:
    destination_path = tmp_path / _FILENAME
    other_partial_path = _leftover_partial_path(tmp_path, "other.bin")
    other_partial_path.write_bytes(b"left by an interrupted run")

    assert download.ensure_file(_source(), tmp_path, _FakeDownloader(_CONTENT))

    assert other_partial_path.read_bytes() == b"left by an interrupted run"
    assert set(tmp_path.iterdir()) == {destination_path, other_partial_path}


def test_ensure_file_partial_path_is_symlink(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    real_file = tmp_path / "real.bin"
    real_file.write_bytes(_OLD_CONTENT)
    partial_path = _leftover_partial_path(data_dir)
    partial_path.symlink_to(real_file)
    downloader = _FakeDownloader(_CONTENT)

    with pytest.raises(OSError) as excinfo:
        download.ensure_file(_source(), data_dir, downloader)

    assert str(excinfo.value) == (
        f"{partial_path}: symbolic links are not allowed"
    )
    assert not downloader.calls
    assert partial_path.is_symlink()
    assert real_file.read_bytes() == _OLD_CONTENT
    assert list(data_dir.iterdir()) == [partial_path]


def test_ensure_file_partial_path_is_directory(tmp_path: pathlib.Path) -> None:
    partial_path = _leftover_partial_path(tmp_path)
    partial_path.mkdir()
    downloader = _FakeDownloader(_CONTENT)

    with pytest.raises(FileExistsError) as excinfo:
        download.ensure_file(_source(), tmp_path, downloader)

    assert str(excinfo.value) == (
        f"{partial_path}: exists and is not a regular file"
    )
    assert not downloader.calls
    assert partial_path.is_dir()
    assert list(tmp_path.iterdir()) == [partial_path]
