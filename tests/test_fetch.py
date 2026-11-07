"""Weights arrive verified, or they do not arrive.

A model file that changed upstream turns every number ever measured with it
into a number about a different model, and nothing else in the system would
notice. So the checksum is the test.
"""

import hashlib

import pytest

from earshot import fetch
from earshot.engines import EngineError


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EARSHOT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("EARSHOT_NO_DOWNLOAD", raising=False)
    return tmp_path


def _asset(tmp_path, payload=b"weights", name="model.bin"):
    source = tmp_path / "upstream.bin"
    source.write_bytes(payload)
    return fetch.Asset(name, source.as_uri(), hashlib.sha256(payload).hexdigest())


def test_a_good_file_is_fetched_once(cache, capsys):
    asset = _asset(cache)
    path = fetch.ensure(asset)
    assert path.read_bytes() == b"weights"
    assert "fetching model.bin" in capsys.readouterr().err

    # Second call is a verification, not a download.
    fetch.ensure(asset)
    assert "fetching" not in capsys.readouterr().err


def test_a_wrong_checksum_is_refused_and_not_left_behind(cache):
    asset = _asset(cache)
    wrong = fetch.Asset(asset.name, asset.url, "0" * 64)
    with pytest.raises(EngineError) as caught:
        fetch.ensure(wrong)
    assert "checksum is wrong" in str(caught.value)
    # Nothing usable is left: a bad file that stays would be picked up next run.
    assert not wrong.path.exists()


def test_a_corrupted_cached_file_is_replaced(cache, capsys):
    asset = _asset(cache)
    fetch.ensure(asset)
    asset.path.write_bytes(b"half a download")
    capsys.readouterr()
    assert fetch.ensure(asset).read_bytes() == b"weights"
    assert "does not match its checksum" in capsys.readouterr().err


def test_downloading_can_be_switched_off(cache, monkeypatch):
    monkeypatch.setenv("EARSHOT_NO_DOWNLOAD", "1")
    asset = _asset(cache)
    with pytest.raises(EngineError) as caught:
        fetch.ensure(asset)
    # The message has to be enough to do it by hand.
    assert "curl -L -o" in str(caught.value) and asset.url in str(caught.value)


def test_a_missing_source_fails_cleanly(cache):
    asset = fetch.Asset("gone.bin", "file:///no/such/file", "0" * 64)
    with pytest.raises(EngineError) as caught:
        fetch.ensure(asset)
    assert "could not fetch" in str(caught.value)


def test_a_download_cut_short_is_retried_not_reported_as_a_new_model(cache, monkeypatch, capsys):
    """On Colab the 6.2 GB AudioSR checkpoint arrived with a wrong digest
    while the same pinned revision verified on the Mac: the connection ended
    early and the short file was taken as whole. A body shorter than its
    Content-Length is a broken transfer, and is fetched again."""
    import io

    payload = b"weights" * 1000
    asset = fetch.Asset("big.bin", "https://example.invalid/big.bin",
                        hashlib.sha256(payload).hexdigest())
    calls = []

    class Response(io.BytesIO):
        def __init__(self, body):
            super().__init__(body)
            self.headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(url):
        calls.append(url)
        return Response(payload[:100] if len(calls) == 1 else payload)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    assert fetch.ensure(asset).read_bytes() == payload
    assert len(calls) == 2
    assert "cut short" in capsys.readouterr().err
