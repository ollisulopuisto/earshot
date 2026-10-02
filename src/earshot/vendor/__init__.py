"""Third-party code, copied in and pinned.

Vendoring is a fork unless the drift is handled, so it is handled here:

* ``PINNED`` records what was copied, from where, at which commit, and the
  SHA-256 of the file as it was taken. Results in ``results/`` were measured
  with exactly this code.
* ``tests/test_vendor.py`` checks the file on disk still hashes to the
  recorded value. That catches the dangerous case — a local edit — which
  would otherwise make our numbers silently not-LavaSR any more.
* ``.github/workflows/vendor-check.yml`` fetches upstream on a schedule and
  fails if it has moved. It never updates anything: it tells a human, who
  re-vendors deliberately and re-runs the bench, because new code means new
  numbers and the old ones stop being comparable.

Model weights are not vendored. They are downloaded and checked against a
digest (``earshot.fetch``), so the same pinning applies to them without
putting 58 MB in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pinned:
    """One vendored file and where it came from."""

    path: str
    project: str
    url: str
    commit: str
    taken: str
    sha256: str
    licence: str
    # Where the file lives upstream, when that differs from ``path``.
    upstream: str = ""


PINNED: tuple[Pinned, ...] = (
    Pinned(
        path="lavasr_core.py",
        project="LavaSR-ONNX",
        url="https://github.com/Topping1/LavaSR-ONNX",
        commit="1a979b80d760f00d973b13d530fdd8da51be160b",
        taken="2026-08-24",
        sha256="f09194539d4b80b203ad95d92416f62f7f1675b03aa415603a727d24227c766f",
        licence="Apache-2.0",
    ),
    Pinned(
        path="lavasr_config.yaml",
        project="LavaSR-ONNX",
        url="https://github.com/Topping1/LavaSR-ONNX",
        commit="1a979b80d760f00d973b13d530fdd8da51be160b",
        taken="2026-08-24",
        sha256="5fd2cb08f6cb4b23eb20b6a71e8a4125d2fc6ff74575aecb8c8883fbf161830c",
        licence="Apache-2.0",
    ),
    Pinned(
        path="unipase/unipase.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="da557ea82568f6c37eeb2933e2d7b386da8b1ad039f24fb4cd9deb2e6a67e8f1",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/unipase.py",
    ),
    Pinned(
        path="unipase/wavlm/__init__.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/wavlm/__init__.py",
    ),
    Pinned(
        path="unipase/wavlm/feature_extractor_plc.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="c1ee5e5d8b7bb4883165695031bd49d29ea0a77e5b4d2dfd504a3c9cb223aa9d",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/wavlm/feature_extractor_plc.py",
    ),
    Pinned(
        path="unipase/wavlm/modules.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="f2cfb5e75f40baed7ddde3f80f7ebac0b79e9d25515e3d636126f7a17f0639ab",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/wavlm/modules.py",
    ),
    Pinned(
        path="unipase/wavlm/WavLM.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="64b34574953eb942b98babf51f2951222cfdbfd4111e135dbdb1196bd6f1c98b",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/wavlm/WavLM.py",
    ),
    Pinned(
        path="unipase/adapter/vocos/__init__.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/adapter/vocos/__init__.py",
    ),
    Pinned(
        path="unipase/adapter/vocos/adapter.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="a31996ac4923ff596f81015b2081951d121a112c892b9207104a4e3886ce4625",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/adapter/vocos/adapter.py",
    ),
    Pinned(
        path="unipase/adapter/vocos/backbone.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="3d8a5ef4ae7871998c9a5e3cf6c62920f25c778b18e9ffa98f2e61c845c199d8",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/adapter/vocos/backbone.py",
    ),
    Pinned(
        path="unipase/adapter/vocos/conv.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="4238c74d0465fe05e3c75ded5e4103635f1dec632e0f94f553f5e8a4f39e12be",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/adapter/vocos/conv.py",
    ),
    Pinned(
        path="unipase/adapter/vocos/head.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="1eeb5149fad0fdecd16251f5de77542571f044bbd48ffc0a66e484583d7f05fd",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/adapter/vocos/head.py",
    ),
    Pinned(
        path="unipase/vocoder/__init__.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/vocoder/__init__.py",
    ),
    Pinned(
        path="unipase/vocoder/vocos/__init__.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/vocoder/vocos/__init__.py",
    ),
    Pinned(
        path="unipase/vocoder/vocos/vocoder.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="dc9c7ba6a14b9bcf4d3c925516dee129c9156d1f490745ae0155693073b2f7eb",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/vocoder/vocos/vocoder.py",
    ),
    Pinned(
        path="unipase/vocoder/vocos/backbone.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="45ac4a2cceae91f416b97602aa410108122314ae143ccda0166aca48d017d069",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/vocoder/vocos/backbone.py",
    ),
    Pinned(
        path="unipase/vocoder/vocos/conv.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="50c79d1c4e46e3c0e975f6b19b9d31bb6d64ef55fc77e0a68a6f68f3a50bff3f",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/vocoder/vocos/conv.py",
    ),
    Pinned(
        path="unipase/vocoder/vocos/head.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="b171bf6b91ed9f516f51ef6a352be3aae7a68a63877bbb0a0f7fb30bd7312a82",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/vocoder/vocos/head.py",
    ),
    Pinned(
        path="unipase/postnet/__init__.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/postnet/__init__.py",
    ),
    Pinned(
        path="unipase/postnet/tfgrid_cws_res.py",
        project="unipase",
        url="https://github.com/xiaobin-rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        taken="2026-10-02",
        sha256="b268d9ddff3cf286de35dedcdad1b47dd7de0b34be52c3e9147b1c4bff455203",
        licence="MIT; parts Apache-2.0 (see LICENSE-unipase-*)",
        upstream="models/postnet/tfgrid_cws_res.py",
    ),
)
