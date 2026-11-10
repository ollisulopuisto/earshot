# Audio Plugin Format Support in earshot

This document covers support for audio plugin formats (**AU**, **VST3**, **ARA2**, and **Pro Tools AAX / AudioSuite**) across two roles:

1. **Benchmarking / Evaluation (Hosting external plugins in Python)**
2. **DAW Integration (Exporting earshot repairs into DAWs like Ableton Live, Reaper, Logic Pro, and Pro Tools)**

---

## 1. Benchmarking External Plugins (Host Role)

earshot uses Spotify's [`pedalboard`](https://github.com/spotify/pedalboard) to host external plugins for head-to-head evaluation against open restoration models (such as in `docs/dxrevive.md`).

### VST3 (`vst3:`)
- **Format:** VST3 (`.vst3` bundle or folder).
- **Supported Platforms:** macOS, Linux, Windows.
- **Usage:**
  ```bash
  earshot bench --engine 'vst3:/path/to/Plugin.vst3?mix=100'
  ```
- **Requirements:** Requires the `vst3` optional dependency (`uv pip install -e ".[vst3]"`).

### Audio Unit (`au:`)
- **Format:** Audio Unit component (`.component`).
- **Supported Platforms:** macOS.
- **Usage:**
  ```bash
  earshot bench --engine 'au:/Library/Audio/Plug-Ins/Components/Plugin.component?parameter=value'
  ```
- **Engine details:** `au:` is powered by the same underlying `pedalboard` plugin loader as `vst3:`, enforcing `reset=True` to maintain the strict sample contract (no truncation or latency offset).

### ARA2 and Pro Tools AAX
- **ARA2:** ARA2 is not a standalone plugin format for sequential audio processing; it is an extension protocol between DAWs and plugins for non-realtime timeline access. It cannot be hosted directly as an audio processing engine block inside the Python bench.
- **Pro Tools (AAX / AudioSuite):** AAX is Avid's proprietary plugin architecture. It is not supported by `pedalboard` or open Python hosting libraries.
- Specifying `ara2:` or `aax:`/`protools:` in an engine spec provides a direct error message referencing this architecture and the bridge plugin roadmap.

---

## 2. DAW Plugin Integration (Client Role)

How earshot is integrated into digital audio workstations (DAWs) such as Ableton Live, Reaper, Logic Pro, and Pro Tools.

### Why a Bridge Plugin?
Everything in earshot is offline by construction:
- Engines like LavaSR, UniPASE, and Sidon resample and process wide contexts.
- The adaptive router requires STFT analysis across 0.75 s windows before detecting cutoffs.
- The core contract requires **exact sample alignment and preservation** without latency artifacts, which real-time buffers (64–512 samples) cannot maintain without unacceptable monitoring latency.
- Embedding Python into real-time DAW audio threads risks GIL contention, crashes, and packaging instability.

### Architecture: The Bridge Plugin (`plugin/earshot_bridge`)
Located in `plugin/earshot_bridge/`, the bridge plugin compiles into native plugin targets:
- **Format targets (not all host-tested):**
  - **VST3:** Wrapper target for Reaper, Ableton Live, Bitwig, Cubase and Studio One. The native bridge repair is tested on Linux; Windows packaging is not verified.
  - **AU (AUv2):** For Logic Pro, Ableton Live, Reaper on macOS.
  - **ARA2:** Not implemented. No ARA factory is advertised; this plugin cannot access a DAW-selected clip through ARA.
  - **AAX:** For Pro Tools (macOS and Windows when configured with Avid AAX SDK).
- **Building:**
  ```bash
  cmake -B plugin/earshot_bridge/build -S plugin/earshot_bridge
  cmake --build plugin/earshot_bridge/build
  ```
- **Workflow:**
  1. Insert `Earshot Bridge` as an effect. Enable Capture Selection and play the desired audio, then disable capture (maximum 60 seconds, float32 input, downmixed to mono).
  2. Trigger Restore With Earshot. A persistent background worker runs the local `earshot restore` command; no subprocess, thread creation, join, or file access runs in the audio callback. A new capture while that job owns the captured samples is refused. Private recordings live in an atomic mode-0700 temporary directory per instance, removed on destruction.
  3. The result is not inserted as a DAW clip and does not start playing when processing completes. Seek back and play the captured region again: the plugin replaces only matching host timeline samples. A host must supply a seconds timeline and playing state; without that information, or after a discontinuous capture, the plugin remains passthrough. Stopping, seeking and reset never advance an independent playback counter.
  4. The replacement is mono on each output channel. Float64 audio remains passthrough, and mismatched output sample rates or sample counts are rejected. AU/AAX wrapper builds, real DAW transport behavior and ARA integration are not verified by the Linux native tests.

### Native safety and workflow tests

The existing tests workflow builds the C++ tests with AddressSanitizer/UndefinedBehaviorSanitizer, independently of any VST3 bundle. `wav_io_test.cpp` covers odd-size PCM24 (the old decoder overran its allocation), PCM16/32, float32, stereo downmix, partial frames and truncated files. `bridge_state_test.cpp` uses a local fake CLI, waits for the background job, checks the real per-instance directory and tests immutable capture ownership, new capture while a prior result plays, delayed completion, seek/reset and missing transport. A separate local ThreadSanitizer run checks the threaded handoff. No model or restoration service is called by these tests.
