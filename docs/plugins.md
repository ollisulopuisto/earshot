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
- **Formats supported:**
  - **VST3:** For Reaper, Ableton Live, Bitwig, Cubase, Studio One (Linux, macOS, Windows).
  - **AU (AUv2):** For Logic Pro, Ableton Live, Reaper on macOS.
  - **ARA2:** Direct timeline integration extension (in research/specification phase for DAWs supporting ARA2 like Reaper, Logic Pro, Studio One).
  - **AAX:** For Pro Tools (macOS and Windows when configured with Avid AAX SDK).
- **Building:**
  ```bash
  cmake -B plugin/earshot_bridge/build -S plugin/earshot_bridge
  cmake --build plugin/earshot_bridge/build
  ```
- **Workflow:**
  1. The user inserts `Earshot Bridge` in their DAW or ARA clip editor.
  2. The bridge passes audio transparently during playback.
  3. For offline restoration, the bridge hands the selected audio to `earshot.cli restore`, which computes the restoration sample-accurately and returns the audio to the DAW track in-place.
