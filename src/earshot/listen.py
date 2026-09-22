"""A listening page: every take of a comparison playing in sync.

    earshot listen out/kuuntelu

Numbers here disagree with each other on purpose, and the only thing that
settles a disagreement between them is an ear. An ear is easily fooled in
two ways this page exists to prevent:

* **Louder sounds better.** Every take is loudness-matched to the first one
  in its folder, measured on the loud part of the speech, and the match can
  be switched off to hear what the engine did to the level.
* **Restarting is not comparing.** A timbre difference is gone from memory
  in the second it takes to reload a player. All takes of a comparison are
  decoded up front and started together, and switching only moves a gain,
  so the switch lands mid-syllable.

A folder of WAV files is a comparison; the first file alphabetically is the
reference the others are matched to. An optional ``about.json`` beside them
names the comparison, says what to listen for, and labels the takes:

    {"title": "...", "listen_for": "...",
     "takes": {"02-dehum(50).wav": {"label": "...", "note": "..."}}}

The page is a single HTML file next to the folders and loads the audio by
relative path, so the same directory works opened locally and published.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from . import audio as audio_io
from .degrade import _speech_rms


def _level_db(x: np.ndarray, rate: int) -> float:
    if not np.any(x):
        return -120.0
    return 20 * np.log10(_speech_rms(np.asarray(x, dtype=np.float64), rate) + 1e-12)


def manifest(directory: Path) -> list[dict]:
    comparisons = []
    for folder in sorted(p for p in directory.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.wav"))
        if len(files) < 2:
            continue
        about = {}
        if (folder / "about.json").exists():
            about = json.loads((folder / "about.json").read_text(encoding="utf-8"))
        labels = about.get("takes", {})
        takes, reference = [], None
        for path in files:
            audio, rate = audio_io.read(path)
            level = _level_db(audio, rate)
            if reference is None:
                reference = level
            info = labels.get(path.name, {})
            takes.append({
                "file": f"{folder.name}/{path.name}",
                "label": info.get("label") or path.stem.split("-", 1)[-1],
                "note": info.get("note", ""),
                # Gain to bring this take to the reference's speech level,
                # capped so a take that is mostly silence is not blown up.
                "match_db": float(np.clip(reference - level, -24.0, 24.0)),
                "peak_db": float(20 * np.log10(np.max(np.abs(audio)) + 1e-12)),
            })
        comparisons.append({
            "id": folder.name,
            "title": about.get("title", folder.name),
            "listen_for": about.get("listen_for", ""),
            "takes": takes,
        })
    return comparisons


def build(directory: Path, title: str = "Earshot-kuuntelu", intro: str = "") -> Path:
    directory = Path(directory)
    data = manifest(directory)
    page = TEMPLATE.replace("__TITLE__", html.escape(title))
    page = page.replace("__INTRO__", intro)
    page = page.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    target = directory / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


TEMPLATE = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --ground: #eaeeec;
  --surface: #ffffff;
  --sunk: #f3f6f4;
  --ink: #16201c;
  --muted: #56645e;
  --line: #cfd8d3;
  --accent: #2f6f5e;
  --accent-ink: #ffffff;
  --live: #c77d05;
  --live-soft: #fbefd9;
  --focus: #1f5fbf;
  --display: "Bricolage Grotesque", "Avenir Next", "Segoe UI", sans-serif;
  --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #101614;
    --surface: #18201d;
    --sunk: #131a17;
    --ink: #e3eae6;
    --muted: #93a39b;
    --line: #2b3632;
    --accent: #72b7a0;
    --accent-ink: #0e1512;
    --live: #eaa938;
    --live-soft: #3a2c12;
    --focus: #8ab4ff;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #101614;
  --surface: #18201d;
  --sunk: #131a17;
  --ink: #e3eae6;
  --muted: #93a39b;
  --line: #2b3632;
  --accent: #72b7a0;
  --accent-ink: #0e1512;
  --live: #eaa938;
  --live-soft: #3a2c12;
  --focus: #8ab4ff;
}
* { box-sizing: border-box; }
body {
  background: var(--ground);
  color: var(--ink);
  font: 15px/1.55 var(--body);
  padding: 0 16px;
  padding-block: 0 48px;
}
.wrap { max-width: 780px; margin: 0 auto; min-width: 0; }
html, body { overflow-x: hidden; }
header.top {
  position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5;
  background: var(--ground);
  border-bottom: 1px solid var(--line);
  padding-block: 12px;
  display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center;
}
header.top h1 {
  font: 700 22px/1.1 var(--display); margin: 0; letter-spacing: -0.01em;
  flex: 1 1 200px;
}
.switches { display: flex; flex-wrap: wrap; gap: 6px; }
.switch {
  font: 500 12px/1 var(--mono); letter-spacing: 0.04em; text-transform: uppercase;
  border: 1px solid var(--line); background: var(--surface); color: var(--muted);
  padding: 8px 10px; border-radius: 4px; cursor: pointer;
}
.switch[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.intro { color: var(--muted); max-width: 65ch; margin: 18px 0 8px; }
.intro p { margin: 0 0 10px; }
.keys { font: 12px/1.5 var(--mono); color: var(--muted); margin: 0 0 8px; }
section.cmp {
  background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
  padding: 16px; margin-top: 18px;
}
section.cmp.playing { border-color: var(--live); }
.cmp h2 { font: 600 18px/1.25 var(--display); margin: 0 0 4px; text-wrap: balance; }
.cmp .for { color: var(--muted); margin: 0 0 12px; max-width: 65ch; }
.transport { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; }
.play {
  flex: none; width: 44px; height: 44px; border-radius: 50%;
  border: none; background: var(--accent); color: var(--accent-ink);
  font: 600 16px/1 var(--mono); cursor: pointer;
}
.play[data-state="loading"] { opacity: 0.6; }
.bar {
  flex: 1; min-width: 0; height: 28px; position: relative; cursor: pointer;
  background: var(--sunk); border: 1px solid var(--line); border-radius: 4px; overflow: hidden;
}
.bar .fill { position: absolute; inset: 0 auto 0 0; width: 0; background: var(--live-soft); }
.bar .head { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--live); left: 0; }
.time { flex: none; font: 12px/1 var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; text-align: right; }
.takes { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(210px, 100%), 1fr)); gap: 8px; }
.take {
  text-align: left; cursor: pointer; font: inherit; color: inherit;
  border: 1px solid var(--line); background: var(--sunk); border-radius: 4px;
  padding: 9px 10px; display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; align-items: baseline;
}
.take .n { font: 500 12px/1 var(--mono); color: var(--muted); grid-row: span 2; padding-top: 3px; }
.take .label { font-weight: 600; overflow-wrap: anywhere; }
.take .note { grid-column: 2; font-size: 13px; color: var(--muted); }
.take .gain { grid-column: 2; font: 11px/1.4 var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.take[aria-pressed="true"] { border-color: var(--live); background: var(--live-soft); }
.take[aria-pressed="true"] .n { color: var(--live); }
.reveal { margin-top: 10px; }
button:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.err { color: var(--live); font-size: 13px; margin-top: 8px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header class="top">
    <h1>__TITLE__</h1>
    <div class="switches">
      <button class="switch" id="sw-match" aria-pressed="true">Tasoitus</button>
      <button class="switch" id="sw-blind" aria-pressed="false">Sokko</button>
      <button class="switch" id="sw-loop" aria-pressed="true">Silmukka</button>
    </div>
  </header>
  <div class="intro">__INTRO__</div>
  <p class="keys">välilyönti soita/pysäytä · 1–9 vaihda ottoa · ←/→ 2 s taakse/eteen</p>
  <div id="list"></div>
</div>

<script>
const DATA = __DATA__;
const state = { match: true, blind: false, loop: true, active: null };
let ctx = null;

function store(key, value) { try { localStorage.setItem("earshot." + key, value); } catch (e) {} }
function recall(key) { try { return localStorage.getItem("earshot." + key); } catch (e) { return null; } }

function shuffle(n, seed) {
  const order = [...Array(n).keys()];
  let s = seed;
  for (let i = n - 1; i > 0; i--) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    const j = s % (i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

function fmt(t) { const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0"); return m + ":" + s; }

class Comparison {
  constructor(cmp, index) {
    this.cmp = cmp; this.index = index; this.buffers = null; this.sources = [];
    this.gains = []; this.selected = 0; this.offset = 0; this.startedAt = 0;
    this.playing = false; this.revealed = false;
    // Blind order: the reference stays first, the rest are shuffled.
    const rest = shuffle(cmp.takes.length - 1, 7919 * (index + 3));
    this.order = [0, ...rest.map(i => i + 1)];
    this.el = document.createElement("section");
    this.el.className = "cmp";
    this.el.id = cmp.id.replace(/[^A-Za-z0-9._~-]/g, "-");
    this.el.innerHTML = `
      <h2></h2><p class="for"></p>
      <div class="transport">
        <button class="play" aria-label="Soita">▶</button>
        <div class="bar" role="slider" aria-label="Kohta" tabindex="0"><div class="fill"></div><div class="head"></div></div>
        <span class="time">0:00.0</span>
      </div>
      <div class="takes"></div>
      <button class="switch reveal" hidden>Paljasta nimet</button>
      <div class="err" hidden></div>`;
    this.el.querySelector("h2").textContent = cmp.title;
    const why = this.el.querySelector(".for");
    why.textContent = cmp.listen_for; why.hidden = !cmp.listen_for;
    this.el.querySelector(".play").onclick = () => this.toggle();
    this.el.querySelector(".bar").onclick = (e) => {
      const r = e.currentTarget.getBoundingClientRect();
      this.seek(((e.clientX - r.left) / r.width) * this.duration());
    };
    this.el.querySelector(".reveal").onclick = () => { this.revealed = true; this.render(); };
    this.render();
  }
  duration() { return this.buffers ? Math.min(...this.buffers.map(b => b.duration)) : 0; }
  visibleOrder() { return state.blind && !this.revealed ? this.order : this.cmp.takes.map((_, i) => i); }
  render() {
    const box = this.el.querySelector(".takes");
    box.innerHTML = "";
    const blind = state.blind && !this.revealed;
    this.visibleOrder().forEach((takeIndex, pos) => {
      const t = this.cmp.takes[takeIndex];
      const b = document.createElement("button");
      b.className = "take";
      b.setAttribute("aria-pressed", String(takeIndex === this.selected));
      const name = blind ? (pos === 0 ? "Referenssi" : "Otto " + String.fromCharCode(64 + pos)) : t.label;
      const note = blind ? "" : t.note;
      const gain = state.match ? `tasoitus ${t.match_db >= 0 ? "+" : ""}${t.match_db.toFixed(1)} dB` : `huippu ${t.peak_db.toFixed(1)} dBFS`;
      b.innerHTML = `<span class="n">${pos + 1}</span><span class="label"></span><span class="note"></span><span class="gain"></span>`;
      b.querySelector(".label").textContent = name;
      b.querySelector(".note").textContent = note; b.querySelector(".note").hidden = !note;
      b.querySelector(".gain").textContent = gain;
      b.onclick = () => { this.select(takeIndex); if (!this.playing) this.play(); };
      box.appendChild(b);
    });
    this.el.querySelector(".reveal").hidden = !(state.blind && !this.revealed);
  }
  async load() {
    if (this.buffers) return;
    const btn = this.el.querySelector(".play");
    btn.dataset.state = "loading"; btn.textContent = "…";
    try {
      this.buffers = await Promise.all(this.cmp.takes.map(async t => {
        const res = await fetch(t.file);
        if (!res.ok) throw new Error(t.file + ": " + res.status);
        return await ctx.decodeAudioData(await res.arrayBuffer());
      }));
    } catch (e) {
      const err = this.el.querySelector(".err");
      err.hidden = false; err.textContent = "Äänen lataus epäonnistui: " + e.message;
      throw e;
    } finally { btn.dataset.state = ""; btn.textContent = "▶"; }
  }
  level(i) {
    const t = this.cmp.takes[i];
    const db = state.match ? t.match_db : 0;
    return i === this.selected ? Math.pow(10, db / 20) : 0;
  }
  async play() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") await ctx.resume();
    if (state.active && state.active !== this) state.active.stop();
    await this.load();
    this.stopSources();
    const when = ctx.currentTime + 0.05;
    this.sources = this.buffers.map((buf, i) => {
      const src = ctx.createBufferSource(); src.buffer = buf;
      src.loop = state.loop; src.loopEnd = this.duration();
      const g = ctx.createGain(); g.gain.value = this.level(i);
      src.connect(g).connect(ctx.destination);
      src.start(when, this.offset % this.duration());
      this.gains[i] = g;
      return src;
    });
    this.sources[0].onended = () => { if (this.playing && !state.loop) { this.offset = 0; this.stop(); } };
    this.startedAt = when - this.offset; this.playing = true; state.active = this;
    this.el.classList.add("playing"); this.el.querySelector(".play").textContent = "■";
    this.tick();
  }
  position() {
    if (!this.playing) return this.offset;
    const d = this.duration(), t = ctx.currentTime - this.startedAt;
    return state.loop ? ((t % d) + d) % d : Math.min(t, d);
  }
  stopSources() { this.sources.forEach(s => { try { s.onended = null; s.stop(); } catch (e) {} }); this.sources = []; }
  stop() {
    if (this.playing) this.offset = this.position();
    this.stopSources(); this.playing = false;
    this.el.classList.remove("playing"); this.el.querySelector(".play").textContent = "▶";
    if (state.active === this) state.active = null;
  }
  toggle() { this.playing ? this.stop() : this.play(); }
  seek(t) { this.offset = Math.max(0, Math.min(t, this.duration() - 0.01)); if (this.playing) this.play(); else this.draw(); }
  select(i) {
    this.selected = i;
    if (this.playing) {
      const now = ctx.currentTime;
      this.gains.forEach((g, k) => { g.gain.cancelScheduledValues(now); g.gain.setTargetAtTime(this.level(k), now, 0.004); });
    }
    this.render();
  }
  draw() {
    const d = this.duration() || 1, p = this.position();
    this.el.querySelector(".fill").style.width = (100 * p / d) + "%";
    this.el.querySelector(".head").style.left = (100 * p / d) + "%";
    this.el.querySelector(".time").textContent = fmt(p) + " / " + fmt(this.duration());
  }
  tick() { this.draw(); if (this.playing) requestAnimationFrame(() => this.tick()); }
}

const views = DATA.map((c, i) => new Comparison(c, i));
views.forEach(v => document.getElementById("list").appendChild(v.el));

function bindSwitch(id, key) {
  const el = document.getElementById(id);
  const saved = recall(key);
  if (saved !== null) state[key] = saved === "1";
  el.setAttribute("aria-pressed", String(state[key]));
  el.onclick = () => {
    state[key] = !state[key]; store(key, state[key] ? "1" : "0");
    el.setAttribute("aria-pressed", String(state[key]));
    views.forEach(v => { v.render(); if (v.playing && (key === "loop")) v.play(); if (v.playing && key === "match") v.select(v.selected); });
  };
}
bindSwitch("sw-match", "match"); bindSwitch("sw-blind", "blind"); bindSwitch("sw-loop", "loop");
views.forEach(v => v.render());

document.addEventListener("keydown", (e) => {
  if (e.target.closest && e.target.closest("input, textarea")) return;
  const v = state.active || views[0];
  if (!v) return;
  if (e.code === "Space") { e.preventDefault(); v.toggle(); }
  else if (/^[1-9]$/.test(e.key)) {
    const order = v.visibleOrder(); const k = Number(e.key) - 1;
    if (k < order.length) v.select(order[k]);
  }
  else if (e.key === "ArrowLeft") v.seek(v.position() - 2);
  else if (e.key === "ArrowRight") v.seek(v.position() + 2);
});
</script>
"""
