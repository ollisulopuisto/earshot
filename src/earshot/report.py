"""Runs in, a table someone will actually read out.

Markdown because it goes in a pull request, an issue and a README without
conversion, and because a bench nobody reads is a bench nobody trusts.

The renderer never decides what is good. It knows only ``better`` from each
result, which is enough to mark the winner of a column and nothing more.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict

from . import __version__
from .probes import Result, Run


def aggregate(runs: list[Run]) -> list[Run]:
    """Merge runs of the same engine and damage across excerpts.

    The mean is what the table shows; the spread is what stops it lying. A
    single excerpt is passed through unchanged, so the one-file case reads
    exactly as before.
    """
    from statistics import fmean, pstdev

    grouped: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        grouped.setdefault((run.engine, run.damage), []).append(run)

    merged: list[Run] = []
    for (engine, damage), group in grouped.items():
        live = [r for r in group if not r.skipped and not r.failed]
        if len(group) == 1 or not live:
            merged.append(group[0])
            continue
        keys: list[tuple[str, str]] = []
        for run in live:
            for result in run.results:
                if (result.probe, result.metric) not in keys:
                    keys.append((result.probe, result.metric))
        out = Run(engine=engine, damage=damage,
                  material=f"{len(live)} excerpts")
        for probe, metric in keys:
            values = [r.value(probe, metric) for r in live]
            values = [v for v in values if v is not None and v == v]
            if not values:
                continue
            sample = next(
                x for r in live for x in r.results
                if x.probe == probe and x.metric == metric
            )
            out.results.append(
                Result(probe, metric, fmean(values), sample.unit, sample.better,
                       sample.detail, pstdev(values) if len(values) > 1 else 0.0,
                       len(values))
            )
        merged.append(out)
    return merged


def _cell(value: float, unit: str) -> str:
    if value != value:  # NaN
        return "—"
    if unit == "x":
        return f"{value:.1f}×"
    if unit == "r":
        return f"{value:+.2f}"
    if unit == "":
        return f"{value:.2f}"
    return f"{value:+.2f}" if unit == "dB" else f"{value:.2f} {unit}"


def table(runs: list[Run], probe: str) -> str:
    """One probe across every engine, engines as rows."""
    live = [r for r in runs if not r.skipped and not r.failed]
    metrics: list[tuple[str, str, str]] = []
    for run in live:
        for result in run.results:
            if result.probe == probe and (result.metric, result.unit, result.better) not in metrics:
                metrics.append((result.metric, result.unit, result.better))
    if not metrics:
        return ""

    best: dict[str, float] = {}
    for metric, _, better in metrics:
        values = [r.value(probe, metric) for r in live]
        values = [v for v in values if v is not None and v == v]
        if values and better:
            best[metric] = max(values) if better == "higher" else min(values)

    head = "| engine | " + " | ".join(m for m, _, _ in metrics) + " |"
    rule = "|---" * (len(metrics) + 1) + "|"
    lines = [f"### {probe}", "", head, rule]
    for run in live:
        cells = []
        for metric, unit, better in metrics:
            value = run.value(probe, metric)
            if value is None:
                cells.append("—")
                continue
            text = _cell(value, unit)
            found = next((r for r in run.results
                          if r.probe == probe and r.metric == metric), None)
            if found is not None and found.n > 1 and found.spread:
                text += f" ±{found.spread:.2f}"
            if better and metric in best and value == best[metric]:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {run.engine} | " + " | ".join(cells) + " |")
    for run in runs:
        if run.skipped:
            lines.append(f"| {run.engine} | _skipped: {run.skipped}_ |")
        elif run.failed:
            lines.append(f"| {run.engine} | _failed: {run.failed}_ |")
    return "\n".join(lines) + "\n"


def render(runs: list[Run], title: str = "") -> str:
    """Every probe present in the runs, in the order the probes define."""
    order: list[str] = []
    for run in runs:
        for result in run.results:
            if result.probe not in order:
                order.append(result.probe)
    parts = [f"## {title}", ""] if title else []
    parts += [t for t in (table(runs, probe) for probe in order) if t]
    if not order:
        parts.append("_no results — every engine skipped or failed_\n")
    return "\n".join(parts)


def legend() -> str:
    """What the columns mean, for the reader who did not write them."""
    return (
        "**Reading this table.** `gained` is the one that says an engine "
        "restored something: it is how much closer to the untouched original "
        "the output is than the damaged input was, in dB, so higher is "
        "better and zero means it changed things without recovering "
        "anything. `origin` is a correlation per band — near 1 the engine "
        "filtered what was there, near 0 it synthesised something new; "
        "neither is wrong, but a low number in a band that was undamaged "
        "means the speaker's own voice was replaced. `floor_change` should "
        "be negative and `speech_change` should be near zero: together they "
        "separate cleaning from turning down. `preservation` is how much of "
        "a non-speech signal the engine deleted.\n"
    )


def as_json(runs: list[Run], material: str = "", indent: int = 2) -> str:
    """The same results, for a machine.

    Results have to accumulate across engines, machines and months, and a
    markdown table cannot be diffed or sorted. This is what a contributor
    attaches to an issue and what CI would compare against a previous run.

    The machine is recorded because ``throughput`` is meaningless without it,
    and the earshot version because a probe changing its definition is the
    obvious way for two numbers to stop being comparable without anyone
    noticing.
    """
    return json.dumps(
        {
            "earshot": __version__,
            "material": material,
            "machine": {
                "platform": platform.platform(),
                "processor": platform.processor() or platform.machine(),
            },
            "runs": [asdict(run) for run in runs],
        },
        indent=indent,
        ensure_ascii=False,
    )


# Metrics worth carrying into a cross-run summary. Not every number: a table
# nobody can hold in their head is one nobody checks.
HEADLINE: tuple[tuple[str, str, str], ...] = (
    ("recovery", "gained", "recovered"),
    ("perceptual", "pesq_gained", "PESQ"),
    ("cleanup", "floor_change", "floor"),
    ("cleanup", "speech_change", "speech"),
    ("origin", "mid", "origin 1-4k"),
    ("throughput", "realtime", "speed"),
)


def scoreboard(files: list) -> str:
    """One table across every stored result file.

    Results accumulate: engines arrive one at a time, over months, measured
    by different people on different machines. The scoreboard is what makes
    that a comparison rather than a pile, and it is generated rather than
    edited so it cannot drift from the numbers it claims to summarise.
    """
    import json as _json
    from pathlib import Path as _Path

    rows: list[tuple[str, str, str, dict]] = []
    for name in files:
        data = _json.loads(_Path(name).read_text(encoding="utf-8"))
        machine = data.get("machine", {}).get("processor", "?")
        merged = aggregate([_run_from(r) for r in data.get("runs", [])])
        for run in merged:
            if run.damage == "sweep" or run.skipped or run.failed:
                continue
            values = {}
            for probe, metric, _ in HEADLINE:
                found = run.value(probe, metric)
                if found is not None:
                    values[(probe, metric)] = found
            rows.append((run.engine, run.damage, machine, values))

    if not rows:
        return "_no stored results_\n"

    head = "| engine | damage | " + " | ".join(l for _, _, l in HEADLINE) + " | machine |"
    lines = [head, "|---" * (len(HEADLINE) + 3) + "|"]
    for engine, damage, machine, values in sorted(rows, key=lambda r: (r[1], r[0])):
        cells = []
        for probe, metric, _ in HEADLINE:
            value = values.get((probe, metric))
            if value is None:
                cells.append("—")
            elif metric == "realtime":
                cells.append(f"{value:.0f}×")
            elif probe == "origin" or probe == "perceptual":
                cells.append(f"{value:+.2f}")
            else:
                cells.append(f"{value:+.2f} dB")
        lines.append(f"| {engine} | {damage} | " + " | ".join(cells) + f" | {machine} |")
    return "\n".join(lines) + "\n"


def _run_from(raw: dict) -> Run:
    run = Run(engine=raw["engine"], damage=raw["damage"],
              material=raw.get("material", ""), skipped=raw.get("skipped", ""),
              failed=raw.get("failed", ""))
    run.results = [Result(**r) for r in raw.get("results", [])]
    return run


# Metrics broken out per speaker. Two, not six: the spread already says an
# engine is inconsistent, and this table exists to say *on whom*. A wall of
# numbers would bury that.
PER_SPEAKER: tuple[tuple[str, str, str], ...] = (
    ("recovery", "gained", "recovered"),
    ("origin", "mid", "origin 1-4k"),
)


def per_speaker(runs: list[Run]) -> str:
    """One table per metric, engines as rows and speakers as columns.

    An engine that wins on average and loses badly on one voice is a
    different proposition from one that wins everywhere, and the mean plus a
    spread cannot tell them apart. This can: the losing column has a name.
    """
    from statistics import fmean

    live = [r for r in runs if r.speaker and not r.skipped and not r.failed]
    speakers = sorted({r.speaker for r in live})
    engines = sorted({r.engine for r in live})
    if len(speakers) < 2 or not engines:
        return ""

    out: list[str] = []
    for probe, metric, label in PER_SPEAKER:
        rows: list[str] = []
        for engine in engines:
            cells = []
            for speaker in speakers:
                values = [
                    r.value(probe, metric)
                    for r in live
                    if r.engine == engine and r.speaker == speaker
                ]
                values = [v for v in values if v is not None and v == v]
                cells.append(f"{fmean(values):+.2f}" if values else "—")
            if any(c != "—" for c in cells):
                rows.append(f"| {engine} | " + " | ".join(cells) + " |")
        if not rows:
            continue
        out += [
            f"#### {label}, per speaker",
            "",
            "| engine | " + " | ".join(speakers) + " |",
            "|---" * (len(speakers) + 1) + "|",
            *rows,
            "",
        ]
    return "\n".join(out)
