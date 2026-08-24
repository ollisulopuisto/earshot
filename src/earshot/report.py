"""Runs in, a table someone will actually read out.

Markdown because it goes in a pull request, an issue and a README without
conversion, and because a bench nobody reads is a bench nobody trusts.

The renderer never decides what is good. It knows only ``better`` from each
result, which is enough to mark the winner of a column and nothing more.
"""

from __future__ import annotations

from .probes import Run


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
