"""The command line: one command that answers the question.

    earshot bench --engine passthrough --engine vst3:/path/Thing.vst3

With no material it uses the synthetic signal, so the command works on a
machine with nothing installed. With ``--input`` it uses a real recording,
which is the only way the numbers mean anything about your voices.

``--write`` puts the audio next to the numbers. Nobody should choose an
engine from a table alone, and a table that cannot be checked by ear is a
table that will eventually be wrong in a way nobody notices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import audio as audio_io
from . import degrade, engines, material, metrics, probes, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="earshot",
        description="Measure what a speech restoration engine actually does.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="run the probes over one or more engines")
    bench.add_argument(
        "--engine", action="append", default=[], metavar="SPEC",
        help="engine to measure, repeatable; e.g. passthrough, "
             "vst3:/path/Thing.vst3?mix=100",
    )
    bench.add_argument(
        "--input", metavar="PATH",
        help="a clean recording, or a directory of them; synthetic material "
             "if omitted. A directory is sampled: see --excerpts",
    )
    bench.add_argument("--start", type=float, default=0.0,
                       help="seconds into the file (single file only)")
    bench.add_argument("--seconds", type=float, default=12.0,
                       help="length of each excerpt")
    bench.add_argument(
        "--excerpts", type=int, default=6, metavar="N",
        help="how many excerpts to take from a directory. One excerpt of one "
             "voice is an anecdote; the table reports the spread across them",
    )
    bench.add_argument(
        "--damage", action="append", default=[], metavar="NAME",
        help=f"repeatable; default is all. Known: {', '.join(d.name for d in degrade.RECIPES)}",
    )
    bench.add_argument("--out", metavar="DIR", help="write the report and audio here")
    bench.add_argument(
        "--write", action="store_true",
        help="also write the damaged and restored audio for listening",
    )
    bench.add_argument(
        "--json", action="store_true",
        help="print the results as JSON instead of a table; with --out, "
             "results.json is written either way",
    )

    listing = sub.add_parser("damages", help="list the damage recipes")
    listing.set_defaults(func=_damages)

    board = sub.add_parser(
        "scoreboard", help="one table across stored results (default results/*.json)"
    )
    board.add_argument("files", nargs="*", metavar="JSON")
    board.set_defaults(func=_scoreboard)
    bench.set_defaults(func=_bench)

    args = parser.parse_args(argv)
    return args.func(args)


def _damages(_args) -> int:
    for recipe in degrade.RECIPES:
        needs = f"  (needs {recipe.needs})" if recipe.needs else ""
        print(f"{recipe.name:18s} {recipe.describe}{needs}")
    return 0


def _scoreboard(args) -> int:
    files = args.files or sorted(str(p) for p in Path("results").glob("*.json"))
    if not files:
        print("no results found; run `earshot bench --out results/`", file=sys.stderr)
        return 1
    print(report.scoreboard(files))
    return 0


def _bench(args) -> int:
    specs = args.engine or ["passthrough"]
    pieces: list[material.Excerpt] = []
    if args.input and Path(args.input).is_dir():
        pieces = material.excerpts(args.input, args.seconds, args.excerpts)
        if not pieces:
            print(f"error: no usable material under {args.input}", file=sys.stderr)
            return 2
        speakers = sorted({p.speaker for p in pieces})
        source = (f"{len(pieces)} excerpts of {args.seconds:g} s from "
                  f"{Path(args.input).name}, {len(speakers)} speakers "
                  f"({', '.join(speakers)})")
    elif args.input:
        audio, rate = audio_io.read(args.input, args.seconds, args.start)
        pieces = [material.Excerpt(Path(args.input), args.start, args.seconds,
                                   audio, rate)]
        source = (f"{Path(args.input).name} "
                  f"({args.start:g}–{args.start + args.seconds:g} s)")
    else:
        rate = 48000
        pieces = [material.Excerpt(Path("synthetic"), 0.0, args.seconds,
                                   probes.default_material(rate, args.seconds), rate)]
        source = "synthetic material (no recording given)"

    wanted = args.damage or [d.name for d in degrade.RECIPES]
    recipes = [degrade.by_name(name) for name in wanted]

    loaded = []
    for spec in specs:
        try:
            loaded.append((spec, engines.load(spec)))
        except engines.EngineError as exc:
            print(f"error: {spec}: {exc}", file=sys.stderr)
            return 2

    out_dir = Path(args.out) if args.out else None
    sections: list[str] = []
    every: list[probes.Run] = []
    for recipe in recipes:
        runs: list[probes.Run] = []
        for piece in pieces:
            for _, item in loaded:
                run = probes.run_all(item, piece.audio, piece.rate, recipe)
                run.material = piece.label
                run.speaker = piece.speaker
                runs.append(run)
        every += runs
        sections.append(
            report.render(report.aggregate(runs), f"{recipe.name} — {recipe.describe}")
        )
        sections.append(report.per_speaker(runs))
        if args.write and out_dir:
            first = pieces[0]
            _write_audio(out_dir / recipe.name, first.audio, first.rate, recipe, loaded)

    sweep = [probes.preservation(item, pieces[0].rate) for _, item in loaded]
    every += sweep
    sections.append(
        report.render(sweep, "non-speech — a logarithmic sweep, not a voice")
    )

    missing = ("" if metrics.perceptual_available() else
               "\n_PESQ and STOI not measured: `pip install earshot[perceptual]`._\n")
    text = "\n".join(
        ["# earshot bench", "", f"Material: {source}", missing, "",
         report.legend(), ""] + sections
    )
    machine_readable = report.as_json(every, source)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(text, encoding="utf-8")
        (out_dir / "results.json").write_text(machine_readable, encoding="utf-8")
        print(f"wrote {out_dir / 'report.md'} and results.json")
    elif args.json:
        print(machine_readable)
    else:
        print(text)
    return 0


def _write_audio(directory: Path, clean, rate: int, recipe, loaded) -> None:
    """The damaged input and every engine's output, for listening.

    The clean original goes in too. Comparing a restoration against the
    damage flatters it; comparing against what it was trying to get back
    to does not.
    """
    if recipe.needs and not probes._have(recipe.needs):
        return
    broken = recipe.apply(clean, rate)
    audio_io.write(directory / "00-clean.wav", clean, rate)
    audio_io.write(directory / "01-damaged.wav", broken, rate)
    for index, (_, item) in enumerate(loaded, start=2):
        try:
            restored = item.engine.process(broken, rate)
        except Exception:
            continue
        audio_io.write(directory / f"{index:02d}-{item.engine.name}.wav", restored, rate)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
