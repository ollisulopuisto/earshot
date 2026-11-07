## Bolt journal — earshot performance work

Format: `## YYYY-MM-DD - [Title]` / `**Learning:**` / `**Action:**`

## 2026-11-07 - run_all re-filters `restored` 8x when 4x suffice (target selected)
**Learning:** In `probes.run_all`, the same four BANDS filters are applied to
`restored` once in `origin` (via `metrics.band_correlation`) and again in
per-band `stability` (via `signal.sosfilt`). `sosfilt` is ~6.8 ms per call on
10 s at 48 kHz vs ~0.33 ms for `butter` design (20:1), so filtering is ~74% of
run_all measured time and `restored` alone is sosfilt'ed 8x. Filtering
`restored` once per band and reusing it for both origin and stability cuts 4
sosfilt calls (~27 ms, ~11% of run_all) per call with identical numbers (same
math on identical inputs; band_correlation/repeatability functions untouched).
**Action:** Prefilter broken/restored/repeat once per band; origin computes
correlation on the prefiltered signals (replicating band_correlation's NaN
guard), stability calls metrics.repeatability on the prefiltered signals.
