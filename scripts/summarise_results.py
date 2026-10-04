"""Mean of every probe over the excerpts in bench result files, as a table.

    uv run python scripts/summarise_results.py results/2026-10-03-*-pp53-m1max.json

Result files hold one row per excerpt and no summary row. An earlier ad-hoc
summary kept the last row per damage and engine — one excerpt reported as if
it were the bench — and the pp53 tables of 2026-10-03 had to be corrected.
This averages every row and prints how many it averaged.
"""
import json, sys
import numpy as np
from collections import defaultdict
vals=defaultdict(list)
for f in sys.argv[1:]:
    for r in json.load(open(f))["runs"]:
        if r.get("skipped") or r.get("failed"): continue
        for x in r["results"]:
            vals[(r["damage"], r["engine"], x["probe"], x["metric"])].append(x["value"])
dmg=[]; eng=[]
for k in vals:
    if k[0] not in dmg: dmg.append(k[0])
    if k[1] not in eng: eng.append(k[1])
def m(d,e,p,q):
    v=vals.get((d,e,p,q)); return (np.mean(v), len(v)) if v else (np.nan,0)
print("| damage | engine | LSD gain (mean, n) | speaker Δ | body after |")
for d in dmg:
    if d in ("sweep","speed"): continue
    for e in eng:
        g,n=m(d,e,"recovery","gained")
        if n==0: continue
        print(f"| {d} | {e} | {g:+.2f} ({n}) | {m(d,e,'speaker','change')[0]:+.3f} | {m(d,e,'body','after')[0]:+.1f} |")
