"""Schema ablation, widened — the full eligible agentic local set, both arms.

Runs harness_loop_probe.probe for every lane under the FOCUSED surface (24 core
schemas), then again under the FULL surface (every registered schema), each arm
to its own CSV written after every lane so a killed run still leaves data. The
tool-surface setting is restored to focused at the end (and on Ctrl+C). Verdicts
are compared per lane and written to a summary. Nothing is POSTed to
/api/models/measure — the roster stands.

Usage: python scripts/harness_loop_ablation.py --lanes-file lanes.txt [--wait 420]
author = kram
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_loop_probe import BASE, OUT_DIR, probe   # noqa: E402

FIELDS = ["lane", "conversation", "nonce", "verdict", "why", "called", "calls", "latency_s", "reply"]


def set_surface(value: str) -> None:
    req = Request(BASE + "/api/settings", json.dumps({"tool_surface": value}).encode(),
                  {"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as r:
        out = json.load(r)
    got = out.get("tool_surface") or (out.get("settings") or {}).get("tool_surface")
    print(f"tool_surface -> {value} (server says {got})")
    sys.stdout.flush()


def run_arm(name: str, lanes: list[str], wait: int, path: Path, *, resume: bool = False,
            redo: set[str] = frozenset()) -> list[dict]:
    """--resume keeps the rows already in the CSV (except lanes named in --redo) and
    appends the rest, so a killed run continues instead of restarting."""
    rows: list[dict] = []
    if resume and path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["lane"][7:] not in redo]
        for r in rows:
            r["called"] = r["called"] == "True"
            r["latency_s"] = int(r["latency_s"] or 0)
        print(f"[{name}] resuming: {len(rows)} lane(s) kept from {path.name}")
    done = {r["lane"][7:] for r in rows}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
        fh.flush()
        for i, model in enumerate(lanes, 1):
            if model in done:
                continue
            t0 = time.time()
            r = probe(model, wait)
            rows.append(r)
            w.writerow({k: r.get(k) for k in FIELDS})
            fh.flush()
            print(f"[{name} {i}/{len(lanes)}] {r['lane']:<40} {r['verdict']:<11} {r['latency_s']:>4}s "
                  f"called={r['called']} | {r['why']}")
            sys.stdout.flush()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes-file", required=True)
    ap.add_argument("--wait", type=int, default=420)
    ap.add_argument("--tag", default="full_set")
    ap.add_argument("--resume", action="store_true", help="keep rows already in the arm CSVs")
    ap.add_argument("--redo", default="", help="comma-separated lanes to re-run even when resuming")
    a = ap.parse_args()
    redo = {m.strip() for m in a.redo.split(",") if m.strip()}
    lanes = [m.strip() for m in Path(a.lanes_file).read_text().replace("\n", ",").split(",") if m.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    focused_csv = OUT_DIR / f"results_focused_{a.tag}.csv"
    full_csv = OUT_DIR / f"results_full_{a.tag}.csv"
    print(f"{len(lanes)} lanes; arms -> {focused_csv.name}, {full_csv.name}")
    started = time.time()
    try:
        set_surface("focused")
        focused = run_arm("focused", lanes, a.wait, focused_csv, resume=a.resume, redo=redo)
        set_surface("full")
        full = run_arm("full", lanes, a.wait, full_csv, resume=a.resume, redo=redo)
    finally:
        set_surface("focused")
    by_f = {r["lane"]: r for r in focused}
    by_u = {r["lane"]: r for r in full}
    changed = [(l, by_f[l]["verdict"], by_u[l]["verdict"]) for l in by_f if l in by_u
               and by_f[l]["verdict"] != by_u[l]["verdict"]]
    lines = [f"schema ablation, {len(lanes)} lanes, wait {a.wait}s, {int(time.time() - started)}s total",
             f"focused (24 schemas): {dict(Counter(r['verdict'] for r in focused))}",
             f"full   (all schemas): {dict(Counter(r['verdict'] for r in full))}",
             f"verdict changes: {len(changed)}"]
    lines += [f"  {l}: {f} -> {u}" for l, f, u in changed]
    lat_f = sorted(r["latency_s"] for r in focused if r["verdict"] == "real")
    lat_u = sorted(r["latency_s"] for r in full if r["verdict"] == "real")
    if lat_f and lat_u:
        med = lambda xs: xs[len(xs) // 2]  # noqa: E731
        lines.append(f"median latency (real): focused {med(lat_f)}s, full {med(lat_u)}s")
    summary = OUT_DIR / f"ablation_{a.tag}_summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("wrote", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
