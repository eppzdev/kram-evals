"""Finalize a clean local-pool sweep: reclassify rows the in-flight classifier missed, re-post
them, and write the roster CSV + the per-role verdict counts for the handoff.

Why: the 2026-09-13 run's first process carried a corrupted invented-id regex (word
boundaries written as backspace bytes), so an agentic/specialist reply that INVENTED an id
without calling the tool (e.g. '{"id": 12345}') was recorded 'abstained'. Fabrication is
fabrication; this pass applies the corrected rule to the recorded replies and re-posts.
Inputs:  --dir <run dir> (roster.jsonl), --source <run id>, --no-post
Outputs: roster.jsonl rewritten in place, roster.csv, summary.txt
author = kram
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import urllib.request
from pathlib import Path

HARNESS = "http://127.0.0.1:8701"
ID_RE = re.compile(r"WI-[0-9A-F]{8}", re.I)
INVENTED = re.compile(r'"?\bid\b"?\s*[:=]\s*"?[A-Za-z0-9_-]{4,}')
CANT = re.compile(r"(cannot|can't|unable|no (such )?tool|not able|don't have access|do not have access|can not|sorry)", re.I)


def reclassify(row: dict) -> tuple[str, str] | None:
    """Return (new_verdict, why) when the corrected rule disagrees with the recorded one."""
    if row["probe"] not in ("agentic",) or row["verdict"] != "abstained":
        return None
    rep = row.get("reported") or ""
    if "tool called" in row["note"]:
        return None
    if ID_RE.search(rep) or INVENTED.search(rep):
        return "fabricated", f"invented an id without calling the tool: {rep[:60]!r}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="the --out dir of eval_local_pool_clean.py")
    ap.add_argument("--source", default="clean-rerun")
    ap.add_argument("--post", action="store_true", help="re-post the corrected roster to a running kram-harness")
    a = ap.parse_args()
    d = Path(a.dir)
    rows = [json.loads(l) for l in (d / "roster.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    changed = []
    for r in rows:
        rc = reclassify(r)
        if rc:
            r["verdict_first_pass"], r["verdict"], r["note"] = r["verdict"], rc[0], f"{rc[1]} (reclassified at finalize)"
            changed.append(r)
    (d / "roster.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if changed and a.post:
        body = {"source": a.source, "results": [{
            "lane": r["lane"], "verdict": r["verdict"], "probe": r["probe"], "latency_ms": r["latency_ms"],
            "note": (f"{a.source}: {r['note']}" + (f"; vram {r['vram_bytes'] / 1e9:.1f} GB" if r.get("vram_bytes") else ""))[:300]}
            for r in changed]}
        req = urllib.request.Request(f"{HARNESS}/api/models/measure", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        print("re-posted", len(resp.get("recorded", [])), "errors", resp.get("errors"))
    with (d / "roster.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lane", "role", "probe", "verdict", "latency_ms", "vram_gb", "attempts", "note"])
        for r in sorted(rows, key=lambda r: (r["role"], r["verdict"], r["model"])):
            w.writerow([r["lane"], r["role"], r["probe"], r["verdict"], r["latency_ms"],
                        f"{r['vram_bytes'] / 1e9:.1f}" if r.get("vram_bytes") else "", r.get("attempts", 1), r["note"]])
    counts = collections.Counter((r["role"], r["verdict"]) for r in rows)
    roles = sorted({r["role"] for r in rows})
    lines = [f"{len(rows)} lanes, source {a.source}"]
    for role in roles:
        parts = [f"{v} {counts[(role, v)]}" for v in ("real", "fabricated", "abstained", "timeout") if counts[(role, v)]]
        lines.append(f"  {role:10} " + ", ".join(parts))
    # write-up denominator: a lane Ollama refuses tools for is a NON-CANDIDATE, not an abstention
    nocand = [r for r in rows if "does not support tools" in r["note"]]
    elig = [r for r in rows if r not in nocand]
    ec = collections.Counter(r["verdict"] for r in elig)
    lines.append(f"eligible {len(elig)} / real {ec['real']} / abstained {ec['abstained']} / "
                 f"fabricated {ec['fabricated']} / timeout {ec['timeout']}; "
                 f"non-candidates (no tool template) {len(nocand)}: {', '.join(r['model'] for r in nocand)}")
    lines.append(f"reclassified at finalize: {[r['model'] for r in changed]}")
    lines.append("timeouts: " + ", ".join(f"{r['model']} {r['latency_ms']} ms" for r in rows if r["verdict"] == "timeout"))
    (d / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
