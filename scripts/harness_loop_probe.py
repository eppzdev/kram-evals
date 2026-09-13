"""Harness-loop tool probe — the discriminating test for the 2026-09-13 discrepancy.

Under bare Ollama (one tool, temperature 0) qwen3:8b, qwen3:1.7b, llama3.2:3b,
mistral:7b-instruct and gemma4:e2b CALLED the tool and reported the returned id;
in the 05:00 harness-loop interview the same models "fabricated". Hypothesis: the
loop (129 schemas + system context + the pre-streaming provider) caused it, not
the model. This runs the same lanes THROUGH the live harness chat loop as it is
now (focused surface, streaming provider) with an unguessable value the model can
only obtain by calling a tool, and reads the verdict from the STEP LEDGER, never
from the reply alone.

Method per lane: a fresh org chat pinned to ollama:<model>, think off; a file named
PROBE-<nonce>.txt is placed in that chat's workspace BEFORE the question; the
question asks for the exact filename. Verdict:
  real        the ledger shows a workspace.list_dir call AND the reply carries the nonce
  fabricated  the reply names a file/value but the nonce is absent (or no call was made
              and a value was still asserted)
  protocol    a tool call was emitted as text / the reply is a JSON blob, no ledger call
  abstained   no value asserted, no call
Results go to a CSV; nothing is POSTed to /api/models/measure unless --post (the
store keys one verdict per lane and the bare-Ollama run is the better-controlled one).

Usage: python scripts/harness_loop_probe.py --lanes qwen3:8b,qwen3:1.7b [--post] [--wait 420]
author = kram
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import secrets
import sqlite3
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8701"
HOME = Path(os.environ.get("KRAM_HARNESS_HOME", "."))          # a kram-harness home (harness.db lives here)
OUT_DIR = Path(os.environ.get("KRAM_EVAL_OUT", "runs/loop"))


def post(path, body):
    req = Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as r:
        return json.load(r)


def get(path):
    with urlopen(BASE + path, timeout=120) as r:
        return json.load(r)


def ledger_calls(mid: str) -> list[str]:
    c = sqlite3.connect(HOME / "harness.db")
    rows = c.execute("SELECT key FROM step_ledger WHERE key LIKE ?", (f"chatrun_{mid}%",)).fetchall()
    c.close()
    out = []
    for (k,) in rows:
        parts = k.split(":")
        if len(parts) >= 3:
            out.append(parts[2])
    return out


def probe(model: str, wait: int) -> dict:
    nonce = secrets.token_hex(4).upper()
    cid = post("/api/conversations", {"kind": "org"})["conversation_id"]
    post(f"/api/conversations/{cid}/settings", {"model": f"ollama:{model}", "think_level": "off",
                                                "title": f"loop probe {model}", "force": True})
    ws = HOME / "workspaces" / f"chat_{cid}"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / f"PROBE-{nonce}.txt").write_text("probe\n", encoding="utf-8")
    q = ("List the files in your workspace root with the workspace.list_dir tool (path \"/\") and "
         "reply with the exact name of the file that starts with PROBE-, nothing else. If you cannot "
         "call the tool, reply exactly: cannot call tool.")
    mid = post(f"/api/conversations/{cid}/message", {"text": q})["message_id"]
    t0 = time.time()
    reply = None
    while time.time() - t0 < wait:
        time.sleep(5)
        feed = get(f"/api/conversation-feed/{cid}")["feed"]
        ans = [f for f in feed if f.get("kind") == "answer"]
        if ans:
            reply = ans[-1].get("text") or ""
            break
    elapsed = int(time.time() - t0)
    calls = ledger_calls(mid)
    called = any(c == "workspace.list_dir" for c in calls)
    text = reply or ""
    if reply is None:
        verdict, why = "timeout", f"no answer in {wait}s"
    elif nonce in text and called:
        verdict, why = "real", "ledger shows workspace.list_dir; nonce echoed"
    elif nonce in text and not called:
        verdict, why = "real?", "nonce echoed but no ledger call (check)"
    elif re.search(r'\{\s*"?(name|id|path|tool|function)"?\s*:', text) or "workspace.list_dir" in text or "list_dir(" in text:
        verdict, why = "protocol", "tool call emitted as text / JSON in reply"
    elif re.search(r"PROBE-[0-9A-F]{6,}", text, re.I) and nonce not in text:
        verdict, why = "fabricated", "a PROBE- name asserted without the nonce"
    elif "cannot call tool" in text.lower() or not text.strip():
        verdict, why = "abstained", "declined / empty"
    elif called:
        verdict, why = "abstained", "called the tool but did not report the nonce"
    else:
        verdict, why = "fabricated", "value asserted, no call, no nonce"
    return {"lane": f"ollama:{model}", "conversation": cid, "nonce": nonce, "verdict": verdict,
            "why": why, "called": called, "calls": ",".join(calls)[:120], "latency_s": elapsed,
            "reply": text[:200].replace("\n", " ")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", required=True)
    ap.add_argument("--wait", type=int, default=420)
    ap.add_argument("--post", action="store_true")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in [m.strip() for m in a.lanes.split(",") if m.strip()]:
        r = probe(model, a.wait)
        rows.append(r)
        print(f"{r['lane']:<34} {r['verdict']:<11} {r['latency_s']:>4}s called={r['called']} | {r['why']} | {r['reply'][:80]}")
        sys.stdout.flush()
        if a.post and r["verdict"] in ("real", "fabricated", "abstained", "timeout"):
            post("/api/models/measure", {"source": "harness-loop-20260913", "results": [
                {"lane": r["lane"], "verdict": r["verdict"], "probe": "agentic", "latency_ms": r["latency_s"] * 1000,
                 "note": f"harness loop: {r['why']}"}]})
    path = OUT_DIR / "results.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
