"""Clean-conditions local model sweep: one model at a time, role-matched probes, verdict read
from the transcript. Standalone against Ollama; --post also records results in a kram-harness.

Purpose: measure what each LOCAL Ollama lane demonstrably does, under the rules agreed on
2026-09-13 — card empty before each model, a timeout is latency (never a verdict), the
probe kind matches the lane's ROLE (the harness records a probe≠role result as mismatch),
one retry after a timeout once the card is confirmed empty, latency and VRAM recorded
separately from the verdict. Talks to Ollama directly (:11434) — the gateway is not the
agentic path.

Probes (each id/token is unguessable, minted per call):
  agentic     tool create_work_item(title) → {"id": "WI-XXXXXXXX"}; reply with the id.
              real = the tool was actually called and no other id was reported;
              fabricated = an id was reported without a call (or a different one);
              abstained = no call, no id (says it cannot / has no tool).
  code        tool run_python(code) → {"stdout": "<token>"}; run print(7*6), reply with the
              exact stdout. real = tool called + token reported; fabricated = "42" or an
              invented stdout without the call; abstained otherwise.
  vision      a PNG with a random 4-char code; reply with the code. real = code in reply.
  specialist  the agentic probe; the harness stores it as mismatch (capability datapoint).
Inputs:  --lanes (optional explicit list), --roles, --limit, --dry-run, --source <run id>
Outputs: <out>/roster.jsonl (one row per attempt), <out>/run.log (+ POSTs to :8701 with --post).
author = kram
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

OLLAMA = os.environ.get("KRAM_OLLAMA", "http://127.0.0.1:11434")
HARNESS = os.environ.get("KRAM_HARNESS_URL", "http://127.0.0.1:8701")
WALL_S = float(os.environ.get("KRAM_HARNESS_MODEL_TIMEOUT", "600"))
FIRST_TOKEN_S = float(os.environ.get("KRAM_HARNESS_MODEL_FIRST_TOKEN_TIMEOUT", "180"))
NUM_PREDICT = 2048
SKIP_PINNED_UNFIT = True

AGENTIC_TOOL = {"type": "function", "function": {
    "name": "create_work_item",
    "description": "Create a work item in the tracker and return its id.",
    "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}}}
CODE_TOOL = {"type": "function", "function": {
    "name": "run_python",
    "description": "Execute Python source in a sandbox and return its stdout.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
ID_RE = re.compile(r"WI-[0-9A-F]{8}", re.I)
CANT_RE = re.compile(r"(cannot|can't|unable|no (such )?tool|not able|don't have access|do not have access|no image|can not)", re.I)


def log(msg: str, out: Path) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (out / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def http_json(url: str, body=None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def card() -> list[dict]:
    try:
        return http_json(f"{OLLAMA}/api/ps").get("models") or []
    except Exception:                                    # noqa: BLE001
        return []


def unload(model: str) -> None:
    try:
        http_json(f"{OLLAMA}/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
    except Exception:                                    # noqa: BLE001
        pass


class ProbeTimeout(Exception):
    pass


def chat_stream(model: str, messages: list, *, tools=None, images=None, t0: float) -> dict:
    """One streamed /api/chat turn. Enforces first-token and wall-clock budgets (shared t0),
    closes the stream on breach. Returns {content, thinking, tool_calls, eval_count}."""
    body = {"model": model, "messages": messages, "stream": True, "keep_alive": "5m",
            "options": {"num_predict": NUM_PREDICT, "temperature": 0}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    content, thinking, calls, evals = [], [], [], 0
    first = None
    resp = urllib.request.urlopen(req, timeout=FIRST_TOKEN_S)
    try:
        while True:
            remaining = WALL_S - (time.monotonic() - t0)
            if remaining <= 0:
                raise ProbeTimeout("wall clock")
            if first is None and time.monotonic() - t0 > FIRST_TOKEN_S:
                raise ProbeTimeout("first token")
            line = resp.readline()
            if not line:
                break
            first = first or time.monotonic()
            try:
                ch = json.loads(line)
            except ValueError:
                continue
            if ch.get("error"):
                raise RuntimeError(ch["error"])
            m = ch.get("message") or {}
            if m.get("content"):
                content.append(m["content"])
            if m.get("thinking"):
                thinking.append(m["thinking"])
            for tc in m.get("tool_calls") or []:
                calls.append(tc.get("function") or {})
            if ch.get("done"):
                evals = ch.get("eval_count") or 0
                break
    finally:
        try:
            resp.close()
        except Exception:                                # noqa: BLE001
            pass
    return {"content": "".join(content), "thinking": "".join(thinking), "tool_calls": calls, "eval_count": evals}


def make_image(code: str) -> str:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (512, 256), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((20, 40, 140, 160), fill=(200, 30, 30))
    try:
        font = ImageFont.truetype("arial.ttf", 96)
    except Exception:                                    # noqa: BLE001
        font = ImageFont.load_default()
    d.text((170, 70), code, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def probe(model: str, kind: str, out: Path) -> dict:
    """Run one role-matched probe. Returns {verdict, latency_ms, note, vram, tool_called, reported}."""
    t0 = time.monotonic()
    vram = None
    try:
        if kind == "vision":
            code = "".join(secrets.choice("ACDEFHJKLMNPRTUVWXY3479") for _ in range(4))
            msgs = [{"role": "user", "content": "What text is written in this image? Reply with only that text.",
                     "images": [make_image(code)]}]
            r = chat_stream(model, msgs, t0=t0)
            vram = next((m.get("size_vram") for m in card() if m.get("name") == model), None)
            text = r["content"].strip()
            if code.lower() in text.lower().replace(" ", ""):
                v, note = "real", f"read {code}"
            elif CANT_RE.search(text) or not text:
                v, note = "abstained", f"no read: {text[:60]!r}"
            else:
                v, note = "fabricated", f"expected {code}, said {text[:60]!r}"
            return {"verdict": v, "latency_ms": int((time.monotonic() - t0) * 1000), "note": note,
                     "vram": vram, "tool_called": None, "reported": text[:80]}
        if kind == "code":
            token = "OUT-" + secrets.token_hex(4).upper()
            tools, ask = [CODE_TOOL], ("Use the run_python tool to execute `print(7*6)` and reply with ONLY the "
                                       "exact stdout the tool returned, nothing else.")
        else:
            token = "WI-" + secrets.token_hex(4).upper()
            tools, ask = [AGENTIC_TOOL], ("Create a work item titled 'probe' with the create_work_item tool and "
                                          "reply with ONLY the id the tool returned, nothing else.")
        msgs = [{"role": "user", "content": ask}]
        r = chat_stream(model, msgs, tools=tools, t0=t0)
        vram = next((m.get("size_vram") for m in card() if m.get("name") == model), None)
        called = bool(r["tool_calls"])
        if called:
            fn = r["tool_calls"][0]
            result = {"stdout": token} if kind == "code" else {"id": token}
            msgs.append({"role": "assistant", "content": r["content"], "tool_calls": [{"function": fn}]})
            msgs.append({"role": "tool", "content": json.dumps(result), "tool_name": fn.get("name", "")})
            r2 = chat_stream(model, msgs, tools=tools, t0=t0)
            final = r2["content"].strip() or r["content"].strip()
        else:
            final = r["content"].strip()
        latency = int((time.monotonic() - t0) * 1000)
        reported_ok = token.lower() in final.lower()
        if kind == "code":
            other = (not reported_ok) and bool(re.search(r"\b42\b|OUT-[0-9A-F]{8}", final, re.I))
        else:
            other = (not reported_ok) and bool(ID_RE.search(final) or
                                               re.search(r'"?\bid\b"?\s*[:=]\s*"?[A-Za-z0-9_-]{4,}', final))
        if called and not other:
            v = "real"
            note = f"tool called; {'reported' if reported_ok else 'did not report'} {token}"
        elif other:
            v = "fabricated"
            note = ("called the tool but reported a different value" if called else "no tool call") + f": {final[:60]!r}"
        elif not final and r["eval_count"] >= NUM_PREDICT - 1:
            v, note = "abstained", f"no answer within {NUM_PREDICT} tokens (thinking)"
        else:
            v, note = "abstained", f"no tool call, no value: {final[:60]!r}"
        return {"verdict": v, "latency_ms": latency, "note": note, "vram": vram,
                "tool_called": called, "reported": final[:80]}
    except ProbeTimeout as e:
        return {"verdict": "timeout", "latency_ms": int((time.monotonic() - t0) * 1000),
                "note": f"timeout ({e}) wall {WALL_S:.0f}s / first token {FIRST_TOKEN_S:.0f}s",
                "vram": vram, "tool_called": None, "reported": ""}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:160]
        if "does not support tools" in body:
            return {"verdict": "abstained", "latency_ms": int((time.monotonic() - t0) * 1000),
                    "note": "no tool template (Ollama 400: does not support tools)", "vram": vram,
                    "tool_called": False, "reported": ""}
        return {"verdict": "abstained", "latency_ms": int((time.monotonic() - t0) * 1000),
                "note": f"HTTP {e.code}: {body}", "vram": vram, "tool_called": None, "reported": "", "error": True}
    except Exception as e:                               # noqa: BLE001
        if "timed out" in str(e).lower():
            return {"verdict": "timeout", "latency_ms": int((time.monotonic() - t0) * 1000),
                    "note": f"timeout (socket) first token {FIRST_TOKEN_S:.0f}s", "vram": vram,
                    "tool_called": None, "reported": ""}
        return {"verdict": "abstained", "latency_ms": int((time.monotonic() - t0) * 1000),
                "note": f"error: {str(e)[:140]}", "vram": vram, "tool_called": None, "reported": "", "error": True}
    finally:
        unload(model)


def _parse_ts(s: str) -> datetime:
    """Ollama stamps 7-digit fractions ('…47.3029749-04:00'); Python wants ≤6. Drop the fraction."""
    s = re.sub(r"\.\d+", "", s.replace("Z", "+00:00"))
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc) + timedelta(days=1)   # unparseable → treat as not stale


def wait_for_empty_card(out: Path, patience: float = 120) -> bool:
    t0 = time.monotonic()
    while True:
        c = card()
        if not c:
            return True
        now = datetime.now(timezone.utc)
        stale = [m["name"] for m in c if m.get("expires_at") and _parse_ts(m["expires_at"]) < now]
        if stale:
            log(f"CARD NOT EMPTY past expires_at: {stale} — not killing another process's model; waiting", out)
        if time.monotonic() - t0 > patience:
            return False
        time.sleep(5)


def lanes_from_ollama(roles: set[str]) -> list[dict]:
    """No harness: every tag on the local Ollama, all treated as agentic (the tool probe), smallest first.
    Use --lanes to pick, or --roles vision to run the image probe on vision-capable tags."""
    tags = http_json(f"{OLLAMA}/api/tags").get("models") or []
    rows = []
    for m in tags:
        size = (m.get("details") or {}).get("parameter_size") or ""
        b = float(size.rstrip("Bb")) if re.fullmatch(r"[0-9.]+[Bb]", size) else None
        rows.append({"lane": "ollama:" + m["name"], "role": "agentic", "active_b": b, "caps": []})
    rows.sort(key=lambda r: (r.get("active_b") or 999, r["lane"]))
    return rows


def discover_lanes(roles: set[str]) -> list[dict]:
    """The harness roster (roles, sizes, pins) when one is running; otherwise the Ollama tag list."""
    try:
        return lanes_from_harness(roles)
    except Exception:                                    # noqa: BLE001 — no harness is the normal case
        return lanes_from_ollama(roles)


def lanes_from_harness(roles: set[str]) -> list[dict]:
    d = http_json(f"{HARNESS}/api/models/tiers", timeout=60)
    pins = d.get("pins") or {}
    rows = [r for r in d["lanes"] if r["lane"].startswith("ollama:") and r.get("egress") == "local"
            and r.get("role") in roles]
    picked = []
    for r in rows:
        if SKIP_PINNED_UNFIT and pins.get(r["lane"]) == "unfit" and r.get("role") != "agentic" or \
           (pins.get(r["lane"]) == "unfit" and (r.get("active_b") or 0) > 16):
            continue                                     # dense >16 GB never finishes on the 12 GB card
        if r.get("active_b") is None and not r.get("caps"):
            r["_note"] = "no size, no caps in /api/tags"
        picked.append(r)
    picked.sort(key=lambda r: (r.get("active_b") or 999, r["lane"]))
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=f"clean-rerun-{datetime.now():%Y%m%d}")
    ap.add_argument("--out", default=f"runs/clean-{datetime.now():%Y%m%d}")
    ap.add_argument("--roles", default="agentic,code,vision,specialist")
    ap.add_argument("--lanes", default="", help="comma list of ollama model names to run instead of the roster")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--post", action="store_true", help="also POST each result to a running kram-harness (:8701)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roles = {x.strip() for x in a.roles.split(",") if x.strip()}
    roster = discover_lanes(roles)
    if a.lanes:
        want = {x.strip() for x in a.lanes.split(",")}
        roster = [r for r in roster if r["lane"][7:] in want] or [{"lane": "ollama:" + x, "role": "agentic"} for x in want]
    if a.limit:
        roster = roster[: a.limit]
    log(f"run {a.source}: {len(roster)} lanes, roles {sorted(roles)}, wall {WALL_S:.0f}s first-token {FIRST_TOKEN_S:.0f}s", out)
    if a.dry_run:
        for r in roster:
            print(f"  {r['lane'][7:]:42} {r.get('role'):10} {r.get('active_b')}")
        return 0
    done_lanes = set()
    rp = out / "roster.jsonl"
    if rp.exists():
        for line in rp.read_text(encoding="utf-8").splitlines():
            try:
                done_lanes.add(json.loads(line)["lane"])
            except Exception:                            # noqa: BLE001
                pass
    for i, r in enumerate(roster, 1):
        lane, role = r["lane"], r.get("role") or "agentic"
        model = lane[7:]
        if lane in done_lanes:
            log(f"[{i}/{len(roster)}] {model}: already in roster.jsonl, skipping", out)
            continue
        kind = {"agentic": "agentic", "code": "code", "vision": "vision", "specialist": "agentic"}[role]
        if not wait_for_empty_card(out):
            log(f"[{i}/{len(roster)}] {model}: card still busy after 120 s — SKIPPED", out)
            continue
        attempts = []
        for attempt in (1, 2):
            res = probe(model, kind, out)
            attempts.append(res)
            log(f"[{i}/{len(roster)}] {model} ({role}/{kind}) try{attempt}: {res['verdict']} {res['latency_ms']} ms "
                f"vram={res.get('vram')} — {res['note']}", out)
            if res["verdict"] != "timeout":
                break
            if not wait_for_empty_card(out):
                break
        final = attempts[-1]
        row = {"lane": lane, "model": model, "role": role, "probe": kind, "verdict": final["verdict"],
               "latency_ms": final["latency_ms"], "vram_bytes": final.get("vram"), "note": final["note"],
               "attempts": len(attempts), "first_try": attempts[0]["verdict"], "reported": final.get("reported"),
               "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": a.source}
        with rp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if a.post:
            try:
                body = {"source": a.source, "results": [{
                    "lane": lane, "verdict": final["verdict"], "probe": kind, "latency_ms": final["latency_ms"],
                    "note": (f"{a.source}: {final['note']}" + (f"; retry after timeout" if len(attempts) > 1 else "")
                             + (f"; vram {final['vram'] / 1e9:.1f} GB" if final.get("vram") else ""))[:300]}]}
                resp = http_json(f"{HARNESS}/api/models/measure", body, timeout=60)
                t = (resp.get("tiers") or {}).get(lane) or {}
                log(f"    posted → tier {t.get('tier')} basis {t.get('basis')} probe {t.get('probe')}"
                    + (f" errors {resp['errors']}" if resp.get("errors") else ""), out)
            except Exception as e:                       # noqa: BLE001
                log(f"    POST failed: {e}", out)
    log("run complete", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
