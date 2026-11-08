"""End-to-end run: start the enterprise server, handle every event in the feed, write traces + UI.

python run.py --out ../runs/run1 [--overwrite] [--llm auto|claude|rules] [--port 8765]
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, PKG, RUNS  # noqa: E402
from kg import KnowledgeGraph  # noqa: E402
from nlu import RuleExtractor, make_extractor  # noqa: E402
from agents import Coordinator  # noqa: E402
from tools import ToolClient  # noqa: E402


def port_free(port):
    """True if the organizers' server could bind it. Their server_bind skips SO_REUSEADDR, so a port in
    TIME_WAIT from the previous run is unusable for ~60 s; probe with a plain bind, like they do."""
    import socket
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_up(url, proc, secs=20):
    for _ in range(secs * 10):
        if proc.poll() is not None:
            raise SystemExit(f"enterprise server exited: {proc.stdout.read()}")
        try:
            urllib.request.urlopen(url + "/tools", timeout=1).read()
            return
        except Exception:
            time.sleep(0.1)
    raise SystemExit("enterprise server did not start")


CONSOLE = []


def say(msg=""):
    print(msg, flush=True)
    CONSOLE.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "run1"))
    ap.add_argument("--events", default=str(DATA / "events.json"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--llm", default=None, help="auto | gemini | claude | rules")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    (out / "traces").mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "enterprise_server.py", "--data", "data", "--events", str(Path(a.events).resolve()),
           "--out", str(out), "--port", str(a.port)] + (["--overwrite"] if a.overwrite else [])
    port = next((p for p in range(a.port, a.port + 50) if port_free(p)), None)
    if port is None:
        raise SystemExit(f"no bindable port in {a.port}..{a.port + 49}")
    a.port = port
    cmd[cmd.index("--port") + 1] = str(port)
    proc = subprocess.Popen(cmd, cwd=PKG, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{a.port}"
    try:
        wait_up(url, proc)
        t0 = time.time()
        kg = KnowledgeGraph()
        kg.export_ontology(out / "ontology.json", out / "ontology.mmd")
        rules = RuleExtractor(kg)
        extractor, note = make_extractor(kg, a.llm) if a.llm else make_extractor(kg)
        say(f"knowledge graph: {kg.stats()['nodes']} nodes, {kg.stats()['edges']} edges | extractor: {extractor.name}"
            + (f" | model: {extractor.model}" if getattr(extractor, "model", None) else "")
              + (f" ({note})" if note else ""))
        coord = Coordinator(kg, extractor, rules, note)
        client = ToolClient(url)
        traces = []
        say(f"GET /events -> {len(client.events())} events; GET /tools -> {len(client.catalog)} tools")
        for ev in client.events():
            say(f"--> {ev['event_id']} arrived ({ev['source']}, {ev['received_at'][11:16]}): {' '.join(ev['text'].split())[:90]}")
            tr = coord.handle(ev, client).to_dict()
            traces.append(tr)
            (out / "traces" / f"{ev['event_id']}.json").write_text(json.dumps(tr, indent=2, default=str))
            w = sum(1 for x in tr["audit_trail"] if x["write"])
            f = sum(1 for x in tr["audit_trail"] if x.get("sent") and not x["ok"])
            say(f"{ev['event_id']}  {'+'.join(tr['category']):32s} writes={w:3d} failed={f}  {tr['summary'][:110]}")
            for iss in tr["issues"]:
                say(f"      ISSUE: {iss}")
        meta = {"model": getattr(coord.extractor, "model", None) if coord.extractor.name != "rules" else None,
                "provider": {"gemini": "Google Gemini API (generativelanguage.googleapis.com v1beta)",
                             "claude": "Anthropic Messages API"}.get(coord.extractor.name),
                "extractor": coord.extractor.name, "extractor_note": coord.note,
                "llm_usage": getattr(coord.extractor, "usage", None), "runtime_s": round(time.time() - t0, 2),
                "kg": kg.stats(), "events": len(traces)}
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
        (out / "traces.json").write_text(json.dumps(traces, default=str))
        say(f"log: {out / 'action_log.jsonl'}  ui: {out / 'control_tower.html'}")
        (out / "console.log").write_text("\n".join(CONSOLE) + "\n")
        from ui import build_ui
        build_ui(out)
        from ui import build_console
        build_console(out)

    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
