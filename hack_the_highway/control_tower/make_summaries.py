"""Write EVENT_SUMMARIES.md: one readable page per run with, for every event, what the system did and why.

The content is generated from runs/<run>/traces/E-xx.json (nothing is typed by hand), so it always
matches the action log.  Usage:  python control_tower/make_summaries.py [run_dir]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = {"supplier_delay": "Supplier delay", "plant_outage": "Plant outage", "quality_hold": "Quality hold",
         "noise": "Noise", "unconfirmed": "Unconfirmed report"}


def path_text(gp):
    parts = []
    for layer in gp["layers"]:
        if "label" in layer:
            ids = layer.get("ids", [])
            parts.append(f"**{layer['label']}** ({len(ids)})" if len(ids) != 1 else f"**{layer['label']}** `{ids[0]}`")
        else:
            parts.append(f"-{layer['edge']}->")
    return " ".join(parts)


def main(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    traces = [json.loads(p.read_text()) for p in sorted((run_dir / "traces").glob("E-*.json"))]
    out = ["# Event summaries - what the control tower did, and why",
           "",
           f"Run `{run_dir.name}`, model `{meta.get('model')}` ({meta.get('provider')}). "
           "Generated from the per-event traces by `control_tower/make_summaries.py`; the full machine-readable "
           "record of each event is in `runs/run1/traces/E-xx.json`, and every tool call is in `runs/run1/action_log.jsonl`.",
           "",
           "| Event | Source | Category | Decision | Write actions | Failed calls |",
           "|---|---|---|---|---|---|"]
    for t in traces:
        kinds = [LABEL.get(k, k) for k in t["category"]]
        writes = [a for a in t["actions"] if a.get("write")]
        failed = sum(1 for a in t["audit_trail"] if not a.get("ok", True)) if t.get("audit_trail") else 0
        if set(t["category"]) <= {"noise"}:
            decision = "No action (not operational)"
        elif set(t["category"]) <= {"unconfirmed"}:
            decision = "Escalated to a human, no change"
        else:
            decision = "Acted per handbook"
        out.append(f"| {t['event']['event_id']} | {t['event'].get('source', '')} | {' + '.join(kinds)} | {decision} | "
                   f"{len(writes)} | {failed} |")
    for t in traces:
        ev = t["event"]
        out += ["", "---", "", f"## {ev['event_id']} - {' + '.join(LABEL.get(k, k) for k in t['category'])}", "",
                f"> {ev['text'].strip()}", "",
                f"*Received {ev.get('received_at', '')} via {ev.get('source', '')}. Understood by "
                f"`{t['understanding'].get('model')}`; rule-based cross-check "
                f"{'agrees' if t['understanding'].get('agree_on_kinds') else 'DISAGREES'}.*", "",
                f"**Summary.** {t['impact']['headline']}", ""]
        out.append("**Why.**")
        for inc in t["incidents"]:
            bits = [f"{inc['id']} ({LABEL.get(inc['kind'], inc['kind'])}): {inc.get('llm_reason') or inc.get('summary', '')}"]
            res = inc.get("resolution") or {}
            if res:
                bits.append("Grounded: " + "; ".join(f"{k} {v}" for k, v in res.items()))
            dates = {k: v for k, v in (inc.get("dates") or {}).items() if v}
            if dates:
                bits.append("Dates: " + "; ".join(dates.values()))
            out.append("- " + " ".join(b.rstrip(".") + "." for b in bits))
        steps = [r for r in t["reasoning"] if r["agent"] != "Coordinator"][:4]
        for r in steps:
            s = r["step"] if len(r["step"]) < 260 else r["step"][:257] + "..."
            out.append(f"- *{r['agent']}:* {s}")
        if t["graph_paths"]:
            out += ["", "**Graph path(s).**"]
            for gp in t["graph_paths"]:
                out.append(f"- {gp.get('incident', '')} {gp.get('title', '')}: {path_text(gp)}")
        acts = Counter(a["tool"] for a in t["actions"])
        out += ["", "**Actions.** " + (", ".join(f"`{k}` x{v}" for k, v in sorted(acts.items())) if acts else "none") + "."]
        if t["impact"].get("changed"):
            out += ["", "**Changed:** " + " ".join(t["impact"]["changed"])]
        if t["impact"].get("unchanged"):
            out.append("")
            out.append("**Deliberately left alone:** " + " ".join(t["impact"]["unchanged"]))
        if t.get("issues"):
            out.append("")
            out.append("**Issues:** " + "; ".join(map(str, t["issues"])))
    target = ROOT / "EVENT_SUMMARIES.md"
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {target} ({len(traces)} events)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs" / "run1")
