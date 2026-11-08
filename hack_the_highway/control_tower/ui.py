"""Builds control_tower.html: a self-contained control-tower view (event feed, graph path, reasoning trace,
actions, audit trail, ontology) from a run folder. Open the file in any browser; no server needed."""
import json
from pathlib import Path

from kg import ONTOLOGY

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Wacky Wheels Control Tower</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--mute:#667085;--line:#e3e6eb;--accent:#2f5bd3;
--delay:#b45309;--outage:#b42318;--quality:#6d28d9;--noise:#667085;--unconfirmed:#0e7490;--ok:#067647;--bad:#b42318}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}
header{display:flex;gap:24px;align-items:center;padding:14px 20px;background:#111827;color:#fff;flex-wrap:wrap}
header h1{font-size:17px;margin:0;font-weight:650}header .kpi{font-size:12px;color:#cbd5e1}header .kpi b{color:#fff;font-size:15px;display:block}
nav{display:flex;gap:4px;padding:8px 20px 0;background:#111827}nav button{background:none;border:0;color:#cbd5e1;padding:8px 12px;cursor:pointer;border-radius:6px 6px 0 0;font:inherit}
nav button.on{background:var(--bg);color:var(--ink)}
main{display:grid;grid-template-columns:330px 1fr;gap:16px;padding:16px 20px;align-items:start}
@media(max-width:860px){main{grid-template-columns:1fr}}
.feed{display:flex;flex-direction:column;gap:8px;position:sticky;top:8px;max-height:calc(100vh - 140px);overflow:auto}
.ev{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;cursor:pointer}
.ev.on{border-color:var(--accent);box-shadow:0 0 0 2px #2f5bd322}.ev .top{display:flex;justify-content:space-between;font-size:12px;color:var(--mute)}
.ev .txt{margin-top:4px;display:-webkit-box;-webkit-line-clamp:2;line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.chip{display:inline-block;font-size:11px;font-weight:600;padding:1px 8px;border-radius:99px;color:#fff;margin-right:4px}
.k-supplier_delay{background:var(--delay)}.k-plant_outage{background:var(--outage)}.k-quality_hold{background:var(--quality)}
.k-noise{background:var(--noise)}.k-unconfirmed{background:var(--unconfirmed)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.card h2{font-size:14px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute)}
.msg{font-size:15px;background:#f2f4f7;border-left:3px solid var(--accent);padding:10px 12px;border-radius:6px;white-space:pre-wrap}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600}td code{font-size:11.5px}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.scroll{overflow-x:auto;max-height:420px;overflow-y:auto}
.trace li{margin:3px 0}.agent{font-weight:650;font-size:12px;padding:1px 6px;border-radius:4px;background:#eef2ff;color:#3730a3;margin-right:6px}
.agent.SupplyChainPlanner{background:#fff7ed;color:#9a3412}.agent.CustomerDealerAgent{background:#ecfdf3;color:#067647}
.agent.QualityAgent{background:#f4f3ff;color:#5b21b6}
svg text{font:12px system-ui,sans-serif}.muted{color:var(--mute)}.issue{color:var(--bad)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:860px){.grid2{grid-template-columns:1fr}}
.kv{font-size:12.5px}.kv div{padding:2px 0}.kv b{display:inline-block;min-width:110px;color:var(--mute);font-weight:600}
</style></head><body>
<header><h1>Wacky Wheels Co. - Operations Control Tower</h1><div id="kpis" style="display:flex;gap:22px;flex-wrap:wrap"></div></header>
<nav><button class="on" data-tab="events">Events</button><button data-tab="ontology">Ontology</button><button data-tab="audit">Full audit trail</button></nav>
<div id="tab-events"><main><div class="feed" id="feed"></div><div id="detail"></div></main></div>
<div id="tab-ontology" hidden><div style="padding:16px 20px"><div class="card"><h2>Ontology / knowledge graph schema</h2><div id="onto"></div></div></div></div>
<div id="tab-audit" hidden><div style="padding:16px 20px"><div class="card"><h2>Every tool call in this run</h2><div class="scroll" style="max-height:none" id="allaudit"></div></div></div></div>
<script>
const T = __TRACES__, META = __META__, ONTO = __ONTO__, ONTO_SVG = __ONTOSVG__;
const $ = s => document.querySelector(s), esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const chips = ks => ks.map(k => `<span class="chip k-${k}">${k.replace("_"," ")}</span>`).join("");
const writes = t => t.audit_trail.filter(a => a.write);
function kpis(){
  const all = T.flatMap(t => t.audit_trail), w = all.filter(a => a.write), f = all.filter(a => a.sent && !a.ok);
  const items = [["Events", T.length], ["Write actions", w.length], ["Failed calls", f.length], ["Escalations", all.filter(a=>a.tool==="escalate").length],
    ["KG", `${META.kg.nodes.toLocaleString()} nodes / ${META.kg.edges.toLocaleString()} edges`], ["Understanding", META.model ? META.model : META.extractor]];
  $("#kpis").innerHTML = items.map(([k,v]) => `<div class="kpi">${k}<b>${v}</b></div>`).join("");
}
function feed(){
  $("#feed").innerHTML = T.map((t,i) => `<div class="ev" data-i="${i}"><div class="top"><span><b>${t.event.event_id}</b> - ${esc(t.event.source)}</span>
    <span>${t.event.received_at.slice(11,16)}</span></div><div style="margin-top:4px">${chips(t.category)}<span class="muted" style="font-size:12px">${writes(t).length} writes</span></div>
    <div class="txt">${esc(t.event.text)}</div></div>`).join("");
  document.querySelectorAll(".ev").forEach(e => e.onclick = () => show(+e.dataset.i));
}
function pathSVG(p){
  const L = p.layers, nodes = L.filter(l => !l.edge), W = Math.max(720, nodes.length * 175), H = 150;
  let x = 10, out = [], prevX = null; const bw = 150;
  L.forEach(l => {
    if (l.edge) { out.push({edge:l.edge}); return; }
    const ids = l.ids || [], shown = ids.slice(0,3).map(esc).join("<br>") + (ids.length > 3 ? `<br><i>+${ids.length-3} more</i>` : "") || "<i>none</i>";
    out.push({x, label:l.label, n:ids.length, shown}); x += 175;
  });
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;min-width:600px">`, html = "";
  const boxes = out.filter(o => !o.edge); let pendingEdge = null, last = null;
  out.forEach(o => {
    if (o.edge) { pendingEdge = o.edge; return; }
    if (last) { svg += `<line x1="${last.x+bw}" y1="40" x2="${o.x}" y2="40" stroke="#98a2b3" stroke-width="1.5" marker-end="url(#a)"/>`;
      if (pendingEdge) svg += `<title>${esc(pendingEdge)}</title>`; }
    svg += `<rect x="${o.x}" y="18" width="${bw}" height="44" rx="8" fill="#eef2ff" stroke="#2f5bd3"/>
      <text x="${o.x+bw/2}" y="37" text-anchor="middle" font-weight="650">${esc(o.label.split(" ")[0])}</text>
      <text x="${o.x+bw/2}" y="53" text-anchor="middle" fill="#475467">${o.n} record${o.n===1?"":"s"}</text>`;
    html += `<foreignObject x="${o.x}" y="68" width="${bw}" height="80"><div xmlns="http://www.w3.org/1999/xhtml" style="font-size:11px;color:#475467;text-align:center">${o.shown}</div></foreignObject>`;
    last = o; pendingEdge = null;
  });
  svg += html + `<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#98a2b3"/></marker></defs></svg>`;
  const rels = L.map(l => l.edge ? `<span class="muted">-[${esc(l.edge)}]-></span>` : `<b>${esc(l.label)}</b>`).join(" ");
  return `<div style="margin:6px 0 4px;font-weight:600">${esc(p.incident)}: ${esc(p.title)}</div><div class="scroll" style="max-height:none">${svg}</div><div style="font-size:12px;margin-top:4px">${rels}</div>`;
}
function auditTable(rows, showEv){
  return `<table><tr>${showEv?"<th>Event</th>":""}<th>Time</th><th>Agent</th><th>Tool</th><th>Args</th><th>Why</th><th>Result</th></tr>` + rows.map(a => `<tr>
   ${showEv?`<td>${a.ev}</td>`:""}<td>${a.t.slice(11,23)}</td><td>${a.agent}</td><td><code>${a.tool}</code>${a.write?"":" <span class='muted'>(read)</span>"}</td>
   <td><code>${esc(JSON.stringify(a.args)).slice(0,160)}</code></td><td>${esc(a.why)}</td>
   <td class="${a.ok?"ok":"bad"}">${a.ok?"ok":"FAILED"}${a.ok?"":" "+esc(JSON.stringify(a.result||a.error))}</td></tr>`).join("") + `</table>`;
}
function show(i){
  document.querySelectorAll(".ev").forEach(e => e.classList.toggle("on", +e.dataset.i === i));
  const t = T[i], u = t.understanding, w = writes(t);
  const counts = {}; w.forEach(a => counts[a.tool] = (counts[a.tool]||0)+1);
  const inc = t.incidents.map(x => `<div class="card" style="margin:8px 0;padding:10px 12px"><div>${chips([x.kind])}<b>${x.id}</b> ${esc(x.summary)}</div>
    <div class="kv" style="margin-top:6px">${x.reason?`<div><b>Reason</b>${esc(x.reason)}</div>`:""}
    ${Object.entries(x.resolution||{}).map(([k,v])=>`<div><b>${k}</b>${esc(v)}</div>`).join("")}
    ${Object.entries(x.dates||{}).map(([k,v])=>`<div><b>${k}</b>${esc(v)}</div>`).join("")}
    ${x.shipment?`<div><b>Shipment</b>${x.shipment}: ${x.eta} -> ${x.new_eta}</div>`:""}${x.lots?`<div><b>Lots</b>${x.lots.join(", ")}</div>`:""}</div></div>`).join("");
  $("#detail").innerHTML = `
   <div class="card"><h2>${t.event.event_id} - ${esc(t.event.source)} - ${t.event.received_at}</h2><div class="msg">${esc(t.event.text)}</div>
     <div style="margin-top:10px">${chips(t.category)}</div><p style="margin:10px 0 0"><b>Summary.</b> ${esc(t.summary)}</p>
     ${t.issues.length?`<p class="issue">${t.issues.map(esc).join("<br>")}</p>`:""}</div>
   ${t.impact&&t.impact.headline?`<div class="card" id="sec-impact"><h2>What changed - and what did not</h2><p style="font-size:15px;margin:0 0 10px"><b>${esc(t.impact.headline)}</b></p>
     <div class="grid2"><div><div style="font-weight:650;color:var(--ok);margin-bottom:4px">Changed</div><ul style="margin:0;padding-left:18px">${t.impact.changed.map(x=>`<li>${esc(x)}</li>`).join("")}</ul></div>
     <div><div style="font-weight:650;color:var(--mute);margin-bottom:4px">Deliberately not changed</div><ul style="margin:0;padding-left:18px">${t.impact.unchanged.map(x=>`<li>${esc(x)}</li>`).join("")}</ul></div></div></div>`:""}
   <div class="card" id="sec-understanding"><h2>Event understanding (${esc(u.extractor)}${u.model?" - model "+esc(u.model):""}${u.tokens?` - ${u.tokens.input_tokens} in / ${u.tokens.output_tokens} out tokens`:""})</h2>${u.note?`<p class="muted">${esc(u.note)}</p>`:""}
     ${u.agree_on_kinds!==undefined?`<p>LLM vs rule cross-check: <b class="${u.agree_on_kinds?"ok":"bad"}">${u.agree_on_kinds?"agree":"disagree"}</b></p>`:""}${inc}</div>
   <div class="card" id="sec-paths"><h2>Graph path${t.graph_paths.length>1?"s (one per incident)":""}</h2>${t.graph_paths.length?t.graph_paths.map(pathSVG).join("<hr style='border:0;border-top:1px solid var(--line)'>"):"<p class='muted'>No traversal: nothing operational to trace (see reasoning).</p>"}</div>
   <div class="card" id="sec-trace"><h2>Reasoning trace</h2><ol class="trace">${t.reasoning.map(r => `<li><span class="agent ${r.agent}">${r.agent}</span>${esc(r.step)}</li>`).join("")}</ol></div>
   <div class="card" id="sec-actions"><h2>Actions taken (${w.length})</h2><p>${Object.entries(counts).map(([k,v])=>`<code>${k}</code> x${v}`).join(" - ") || "No write actions."}</p>
     <div class="scroll">${auditTable(t.audit_trail.filter(a=>a.write||a.tool==="escalate"))}</div></div>
   <div class="card" id="sec-audit"><h2>Audit trail - every call incl. reads (${t.audit_trail.length})</h2><div class="scroll">${auditTable(t.audit_trail)}</div></div>`;
}
function onto(){
  $("#onto").innerHTML = `<div class="scroll" style="max-height:none">${ONTO_SVG}</div><p class="muted" style="margin:6px 0 0">Arrows read <i>From -RELATION-&gt; To</i>; hover a line for its full name. Not drawn: <code>Alias -REFERS_TO-&gt; any class</code> (${(META.kg.by_type.Alias||0)} free-text name variants used to resolve nicknames).</p><table style="margin-top:10px"><tr><th>Class</th><th>Meaning</th></tr>${Object.entries(ONTO.classes).map(([k,v])=>`<tr><td><b>${k}</b></td><td>${esc(v)}</td></tr>`).join("")}</table>
   <table style="margin-top:10px"><tr><th>From</th><th>Relation</th><th>To</th></tr>${ONTO.relations.map(r=>`<tr><td>${r[0]}</td><td><code>${r[1]}</code></td><td>${r[2]}</td></tr>`).join("")}</table>`;
}
document.querySelectorAll("nav button").forEach(b => b.onclick = () => { document.querySelectorAll("nav button").forEach(x => x.classList.toggle("on", x===b));
  ["events","ontology","audit"].forEach(t => $("#tab-"+t).hidden = t !== b.dataset.tab); });
kpis(); feed(); onto();
const [h, sec] = decodeURIComponent(location.hash.slice(1)).split(":"), ei = T.findIndex(t => t.event.event_id === h);
show(ei >= 0 ? ei : 0);
if (sec) {   // focus mode for demo frames: event header + the requested section only
  document.querySelectorAll("#detail > .card").forEach((c, k) => { if (k > 0 && c.id !== "sec-" + sec) c.hidden = true; });
  document.querySelectorAll("#sec-actions .scroll, #sec-audit .scroll").forEach(s => s.style.maxHeight = "none");
}
if (h === "ontology" || h === "audit") document.querySelector(`nav button[data-tab="${h}"]`).click();
$("#allaudit").innerHTML = auditTable(T.flatMap(t => t.audit_trail.map(a => ({...a, ev:t.event.event_id}))), true);
</script></body></html>"""


# ---------------------------------------------------------------- ontology diagram
# Hand-placed layered layout, left to right = supply -> parts / lots -> production -> orders -> dealers / customers.
BOX_W, BOX_H = 156, 52
ONTO_POS = {  # top-left corner of each class box
    "Shipment": (30, 70), "Supplier": (30, 250),
    "Model": (300, 70), "Part": (300, 250), "Lot": (300, 430),
    "Plant": (570, 70), "ProductionOrder": (570, 250),
    "CustomerOrder": (840, 130), "Vehicle": (840, 400),
    "Customer": (1110, 130), "Dealer": (1110, 400),
}
ONTO_ARC = {("Shipment", "Plant"): -58}   # routed as an arc over the Model box (control-point lift in px)


def _clip(cx, cy, tx, ty, pad=4):
    """Point where the segment from box centre (cx,cy) towards (tx,ty) leaves the box (plus padding)."""
    dx, dy = tx - cx, ty - cy
    hw, hh = BOX_W / 2 + pad, BOX_H / 2 + pad
    s = min(hw / abs(dx) if dx else float("inf"), hh / abs(dy) if dy else float("inf"))
    return cx + dx * s, cy + dy * s


def _seg_cross(p1, p2, p3, p4):
    """Intersection point of segments p1p2 and p3p4 (None if they do not cross away from endpoints)."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / d
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1)) if 0.02 < t < 0.98 and 0.02 < u < 0.98 else None


def ontology_svg(by_type):
    """SVG of the ontology: arrows end at box edges, labels sit on a white backing at a position that overlaps
    no box, no other label and no edge crossing (greedy search along each edge)."""
    import html as _h
    rels = [(a, r, b) for a, r, b in ONTOLOGY["relations"] if b != "*" and a in ONTO_POS and b in ONTO_POS]
    centre = {c: (x + BOX_W / 2, y + BOX_H / 2) for c, (x, y) in ONTO_POS.items()}
    edges = []
    for a, r, b in rels:
        (ax, ay), (bx, by) = centre[a], centre[b]
        lift = ONTO_ARC.get((a, b))
        if lift is None:
            p, q = _clip(ax, ay, bx, by), _clip(bx, by, ax, ay)
            edges.append(dict(a=a, r=r, b=b, p=p, q=q, ctrl=None))
        else:  # leave from the top of a, arrive at the top of b
            p, q = (ax, ay - BOX_H / 2 - 4), (bx, by - BOX_H / 2 - 4)
            edges.append(dict(a=a, r=r, b=b, p=p, q=q, ctrl=((ax + bx) / 2, min(ay, by) + 2 * lift)))
    straight = [e for e in edges if e["ctrl"] is None]
    crossings = [c for i, e in enumerate(straight) for f in straight[i + 1:]
                 if (c := _seg_cross(e["p"], e["q"], f["p"], f["q"]))]

    def at(e, t):
        (x1, y1), (x2, y2) = e["p"], e["q"]
        if e["ctrl"] is None:
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        cx, cy = e["ctrl"]
        return ((1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t * t * x2, (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t * t * y2)

    def overlaps(r1, r2, m=3):
        return not (r1[2] + m <= r2[0] or r2[2] + m <= r1[0] or r1[3] + m <= r2[1] or r2[3] + m <= r1[1])

    boxes = [(x, y, x + BOX_W, y + BOX_H) for x, y in ONTO_POS.values()]
    placed, labels = [], []
    for e in edges:
        w, h = 7.0 * len(e["r"]) + 14, 20
        best = None
        for t in (0.5, 0.42, 0.58, 0.34, 0.66, 0.27, 0.73, 0.2, 0.8):
            x, y = at(e, t)
            rect = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            hit_box = any(overlaps(rect, b) for b in boxes)
            hit_lab = any(overlaps(rect, l) for l in placed)
            hit_x = any(rect[0] - 8 <= cx <= rect[2] + 8 and rect[1] - 8 <= cy <= rect[3] + 8 for cx, cy in crossings)
            if not (hit_box or hit_lab or hit_x):
                best = (x, y, rect)
                break
        if best is None:  # fall back to the midpoint (never observed with this layout)
            x, y = at(e, 0.5)
            best = (x, y, (x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        placed.append(best[2])
        labels.append((e, best))

    W = max(x for x, _ in ONTO_POS.values()) + BOX_W + 30
    H = max(y for _, y in ONTO_POS.values()) + BOX_H + 30
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="min-width:820px;font-family:inherit" role="img" '
         f'aria-label="Ontology: {len(ONTO_POS)} classes and {len(rels)} relations">',
         '<defs><marker id="oa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
         'orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#475467"/></marker></defs>']
    for e in edges:
        (x1, y1), (x2, y2) = e["p"], e["q"]
        d = f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" if e["ctrl"] is None else \
            f"M{x1:.1f},{y1:.1f} Q{e['ctrl'][0]:.1f},{e['ctrl'][1]:.1f} {x2:.1f},{y2:.1f}"
        s.append(f'<path d="{d}" fill="none" stroke="#98a2b3" stroke-width="1.6" marker-end="url(#oa)">'
                 f'<title>{e["a"]} -{e["r"]}-> {e["b"]}</title></path>')
    for e, (x, y, rect) in labels:
        s.append(f'<rect x="{rect[0]:.1f}" y="{rect[1]:.1f}" width="{rect[2] - rect[0]:.1f}" height="{rect[3] - rect[1]:.1f}" '
                 f'rx="10" fill="#ffffff" stroke="#e4e7ec"/><text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
                 f'font-size="11.5" fill="#344054" font-family="ui-monospace,Consolas,monospace">{_h.escape(e["r"])}</text>')
    for c, (x, y) in ONTO_POS.items():
        s.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" fill="#eef2ff" stroke="#2f5bd3" stroke-width="1.5"/>'
                 f'<text x="{x + BOX_W / 2}" y="{y + 22}" text-anchor="middle" font-weight="650" font-size="15" fill="#101828">{c}</text>'
                 f'<text x="{x + BOX_W / 2}" y="{y + 40}" text-anchor="middle" font-size="12" fill="#475467">'
                 f'{by_type.get(c, 0):,} nodes</text>')
    s.append("</svg>")
    return "".join(s)


def build_ui(out):
    out = Path(out)
    traces = json.loads((out / "traces.json").read_text())
    meta = json.loads((out / "run_meta.json").read_text())
    html = HTML.replace("__TRACES__", json.dumps(traces, default=str)).replace("__META__", json.dumps(meta)) \
               .replace("__ONTO__", json.dumps(ONTOLOGY)) \
               .replace("__ONTOSVG__", json.dumps(ontology_svg(meta["kg"]["by_type"])))
    (out / "control_tower.html").write_text(html, encoding="utf-8")
    svg = ontology_svg(meta["kg"]["by_type"]).replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1) \
        .replace('style="min-width:820px;font-family:inherit"', 'style="font-family:Segoe UI,Arial,sans-serif;background:#fff"', 1)
    (out / "ontology.svg").write_text(svg, encoding="utf-8")


def build_console(out):
    """Terminal-style page of the live run log (events arriving, agents acting) for the demo storyboard."""
    import html as _h
    out = Path(out)
    lines = (out / "console.log").read_text().splitlines()
    colour = lambda l: "#fbbf24" if "-->" in l else "#f87171" if "ISSUE" in l else "#86efac" if l.split("] ")[-1][:2] == "E-" else "#cbd5e1"
    body = "".join(f'<div style="color:{colour(l)}">{_h.escape(l)}</div>' for l in lines)
    (out / "console.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>run console</title>
<body style="margin:0;background:#0b1020;font:13px/1.55 ui-monospace,Consolas,monospace;padding:18px 22px;white-space:pre-wrap">
<div style="color:#94a3b8">$ python run.py --out ../runs/run1</div>{body}</body>""", encoding="utf-8")
