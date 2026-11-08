# Hack the Highway — Event-Driven Multi-Agent Control Tower

Team **penberthy-pushers**. See [DESIGN_CARD.md](DESIGN_CARD.md) / [DESIGN_CARD.pdf](DESIGN_CARD.pdf) (one page) for the architecture and design decisions, and [EVENT_SUMMARIES.md](EVENT_SUMMARIES.md) for what the system did on each event and why.

## Where each submission requirement lives

| Requirement (brief, "Submission Notes") | File(s) |
|---|---|
| `action_log.jsonl` from the run on the 15 events | [runs/run1/action_log.jsonl](runs/run1/action_log.jsonl) |
| Source code | [control_tower/](control_tower/) |
| Export or diagram of the ontology / graph | [runs/run1/ontology.svg](runs/run1/ontology.svg) (diagram, opens in any browser), [runs/run1/ontology.json](runs/run1/ontology.json) (export), [runs/run1/ontology.mmd](runs/run1/ontology.mmd) (Mermaid), [screenshots/11_ontology.png](runs/run1/screenshots/11_ontology.png) |
| A trace for each event | [runs/run1/traces/E-01.json ... E-15.json](runs/run1/traces/) (all in one file: `runs/run1/traces.json`) |
| Short end-to-end demo video (or screenshots) | [runs/run1/demo.mp4](runs/run1/demo.mp4) (70 s), [runs/run1/demo.gif](runs/run1/demo.gif), [runs/run1/screenshots/](runs/run1/screenshots/), storyboard [DEMO.md](DEMO.md) |
| Per-event summary + reason for what it did / did not do | [EVENT_SUMMARIES.md](EVENT_SUMMARIES.md); `summary`, `reasoning`, `impact` in each trace; the UI |
| One-page design card | [DESIGN_CARD.pdf](DESIGN_CARD.pdf) (source [DESIGN_CARD.md](DESIGN_CARD.md); longer version [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md)) |
| Cite the model(s) used | Google Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`): design card "Model used", `runs/run1/run_meta.json`, every trace (`understanding.model`) |

## Run

Needs Python 3.11+, standard library only.

The default LLM is Google **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`). It is the only model used in the submitted run; see the model citation in the design card. Put an API key in the environment as `GEMINI_API_KEY`, or in `~/.config/hack_the_highway/gemini_api_key`, kept outside the repo. Optional Claude backend: `pip install anthropic` plus `ANTHROPIC_API_KEY`. With no key at all, the system falls back to the rule-based extractor automatically.

```sh
cd control_tower
python run.py --out ../runs/run2                  # starts the organizers' server, handles all 15 events, builds the UI
python verify.py ../runs/run2/action_log.jsonl    # independent replay + policy check (run1 = the submitted run)
```

Options:
- `--llm auto|gemini|claude|rules` (default `auto`)
- `HTW_GEMINI_MODEL` or `HTW_MODEL` (Claude) to choose the model
- `HTW_TRANSFER=earliest|after_restart` for the transfer-day rule
- `--port` (a free port is chosen automatically)
- `--overwrite` to reuse an existing output folder (the organizers' server never overwrites a log otherwise)

A rerun reproduces the submitted Run 1 call for call; only the free-text wording of the two escalation reasons can differ. `--llm rules` (no API key, fully offline) produces the same actions.

The organizers' package is expected unzipped at `hack_the_highway_student_package/hack_the_highway_package/`.

## Deliverables (in `runs/run1/`)

| File | Contents |
|---|---|
| `action_log.jsonl` | Every tool call from the run on the 15 events (written by the organizers' server) |
| `traces/E-xx.json` | Per-event trace: understanding, grounding, graph paths, reasoning, actions, audit trail, summary |
| `control_tower.html` | Control-tower UI (open in a browser): event feed, graph path, reasoning, actions, audit trail, ontology |
| `ontology.svg`, `ontology.json`, `ontology.mmd` | Ontology diagram (SVG), export (classes, relations, graph stats) and Mermaid source |
| `demo.mp4`, `demo.gif`, `screenshots/*.png` | Demo: 70 s video (title, 12 captioned frames, results) and the same 12 frames as a ~54 s GIF and as PNGs (storyboard in [DEMO.md](DEMO.md)) |
| `console.log`, `console.html` | Live run console: events arriving and being handled |
| `run_meta.json` | Model / extractor used, runtime, graph stats |

## Tests

`tests/explainer_events.json` holds the six sample messages from the organizers' explainer. Run them with:

```sh
python run.py --events ../tests/explainer_events.json --out ../runs/test_explainer
```

Expected: five escalations, one noise, 0 writes. Results are in `runs/test_explainer_gemini/` and `runs/test_explainer_rules/`.

## Source (`control_tower/`)

| Module | Role |
|---|---|
| `kg.py` | Ontology + knowledge graph, name resolution, traversals that return their graph path |
| `nlu.py` | Gemini extractor (used in Run 1, stdlib REST, JSON schema), optional Claude extractor, rule-based extractor, relative-date resolver |
| `agents.py` | Coordinator, SupplyChainPlanner, CustomerDealerAgent, QualityAgent |
| `tools.py` | Audited, validated tool client for `POST /act` |
| `run.py`, `ui.py`, `verify.py` | Runner, UI builder, independent checker |
| `make_demo.py`, `make_summaries.py`, `make_card.py` | Demo GIF + MP4 (`pip install pillow imageio-ffmpeg`), `EVENT_SUMMARIES.md`, design-card HTML (`pip install markdown`; print to PDF) |
# Datathon_26
