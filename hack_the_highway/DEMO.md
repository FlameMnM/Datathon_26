# Demo — end-to-end run (storyboard)

The brief accepts "a short end-to-end demo video (or screenshots) showing events arriving, the graph path, the agents acting and the audit trail". This is the screenshot version, in playback order:

- **[runs/run1/demo.mp4](runs/run1/demo.mp4):** 70 s video: a title card, the 12 captioned frames (5 s each) and a results card.
- **[runs/run1/demo.gif](runs/run1/demo.gif):** the same 12 frames, about 54 s, plays in any browser.
- **[runs/run1/screenshots/](runs/run1/screenshots/):** the same frames as full-resolution PNGs.

Every frame comes from the actual Run 1: model `gemini-3.5-flash-lite`, 15 events, 879 tool calls, 0 failed.

| # | Frame | What it shows | Brief requirement |
|---|---|---|---|
| 1 | `01_run_console.png` | Live console: the feed is read from `GET /events`, each event "arrives" with a timestamp and is classified and handled; per-event write and failure counts | events arriving, agents acting |
| 2 | `02_E-01_event.png` | Compound digest E-01: the LLM splits it into a plant outage and a supplier delay; each mention and date is grounded to an id or ISO date (e.g. "next Friday" → 2027-03-19) | event understanding |
| 3 | `03_E-01_graph_paths.png` | One graph path per incident: Plant → ProductionOrder → CustomerOrder → Dealer/Customer, and Shipment → Part → Model (BOM) → ProductionOrder → … | graph path (TC1, TC2, TC5) |
| 4 | `04_E-01_trace.png` | Reasoning trace by agent: Coordinator routes; SupplyChainPlanner moves each order to the earliest free slot; CustomerDealerAgent notifies dealers and assigns loaners by tier, date and pool | agents acting |
| 5 | `05_E-01_actions.png` | Actions taken, each with agent, arguments, reason and result | actions, audit |
| 5b | `05b_E-01_what_changed.png` | What changed and what did not: the business-impact headline (builds moved, average slip, customers and dealers affected, loaners given vs waiting), plus every record deliberately left alone, with counts | "What changed? What did not?" (explainer chapter 6) |
| 6 | `06_E-11_slot_reasoning_transfers.png` | Capacity-aware slot reasoning: same-plant days fill up, so three orders are transferred to other plants that build the model | slot reasoning (TC2) |
| 7 | `07_E-03_quality_graph_path.png` | Quality hold: Lot → ProductionOrder → Vehicle → Dealer (unsold → stop sale) / Customer (sold → contact owner) | TC3 |
| 8 | `08_E-05_noise_no_action.png` | Noise: judged not operational, so zero tool calls | TC4 |
| 9 | `09_E-13_rumor_escalated.png` | Unconfirmed rumor: a single `escalate`, no write actions | TC6, human-in-the-loop |
| 10 | `10_audit_trail.png` | Full audit trail of the run: every read and write, with agent, arguments, reason and result | audit trail |
| 11 | `11_ontology.png` | Ontology: 12 classes, 19 relations, 17,040 nodes | knowledge graph |

To reproduce: run `python control_tower/run.py`, open `runs/run1/control_tower.html`, and click through the events. Deep links such as `control_tower.html#E-11:trace` open a single section; `#ontology` and `#audit` open those tabs. To rebuild the video and GIF, run `python control_tower/make_demo.py`.
