## Inspiration

Wacky Wheels Co. builds cars just-in-time and keeps no parts inventory. One late battery shipment or a plant losing power for three days doesn't stay one problem for long: production schedules slip, dealers have to be told, customers need loaner cars, and a defective part lot can mean stopping sales and calling owners. In real operations centers, people work through this by hand, reading messy messages, cross-referencing spreadsheets and making dozens of calls.

We wanted to see whether a small team of AI agents could do that job **correctly**: not just sound confident, but take exactly the right actions, touch nothing it shouldn't, and show its work.

## What it does

Our **Control Tower** reads a feed of operational messages: shipment notices, chat messages, quality bulletins, daily digests and plain noise. For each one it:

1. **Understands the message.** A small LLM classifies it (supplier delay, plant outage, quality hold, noise or unconfirmed rumor), splits multi-incident digests into separate incidents, and pulls out the names and dates people actually wrote: "the Huntsville line", "Quanta", "next Friday".
2. **Grounds it in a knowledge graph.** Code resolves every nickname to a real record and every relative date to a calendar date, then walks the company's knowledge graph to find everything the event touches. For a supplier delay that path is Shipment → Part → Model → Production Order → Customer Order → Dealer / Customer.
3. **Acts through the enterprise tools, following the policy handbook.** Four agents split the work:
   - **Coordinator:** understands each message, decides between noise, escalation and action, routes the work and enforces the guardrails.
   - **Supply Chain Planner:** moves each affected build to the earliest feasible slot, at the same plant or by transfer, reading live capacity before every move.
   - **Customer & Dealer agent:** sends one notice per affected dealer, listing only that dealer's orders, and assigns loaners by customer tier and pool limits.
   - **Quality agent:** holds a defective lot, stops sale of unsold vehicles built with it, and contacts owners of the sold ones.
4. **Knows when not to act.** Noise gets zero tool calls. A vague rumor gets exactly one call: escalation to a human, with no operational change.
5. **Explains itself.** Every event gets a summary, a reasoning trace, the graph path it walked, a full audit trail, and a "What changed and what did not" report. The report covers the business impact and every record deliberately left alone. It's all shown in a browser-based control-tower dashboard.

**On the 15-event challenge feed:**

| Metric | Result |
|---|---|
| Events passing our independent policy checker | 15 / 15 |
| Tool calls | 879 (766 writes) |
| Failed calls | 0 |
| Production orders moved | 50 (45 reschedules, 5 transfers between plants) |
| Noise events (3) | 0 calls |
| Rumors (2) | 1 escalation each, no writes |

## How we built it

- **The LLM extracts; code decides.** We used **Google Gemini 3.5 Flash-Lite** with JSON-schema-constrained output at temperature 0, one call per event: 15 calls and about 9,000 tokens for the whole run. The model only copies phrases verbatim. Python resolves aliases, computes dates from "today", identifies shipments and lots, and applies the handbook rules. The model never invents an ID or a date, so the same event always produces the same actions.
- **The knowledge graph:** 12 classes and 19 typed relations, giving 17,040 nodes and 28,251 edges, built from the company CSVs in under 0.2 seconds. Every traversal returns both the records it found and the typed path it walked, so each trace can cite its evidence. We added a derived relation (Lot → Production Order) so a quality hold is a single graph walk to the affected vehicles.
- **Guardrails:**
  - **Grounding gate:** anything that can't be resolved to exactly one record becomes "unconfirmed" and is escalated instead of guessed.
  - **Pre-send validation:** every tool call is checked against the server's tool schema before it's sent.
  - **Retry discipline:** network errors are retried; business errors never are blindly.
  - **Circuit breaker:** if the LLM fails, a rule-based extractor takes over.
  - **Compound events:** incidents in the same message share one capacity view and one loaner pool, so the agents can never double-book a slot or a loaner.
- **Verification:** `verify.py` replays our action log on a fresh copy of the organizers' world. It compares the result with a separate implementation of the handbook that shares no code with the agents.
- **The UI:** a single static HTML dashboard with an event feed, impact report, graph-path diagrams, reasoning trace, actions, full audit trail and ontology diagram. It's backed by per-event JSON traces, a one-page design card and a 70-second demo video.

**Stack:** Python (standard library only, for the full pipeline), Google Gemini API, HTML / SVG / vanilla JavaScript, Pillow and ffmpeg for the demo video, Mermaid for the ontology export.

## Challenges we ran into

- **Dates in human language.** "Next Friday" written on a Monday is ambiguous. We checked our interpretation against the shipment and lot tables, then made it explicit and documented it rather than leaving it to the model.
- **Compound events.** A digest with a plant outage *and* a supplier delay competes for the same build slots and the same loaner cars. Handling the two incidents separately produced contradictions, so we moved to one shared capacity and loaner state per event.
- **Knowing when not to act.** We stress-tested the system on six deliberately vague messages from the organizers' explainer. The LLM handled them perfectly, but the test exposed a gap in our rule-based fallback: it labeled vague operational reports as noise, which would have meant no escalation. We fixed it and re-checked every message.
- **Free-tier rate limits.** We added client-side pacing and honored the API's requested retry delay, and finished the run with zero retries.
- **Tooling surprises.** For example, the local enterprise server couldn't immediately reuse its port after a run, which silently produced empty logs until we added automatic port selection.

## Accomplishments that we're proud of

- **15 / 15 events correct with zero failed calls**, verified by an independent checker, not just by our own agents.
- **Reproducibility:**
  - A fresh rerun reproduces all 877 operational calls exactly, in the same order. Only the wording of the two escalation notes changes.
  - With the LLM switched off entirely, the offline rule-based fallback produces the same actions.
- **A small, cheap model was enough:** about 9,000 tokens for the whole run, because the LLM only does the part that needs language.
- **Restraint and transparency:** noise left untouched, rumors escalated, and every action traceable to a graph path and a policy rule.

## What we learned

- Put the LLM where language is hard and keep it away from facts. Letting code own IDs, dates and policy made the system both more accurate and easier to explain.
- An independent checker is worth building early. It caught mistakes our agents were confidently making.
- "What did we deliberately *not* change?" is as important to a business user as "what did we change?"

## What's next for the Control Tower

- A head-to-head comparison of small models (for example Claude Haiku vs. Gemini Flash-Lite) on extraction accuracy, cost and latency.
- An adversarial event set (typos, conflicting dates, unknown suppliers) to stress-test the grounding gate.
- Human approval in the UI before large batches of write actions.
- An optimizer instead of greedy slot assignment, for when several disruptions compete for the same capacity.
