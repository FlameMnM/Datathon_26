# Design notes (extended) — Wacky Wheels Control Tower (team penberthy-pushers)

The one-page design card is `DESIGN_CARD.md` / `DESIGN_CARD.pdf`. This file keeps the longer version of every section.

**Architecture.** One Coordinator plus three specialist agents, all calling the enterprise tools through one audited `ToolSession` (`POST /act`).

| Agent | Responsibility | Tools it may call |
|---|---|---|
| **Coordinator** | Understands the event with the LLM, grounds every mention in the knowledge graph, decides noise / escalate / act, routes incidents, runs guardrails, writes the summary | `escalate` |
| **SupplyChainPlanner** | Supplier delays and plant outages: finds affected production orders by graph traversal and moves each to the earliest feasible slot (same plant at or after the restart, else transfer) using a live capacity view | `mes.get_schedule`, `mes.reschedule`, `mes.transfer` |
| **CustomerDealerAgent** | One `dealer.notify` per affected dealer, listing only that dealer's affected customer orders. Loaners for Platinum/Gold customers in policy priority, capped by each dealer's pool | `dealer.notify`, `crm.loaner_availability`, `crm.assign_loaner` |
| **QualityAgent** | Lot → vehicles. Holds the lot, stops sale of unsold vehicles per dealer, contacts owners of sold vehicles | `quality.hold_lot`, `dealer.stop_sale`, `crm.notify_customer` |

**Knowledge graph.** 12 classes and 19 typed relations; 17,040 nodes and 28,251 edges, built in memory from the CSVs in under 0.2 s. The derived edge `Lot -INSTALLED_IN-> ProductionOrder` makes quality traversal a graph walk. Every traversal returns both the records and the typed path it walked, for example `Shipment -CARRIES-> Part <-HAS_PART- Model <-OF_MODEL- ProductionOrder <-FULFILLED_BY- CustomerOrder -> Dealer / Customer`. Each event's trace cites one path per incident. Exports: `ontology.svg` (diagram), `ontology.json`, `ontology.mmd` (Mermaid), and the Ontology tab of the UI.

**LLM versus code.**
- **LLM (Google Gemini 3.5 Flash-Lite, schema-constrained JSON output)** does only what language needs: classify, split a digest into incidents, and copy mentions and date phrases *verbatim*.
- **Code** does everything that must be exact:
  - resolve names through `aliases.csv` and the graph;
  - compute relative dates ("next Friday", "last Monday") from 2027-03-08;
  - identify the shipment or lot;
  - apply the handbook rules and plan slots.

So the same event always produces the same auditable actions, and the model never invents ids or dates. A rule-based extractor implements the same schema. It is the offline fallback, and it cross-checks the LLM on every event: disagreements show in the trace.

**Guardrails.**
1. **Grounding gate:** an incident whose part, plant, shipment, lot or dates can't be uniquely resolved in the graph is downgraded to *unconfirmed* and escalated. The system never acts on a guess.
2. **Rumors:** handled as "escalate only".
3. **Noise:** gets zero tool calls.
4. **Pre-send validation:** every call is checked against the `/tools` schema before it is sent. Capacity is read live with `mes.get_schedule` (free) before each move, so no move fails.
5. **Failure handling:** transport errors are retried with backoff. Business errors (4xx) are never blindly retried: the planner refreshes its view and tries the next candidate. An order with no feasible slot anywhere is escalated.
6. **LLM failures:** the circuit breaker falls back to rules.
7. **Completeness check:** each real incident is checked for at least 3 kinds of write action.
8. **Compound events:** all incidents share one capacity view and one loaner allocation. A transfer can never land in another incident's blocked plant-days, and each dealer gets exactly one notification.

**Evaluation.** `verify.py` replays `action_log.jsonl` on a fresh copy of the organizers' `Enterprise` world. It compares the end state with outcomes from a *separate* implementation of the handbook, which reads the raw CSVs and shares no code with the agents.

Run 1: **15/15 events pass all checks, 879 calls, 0 failed calls, 766 writes**. All 50 moved orders (45 reschedules, 5 transfers) match an independent greedy earliest-slot plan exactly. Runtime is about 1 minute (63 s), mostly free-tier request pacing; the planning itself takes under 1 s. A fresh rerun reproduces all 877 operational calls exactly, in the same order (only the wording of the 2 escalation reasons varies), and the offline rules-only fallback produces the same actions.

**Robustness test.** The six vague sample messages from the organizers' explainer (`tests/explainer_events.json`) were run through the full system. Gemini escalated all five operational-but-unspecific reports, ignored the brochure question, and made 0 writes. This test also exposed a gap in the rule-based fallback: it called vague operational reports "noise", which would mean no escalation. It now escalates them, re-checked on all 21 messages.

**Business-impact report (UX).** Each event ends with "What changed - and what did not":
- a headline: builds moved and the average slip, customers and dealers affected, loaners given versus entitled customers still waiting;
- the changed records;
- every record deliberately left alone, with counts: other orders at the plant, models without the delayed part, other lots, Standard-tier customers and uncontacted dealers.

**Models used (citation).**

| Model | Provider and access | Used for | Run 1 usage |
|---|---|---|---|
| **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`) | Google Gemini API, `v1beta/models/gemini-3.5-flash-lite:generateContent`, free tier | Event understanding only: classify, split incidents, extract mentions and date phrases. Temperature 0, JSON-schema-constrained output, one call per event | 15 calls, 6,761 input and 2,187 output tokens, 0 retries |

- **No other ML model** is used anywhere in the run.
- **The cross-checker is plain regular expressions** over `aliases.csv`, not a model.
- **All planning and policy logic is deterministic Python.**
- **Every trace names its model:** each `traces/E-xx.json` records `understanding.model` and its token counts.
- **Optional backend, not used in Run 1:** Anthropic Claude via the official `anthropic` SDK (`--llm claude`, `HTW_MODEL`). It is contract-tested offline, but no credentials were available for a live run.

**Rate limits.** Requests are paced 4.5 s apart for the free tier, and a 429 waits for the server's requested `retryDelay`. Gemini agreed with the rule-based parser on the incident types for all 15 events.

**Interpretation choices (documented, switchable).**
- "Next <weekday>" means that weekday in next week; "last <weekday>" means the previous week. This is consistent with the shipment and lot tables.
- A transfer goes to the earliest day with a free slot at any plant that builds the model (handbook: "any day"), with ties broken by plant id. Setting `HTW_TRANSFER=after_restart` keeps transfers on or after the disruption end; it changes only 5 orders.

**What we'd improve.**
- Run and evaluate the LLM path with credentials, comparing Claude Haiku 5.5 against Opus 5.5 on extraction accuracy, cost and latency.
- Add an adversarial event set (typos, conflicting dates, unknown suppliers) to stress-test the grounding gate.
- Add human-in-the-loop approval in the UI before large write batches.
- Replace greedy slot assignment with an optimizer when several disruptions compete for the same capacity.
