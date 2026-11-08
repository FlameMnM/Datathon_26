"""The agent team: one Coordinator + three specialists.

  Coordinator          - reads the event, runs LLM understanding, grounds every mention in the knowledge graph,
                         decides noise / escalate / act, routes incidents to specialists, runs guardrails and
                         writes the per-event summary.
  SupplyChainPlanner   - supplier delays + plant outages: finds affected production orders by KG traversal and
                         moves each one to the earliest feasible slot (capacity-aware, same plant first, else transfer).
  CustomerDealerAgent  - dealer network + customer care: one dealer.notify per affected dealer listing only its
                         affected customer orders; loaners for Platinum/Gold customers by policy priority and pool size.
  QualityAgent         - quality holds: identifies the lot and the affected vehicles, holds the lot, stops sale of unsold
                         vehicles at each dealer, contacts the owners of sold vehicles.
The LLM is used only by the Coordinator for understanding; every operational decision is deterministic code that
applies the policy handbook, so the same event always produces the same, auditable actions.
"""
from collections import defaultdict
from datetime import date

from config import LAST_IDX, LOANER_TIERS, TIER_RANK, TRANSFER_POLICY, date_to_idx, idx_to_date, next_working_idx
from nlu import resolve_date

REAL = ("supplier_delay", "plant_outage", "quality_hold")


class Trace:
    def __init__(self, event):
        self.event = event
        self.understanding = {}
        self.incidents = []          # grounded incidents
        self.reasoning = []          # [{"agent", "step"}]
        self.paths = []              # [{"incident", "title", "layers": [...]}]
        self.audit = []              # tool calls
        self.summary = ""
        self.category = []
        self.issues = []
        self.impact = {}

    def log(self, agent, step):
        self.reasoning.append({"agent": agent, "step": step})

    def to_dict(self):
        return {"event": self.event, "category": self.category, "summary": self.summary,
                "understanding": self.understanding, "incidents": self.incidents, "reasoning": self.reasoning,
                "graph_paths": self.paths, "audit_trail": self.audit, "issues": self.issues, "impact": self.impact,
                "actions": [a for a in self.audit if a["write"] or a["tool"] == "escalate"]}


def _layers(path):
    return [{"label": a, "ids": b} if b is not None else {"edge": a} for a, b in path]


# ====================================================================== Supply Chain Planner
class SupplyChainPlanner:
    name = "SupplyChainPlanner"

    def __init__(self, kg):
        self.kg = kg

    def impact(self, inc, trace):
        kg = self.kg
        if inc["kind"] == "supplier_delay":
            pos, path = kg.supplier_delay_impact(inc["shipment"], inc["new_eta"])
            inc["restart_idx"] = next_working_idx(date.fromisoformat(inc["new_eta"]))
            title = f"{inc['shipment']} ({inc['part']} -> {inc['plant']}) late: {inc['eta']} -> {inc['new_eta']}"
            trace.log(self.name, f"[{inc['id']}] BOM traversal: {inc['part']} is used by "
                                 f"{sorted(kg.neighbors(inc['part'], 'HAS_PART', reverse=True))}; at {inc['plant']} "
                                 f"{len(pos)} future production orders build in [{inc['eta']}, {inc['new_eta']}): {pos}")
        else:
            pos, path = kg.outage_impact(inc["plant"], date.fromisoformat(inc["start"]), date.fromisoformat(inc["end"]))
            inc["restart_idx"] = date_to_idx(date.fromisoformat(inc["end"])) + 1
            title = f"{inc['plant']} down {inc['start']} .. {inc['end']}"
            trace.log(self.name, f"[{inc['id']}] {inc['plant']} builds {len(pos)} future production orders during the "
                                 f"outage: {pos}; restart day {idx_to_date(inc['restart_idx']) if inc['restart_idx'] <= LAST_IDX else 'after window'}")
        _, down = kg.downstream_orders(pos)
        trace.paths.append({"incident": inc["id"], "title": title, "layers": _layers(path + down)})
        inc["affected_pos"] = pos
        return pos

    def blocked(self, disruptions, plant, idx, model):
        for d in disruptions:
            if d["kind"] == "plant_outage" and d["plant"] == plant and \
                    date_to_idx(date.fromisoformat(d["start"])) <= idx <= date_to_idx(date.fromisoformat(d["end"])):
                return True
            if d["kind"] == "supplier_delay" and d["plant"] == plant and d["part"] in self.kg.neighbors(model, "HAS_PART") \
                    and date_to_idx(date.fromisoformat(d["eta"])) <= idx < d["restart_idx"]:
                return True
        return False

    def run(self, disruptions, tools, trace):
        """Plan and execute moves for all delay/outage incidents of the event with one shared capacity view."""
        kg, free_cache = self.kg, {}

        def free(plant, idx):
            if (plant, idx) not in free_cache:
                ok, res = tools.call(self.name, "mes.get_schedule", {"plant_id": plant, "date": idx_to_date(idx).isoformat()},
                                     "read live capacity")
                free_cache[(plant, idx)] = res["free_slots"] if ok else 0
            return free_cache[(plant, idx)]

        affected = {}
        for d in disruptions:
            for po in self.impact(d, trace):
                affected[po] = max(affected.get(po, -99), d["restart_idx"])
        order = sorted(affected, key=lambda p: (kg.nodes[p]["idx"], p))
        moves = {}
        for po in order:
            P = kg.nodes[po]
            plant, model, earliest = P["plant_id"], P["model"], affected[po]
            placed = False
            for idx in range(max(earliest, 0), LAST_IDX + 1):        # 1) same plant, at/after the disruption end
                if self.blocked(disruptions, plant, idx, model) or free(plant, idx) <= 0:
                    continue
                ok, res = tools.call(self.name, "mes.reschedule", {"production_order_id": po, "new_date": idx_to_date(idx).isoformat()},
                                     f"earliest free slot at {plant} on/after restart {idx_to_date(earliest) if earliest <= LAST_IDX else 'n/a'}")
                if ok:
                    free_cache[(plant, idx)] -= 1
                    moves[po] = (plant, idx, "reschedule")
                    trace.log(self.name, f"{po} ({model}, was {P['build_date']}) -> {plant} {idx_to_date(idx)} (same plant, first free slot)")
                    placed = True
                    break
                free_cache.pop((plant, idx), None)                     # stale view: re-read and keep searching
            if placed:
                continue
            others = sorted(p for p in kg.neighbors(model, "BUILDS", reverse=True) if p != plant)
            trace.log(self.name, f"{po}: no free slot at {plant} from {idx_to_date(earliest) if earliest <= LAST_IDX else 'restart'} "
                                 f"to {idx_to_date(LAST_IDX)}; transfer candidates building {model}: {others}")
            t0 = 0 if TRANSFER_POLICY == "earliest" else max(earliest, 0)
            for idx in range(t0, LAST_IDX + 1):                        # 2) transfer: earliest day with a free slot
                for p2 in others:
                    if placed or self.blocked(disruptions, p2, idx, model) or free(p2, idx) <= 0:
                        continue
                    ok, res = tools.call(self.name, "mes.transfer",
                                         {"production_order_id": po, "to_plant_id": p2, "new_date": idx_to_date(idx).isoformat()},
                                         f"no same-plant capacity; earliest free slot at a plant that builds {model}")
                    if ok:
                        free_cache[(p2, idx)] -= 1
                        moves[po] = (p2, idx, "transfer")
                        trace.log(self.name, f"{po} ({model}) transferred {plant} -> {p2} on {idx_to_date(idx)}")
                        placed = True
                    else:
                        free_cache.pop((p2, idx), None)
                if placed:
                    break
            if not placed:
                trace.issues.append(f"{po} could not be placed anywhere in the planning window")
                tools.call(self.name, "escalate", {"reason": f"{po} ({model}) cannot be rescheduled or transferred inside the "
                                                             f"planning window; needs a planner decision"}, "no feasible slot")
        return order, moves


# ====================================================================== Customer & Dealer agent
class CustomerDealerAgent:
    name = "CustomerDealerAgent"

    def __init__(self, kg):
        self.kg = kg

    def run(self, pos, moves, tools, trace):
        kg = self.kg
        cos, _ = kg.downstream_orders(pos)
        by_dealer = defaultdict(list)
        for co in cos:
            by_dealer[kg.nodes[co]["dealer_id"]].append(co)
        trace.log(self.name, f"{len(cos)} affected customer orders at {len(by_dealer)} dealers")
        for dealer in sorted(by_dealer):
            ids = sorted(by_dealer[dealer])
            lines = []
            for co in ids:
                po = kg.nodes[co]["production_order_id"]
                if po in moves:
                    p, idx, how = moves[po]
                    lines.append(f"{co} ({po} now {p} {idx_to_date(idx)})")
                else:
                    lines.append(f"{co} ({po} delayed)")
            tools.call(self.name, "dealer.notify", {"dealer_id": dealer, "customer_order_ids": ids,
                                                    "message": "Delayed customer orders: " + "; ".join(lines)},
                       f"dealer has {len(ids)} affected customer orders")
        # loaners: Platinum before Gold, earlier promised date, lower customer order id; limited per dealer pool
        for dealer in sorted(by_dealer):
            entitled = [co for co in by_dealer[dealer] if kg.nodes[kg.nodes[co]["customer_id"]]["tier"] in LOANER_TIERS]
            if not entitled:
                continue
            entitled.sort(key=lambda c: (TIER_RANK[kg.nodes[kg.nodes[c]["customer_id"]]["tier"]], kg.nodes[c]["promised_date"], c))
            ok, res = tools.call(self.name, "crm.loaner_availability", {"dealer_id": dealer}, "check pool before assigning")
            avail = res.get("available", 0) if ok else 0
            give, skip = entitled[:avail], entitled[avail:]
            for co in give:
                ok, _ = tools.call(self.name, "crm.assign_loaner", {"customer_order_id": co},
                                   f"{kg.nodes[kg.nodes[co]['customer_id']]['tier']} customer, promised {kg.nodes[co]['promised_date']}")
                if not ok:
                    trace.issues.append(f"loaner for {co} failed")
            trace.log(self.name, f"{dealer}: pool {avail}, entitled {len(entitled)} -> loaners {give}"
                                 + (f"; no loaner left for {skip}" if skip else ""))
        return cos


# ====================================================================== Quality agent
class QualityAgent:
    name = "QualityAgent"

    def __init__(self, kg):
        self.kg = kg

    def run(self, inc, tools, trace):
        kg = self.kg
        for lot in inc["lots"]:
            unsold, sold, path = kg.quality_impact(lot)
            L = kg.nodes[lot]
            trace.paths.append({"incident": inc["id"], "title": f"{lot} ({L['part_id']} from {L['supplier_id']} at {L['plant_id']}, "
                                f"built {L['first_build_date']}..{L['last_build_date']})", "layers": _layers(path)})
            trace.log(self.name, f"{lot}: {len(unsold) + len(sold)} affected vehicles ({len(unsold)} in stock, {len(sold)} sold)")
            tools.call(self.name, "quality.hold_lot", {"lot_id": lot}, "defective lot")
            by_dealer = defaultdict(list)
            for v in unsold:
                by_dealer[kg.nodes[v]["dealer_id"]].append(v)
            for dealer in sorted(by_dealer):
                tools.call(self.name, "dealer.stop_sale", {"dealer_id": dealer, "vins": sorted(by_dealer[dealer])},
                           f"{len(by_dealer[dealer])} unsold vehicles from {lot}")
            owners = sorted({kg.nodes[v]["customer_id"] for v in sold})
            for cid in owners:
                vins = sorted(v for v in sold if kg.nodes[v]["customer_id"] == cid)
                tools.call(self.name, "crm.notify_customer",
                           {"customer_id": cid, "message": f"Safety notice: your vehicle(s) {', '.join(vins)} contain a part "
                                                           f"from quality-held lot {lot} ({L['part_id']}). Please contact your dealer."},
                           "owner of a sold affected vehicle")


# ====================================================================== Coordinator
class Coordinator:
    name = "Coordinator"

    def __init__(self, kg, extractor, rules, extractor_note=None):
        self.kg, self.extractor, self.rules, self.note = kg, extractor, rules, extractor_note
        self.scp, self.cda, self.qa = SupplyChainPlanner(kg), CustomerDealerAgent(kg), QualityAgent(kg)

    # ---------------------------------------------------------------- understanding
    def understand(self, text, trace):
        u = {"extractor": self.extractor.name, "model": getattr(self.extractor, "model", None), "note": self.note}
        before = dict(getattr(self.extractor, "usage", {}) or {})
        try:
            raw = self.extractor.extract(text)
            after = getattr(self.extractor, "usage", None)
            if after:
                u["tokens"] = {k: after.get(k, 0) - before.get(k, 0) for k in ("input_tokens", "output_tokens", "retries") if k in after}
        except Exception as e:
            trace.issues.append(f"LLM extraction failed ({e}); fell back to rules")
            raw = self.rules.extract(text)
            u["extractor"], u["model"] = "rules (fallback)", None
            if "auth" in str(e).lower():          # circuit breaker: don't keep calling an LLM we can't reach
                self.note = f"LLM disabled after authentication failure: {e}"
                self.extractor = self.rules
        u["raw"] = raw
        if self.extractor.name != "rules":                       # independent cross-check (LLM vs rules)
            chk = self.rules.extract(text)
            u["rules_crosscheck"] = chk
            u["agree_on_kinds"] = sorted(i["kind"] for i in raw) == sorted(i["kind"] for i in chk)
        trace.understanding = u
        return raw

    # ---------------------------------------------------------------- grounding (code-only validation)
    def ground(self, i, n, trace):
        kg, g = self.kg, {"id": f"I{n}", "kind": i["kind"], "summary": i.get("summary", ""), "llm_reason": i.get("reason", "")}
        if i["kind"] not in REAL:
            g["reason"] = i.get("reason", "")
            return g

        def res(field, typ):
            nid, how = kg.resolve(i.get(field, ""), typ)
            g.setdefault("resolution", {})[field] = f"'{i.get(field, '')}' -> {nid} ({how})"
            return nid

        def fail(why):
            g["kind"], g["reason"] = "unconfirmed", f"not actionable: {why}"
            trace.log(self.name, f"[{g['id']}] downgraded to unconfirmed: {why}")
            return g

        d1, how1 = resolve_date(i.get("date_1", ""))
        d2, how2 = resolve_date(i.get("date_2", ""))
        g["dates"] = {"date_1": f"'{i.get('date_1', '')}' -> {d1} ({how1})", "date_2": f"'{i.get('date_2', '')}' -> {d2} ({how2})"}
        if i["kind"] == "supplier_delay":
            part, plant, sup = res("part", "Part"), res("plant", "Plant"), res("supplier", "Supplier")
            sh = i.get("shipment_id", "").upper()
            if sh and sh in kg.nodes:
                S = kg.nodes[sh]
                if (part and part != S["part_id"]) or (plant and plant != S["plant_id"]):
                    return fail(f"{sh} does not match the part/plant in the text")
                shipments = [sh]
                if d1 and d1.isoformat() != S["eta_date"]:
                    return fail(f"{sh} ETA is {S['eta_date']}, text says {d1}")
            else:
                if not (part and plant and d1):
                    return fail("need part, plant and original arrival date to identify the shipment")
                shipments = kg.find_shipments(part, plant, sup, d1.isoformat())
            if len(shipments) != 1:
                return fail(f"shipment not uniquely identified (candidates {shipments})")
            S = kg.nodes[shipments[0]]
            if not d2 or d2.isoformat() <= S["eta_date"]:
                return fail("new ETA missing or not later than the original ETA")
            g.update(shipment=shipments[0], part=S["part_id"], plant=S["plant_id"], supplier=S["supplier_id"],
                     eta=S["eta_date"], new_eta=d2.isoformat())
        elif i["kind"] == "plant_outage":
            plant = res("plant", "Plant")
            if not (plant and d1 and d2) or d2 < d1:
                return fail("need the plant and a valid first/last day of the outage")
            g.update(plant=plant, start=d1.isoformat(), end=d2.isoformat())
        else:
            lot = i.get("lot_id", "").upper()
            part, plant, sup = res("part", "Part"), res("plant", "Plant"), res("supplier", "Supplier")
            if lot and lot in kg.nodes:
                L = kg.nodes[lot]
                if (part and part != L["part_id"]) or (plant and plant != L["plant_id"]) or (sup and sup != L["supplier_id"]):
                    return fail(f"{lot} does not match the part/plant/supplier in the text")
                lots = [lot]
            else:
                if not (part and plant and d1 and d2):
                    return fail("need part, plant and build window to identify the lot")
                cands = kg.find_lots(part, plant, sup, d1.isoformat(), d2.isoformat())
                exact = [l for l in cands if kg.nodes[l]["first_build_date"] == d1.isoformat() and kg.nodes[l]["last_build_date"] == d2.isoformat()]
                lots = exact or cands
                if not lots:
                    return fail("no lot of that part/plant covers the build window")
            g.update(lots=lots, part=kg.nodes[lots[0]]["part_id"], plant=kg.nodes[lots[0]]["plant_id"])
        g["grounded"] = True
        return g

    # ---------------------------------------------------------------- main loop
    def handle(self, event, toolclient):
        trace = Trace(event)
        tools = toolclient.session(event["event_id"], trace)
        raw = self.understand(event["text"], trace)
        m = trace.understanding.get("model")
        trace.log(self.name, f"understanding ({trace.understanding['extractor']}{' / ' + m if m else ''}): {len(raw)} incident(s): "
                             + ", ".join(f"{i['kind']}" for i in raw))
        inc = [self.ground(i, n + 1, trace) for n, i in enumerate(raw)]
        trace.incidents = inc
        real = [i for i in inc if i["kind"] in REAL]
        unclear = [i for i in inc if i["kind"] == "unconfirmed"]
        noise = [i for i in inc if i["kind"] == "noise"]
        trace.category = sorted({i["kind"] for i in inc})
        for i in noise:
            trace.log(self.name, f"[{i['id']}] noise: {i.get('reason') or 'no operational content'} -> no action")
        if unclear:
            reason = "; ".join(f"{i['summary']} ({i.get('reason', '')})" for i in unclear)
            tools.call(self.name, "escalate", {"reason": f"Unconfirmed / non-specific report, no operational change made: {reason}"},
                       "policy rule 8: unconfirmed or unclear -> escalate only")
            trace.log(self.name, f"escalated {len(unclear)} unclear item(s) to a human; no write actions")
        disruptions = [i for i in real if i["kind"] in ("supplier_delay", "plant_outage")]
        if disruptions:
            trace.log(self.name, f"routing {len(disruptions)} delay/outage incident(s) to SupplyChainPlanner, "
                                 f"then CustomerDealerAgent (shared slots and loaner pools)")
            order, moves = self.scp.run(disruptions, tools, trace)
            self.cda.run(order, moves, tools, trace)
        for q in (i for i in real if i["kind"] == "quality_hold"):
            trace.log(self.name, f"routing [{q['id']}] to QualityAgent: lots {q['lots']}")
            self.qa.run(q, tools, trace)
        # guardrails / completeness report
        writes = [a for a in trace.audit if a["write"]]
        failed = [a for a in trace.audit if a.get("sent") and not a["ok"]]
        kinds = sorted({a["tool"] for a in writes if a["ok"]})
        if real and len(kinds) < 3:
            trace.issues.append(f"only {len(kinds)} kinds of write action ({kinds}); policy rule 9 expects >= 3")
        if not real and writes:
            trace.issues.append("GUARDRAIL VIOLATION: write actions on a non-actionable event")
        trace.summary = self.summarize(inc, writes, failed, kinds)
        trace.impact = self.impact_report(inc, trace)
        return trace

    # ---------------------------------------------------------------- business impact: what changed / what did not
    def impact_report(self, inc, trace):
        kg = self.kg
        ok = [a for a in trace.audit if a["write"] and a["ok"]]
        moves = {a["args"]["production_order_id"]: a for a in ok if a["tool"] in ("mes.reschedule", "mes.transfer")}
        changed, unchanged, head = [], [], []
        real = [i for i in inc if i["kind"] in REAL]
        if not real:
            esc = any(a["tool"] == "escalate" for a in trace.audit)
            head.append("No operational impact." + (" Handed to a human for review." if esc else ""))
            changed.append("Nothing. No production order, dealer, customer, loaner, lot or vehicle record was written.")
            if esc:
                changed.append("One escalation to a human (not an operational change).")
            unchanged.append(f"All {len(kg.by_type['ProductionOrder'])} production orders, {len(kg.by_type['Dealer'])} dealers, "
                             f"all loaner pools and all {len(kg.by_type['Lot'])} lots are untouched.")
            return {"headline": " ".join(head), "changed": changed, "unchanged": unchanged}
        if moves:
            slips, transfers = [], 0
            for po, a in moves.items():
                old = date.fromisoformat(kg.nodes[po]["build_date"])
                slips.append((date.fromisoformat(a["args"]["new_date"]) - old).days)
                transfers += a["tool"] == "mes.transfer"
            later = [s for s in slips if s > 0]
            changed.append(f"{len(moves)} production orders moved ({len(moves) - transfers} same-plant reschedules, {transfers} transfers); "
                           + (f"{len(later)} now build later (avg {sum(later) / len(later):.1f} days)" if later else "none later")
                           + (f", {len(slips) - len(later)} earlier at another plant" if len(slips) > len(later) else "") + ".")
            for d in (i for i in real if i["kind"] in ("supplier_delay", "plant_outage")):
                plant = d["plant"]
                future = [p for p in kg.neighbors(plant, "BUILT_AT", reverse=True) if kg.nodes[p]["idx"] >= 0]
                untouched = [p for p in future if p not in moves]
                if d["kind"] == "supplier_delay":
                    lo, hi = d["eta"], d["new_eta"]
                    same_window = [p for p in untouched if lo <= kg.nodes[p]["build_date"] < hi]
                    unchanged.append(f"{len(same_window)} other orders at {plant} in the delay window ({lo} to {hi}) build models "
                                     f"without {d['part']} ({sorted({kg.nodes[p]['model'] for p in same_window})}), so they were not moved.")
                unchanged.append(f"{len(untouched)} of {len(future)} future production orders at {plant} were not affected and were not moved.")
        notif = [a for a in ok if a["tool"] == "dealer.notify"]
        if notif:
            cos = sorted({c for a in notif for c in a["args"]["customer_order_ids"]})
            loaned = {a["args"]["customer_order_id"] for a in ok if a["tool"] == "crm.assign_loaner"}
            tier = lambda c: kg.nodes[kg.nodes[c]["customer_id"]]["tier"]
            entitled = [c for c in cos if tier(c) in LOANER_TIERS]
            waiting = [c for c in entitled if c not in loaned]
            standard = [c for c in cos if tier(c) not in LOANER_TIERS]
            head.append(f"{len(cos)} customer deliveries affected at {len(notif)} dealers; {len(loaned)} Platinum/Gold customers got a loaner"
                        + (f", {len(waiting)} entitled customer{'s are' if len(waiting) != 1 else ' is'} waiting (dealer pool empty)." if waiting else "."))
            changed.append(f"{len(notif)} dealers told about exactly their affected orders (one message each); {len(loaned)} loaners assigned.")
            unchanged.append(f"{len(standard)} Standard-tier customers: dealer informed, no loaner (policy rule 5).")
            if waiting:
                unchanged.append(f"{len(waiting)} entitled customer{'s' if len(waiting) != 1 else ''} did not get a loaner because the dealer's pool ran out: {waiting[:6]}"
                                 + (" ..." if len(waiting) > 6 else ""))
            unchanged.append(f"{len(kg.by_type['Dealer']) - len(notif)} dealers have no affected orders and were not contacted.")
        for q in (i for i in real if i["kind"] == "quality_hold"):
            stops = [v for a in ok if a["tool"] == "dealer.stop_sale" for v in a["args"]["vins"]]
            owners = [a for a in ok if a["tool"] == "crm.notify_customer"]
            head.append(f"Lot {', '.join(q['lots'])} held; {len(stops)} unsold vehicles stopped from sale, {len(owners)} owners contacted.")
            changed.append(f"Lot {', '.join(q['lots'])} on quality hold; stop-sale on {len(stops)} in-stock vehicles at "
                           f"{sum(a['tool'] == 'dealer.stop_sale' for a in ok)} dealers; {len(owners)} owners notified.")
            L = kg.nodes[q["lots"][0]]
            other_lots = [l for l in kg.find_lots(L["part_id"], L["plant_id"]) if l not in q["lots"]]
            unchanged.append(f"{len(other_lots)} other {L['part_id']} lots at {L['plant_id']} ({', '.join(other_lots)}) are outside the "
                             f"suspect build window and were not held.")
            same_days = [v for v in kg.by_type["Vehicle"] if kg.nodes[v]["plant_id"] == L["plant_id"]
                         and L["first_build_date"] <= kg.nodes[v]["build_date"] <= L["last_build_date"]]
            no_part = [v for v in same_days if L["part_id"] not in kg.neighbors(kg.nodes[v]["model"], "HAS_PART")]
            unchanged.append(f"{len(no_part)} vehicles built at {L['plant_id']} on the same days use models without {L['part_id']} and were not touched."
                             if no_part else f"Every model {L['plant_id']} built on those days uses {L['part_id']}, so all {len(same_days)} vehicles "
                             f"from the window are in scope; vehicles from other days and plants were not touched.")
            unchanged.append("No production orders were rescheduled and no loaners assigned (not part of a quality hold).")
        if moves and not any(i["kind"] == "quality_hold" for i in real):
            head.insert(0, f"{len(moves)} builds rescheduled.")
            unchanged.append("No lots held, no vehicles stopped from sale (not part of a delay or outage).")
        return {"headline": " ".join(head), "changed": changed, "unchanged": unchanged}

    def summarize(self, inc, writes, failed, kinds):
        parts = []
        for i in inc:
            if i["kind"] == "noise":
                parts.append(f"{i['id']} noise - not about operations, no action.")
            elif i["kind"] == "unconfirmed":
                parts.append(f"{i['id']} unconfirmed - escalated to a human, no operational change ({i.get('reason', '')}).")
            elif i["kind"] == "supplier_delay":
                parts.append(f"{i['id']} supplier delay {i['shipment']} {i['part']}->{i['plant']} {i['eta']}->{i['new_eta']}: "
                             f"{len(i.get('affected_pos', []))} production orders affected.")
            elif i["kind"] == "plant_outage":
                parts.append(f"{i['id']} plant outage {i['plant']} {i['start']}..{i['end']}: {len(i.get('affected_pos', []))} production orders affected.")
            else:
                parts.append(f"{i['id']} quality hold on {', '.join(i['lots'])} ({i['part']} at {i['plant']}).")
        c = defaultdict(int)
        for w in writes:
            c[w["tool"]] += 1
        parts.append("Actions: " + (", ".join(f"{k} x{v}" for k, v in sorted(c.items())) or "none")
                     + (f"; {len(failed)} failed call(s)" if failed else "; 0 failed calls"))
        return " ".join(parts)
