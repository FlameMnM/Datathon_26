"""Independent checker: replays an action_log.jsonl on a fresh copy of the organizers' world (enterprise.py)
and compares the end state with outcomes computed here from the raw CSVs by a SEPARATE implementation of the
policy handbook (no shared code with the agents).

The incident parameters below are a hand-written TEST FIXTURE (my reading of each message). They are used only
to grade a run; the agent system never reads this file.

python verify.py ../runs/run1/action_log.jsonl
"""
import csv
import os
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "hack_the_highway_student_package" / "hack_the_highway_package"
sys.path.insert(0, str(PKG))
from enterprise import Enterprise, date_to_idx  # noqa: E402  (organizers' world model)

FIXTURE = {
    "E-01": [("outage", "P_TX", "2027-03-19", "2027-03-22"), ("delay", "SH-0190", "2027-03-19")],
    "E-02": [("unconfirmed",)],
    "E-03": [("quality", "LOT-0130")],
    "E-04": [("outage", "P_KY", "2027-03-16", "2027-03-18")],
    "E-05": [("noise",)], "E-06": [("noise",)], "E-07": [("noise",)],
    "E-08": [("outage", "P_KY", "2027-03-24", "2027-03-25")],
    "E-09": [("delay", "SH-0256", "2027-04-01")],
    "E-10": [("delay", "SH-0248", "2027-04-01")],
    "E-11": [("outage", "P_MS", "2027-03-26", "2027-03-30")],
    "E-12": [("delay", "SH-0182", "2027-03-18")],
    "E-13": [("unconfirmed",)],
    "E-14": [("quality", "LOT-0017")],
    "E-15": [("quality", "LOT-0191")],
}


def rows(n):
    with open(PKG / "data" / n, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


PO = {r["production_order_id"]: r for r in rows("production_orders.csv")}
SH = {r["shipment_id"]: r for r in rows("inbound_shipments.csv")}
LOT = {r["lot_id"]: r for r in rows("part_lots.csv")}
CO = rows("customer_orders.csv")
CUST = {r["customer_id"]: r["tier"] for r in rows("customers.csv")}
FV = rows("finished_vehicles.csv")
POOL = {r["dealer_id"]: int(r["loaners_available"]) for r in rows("loaners.csv")}
SLOTS = {r["plant_id"]: int(r["slots_per_day"]) for r in rows("plants.csv")}
BUILDS = defaultdict(set)
for r in rows("plant_models.csv"):
    BUILDS[r["plant_id"]].add(r["model"])
BOM = defaultdict(set)
for r in rows("bom.csv"):
    BOM[r["model"]].add(r["part_id"])
D = lambda s: date_to_idx(date.fromisoformat(s))


def nwi(s):  # next working-day index on/after a date string
    d = date.fromisoformat(s)
    while d.weekday() > 4:
        d += timedelta(days=1)
    return date_to_idx(d)


def expected(incidents):
    """Independent expected outcome for one event."""
    exp = {"affected": {}, "notify": defaultdict(set), "loaners": set(), "held": set(), "stopped": set(),
           "customers": set(), "escalate": False, "kinds_min": 0, "blocks": []}
    for inc in incidents:
        k = inc[0]
        if k in ("noise",):
            continue
        if k == "unconfirmed":
            exp["escalate"] = True
            continue
        exp["kinds_min"] = 3
        if k == "outage":
            _, plant, s, e = inc
            exp["blocks"].append(("outage", plant, D(s), D(e), None))
            for po, r in PO.items():
                if r["plant_id"] == plant and D(s) <= D(r["build_date"]) <= D(e) and D(r["build_date"]) >= 0:
                    exp["affected"][po] = max(exp["affected"].get(po, -1), D(e) + 1)
        elif k == "delay":
            _, sh, new = inc
            S = SH[sh]
            exp["blocks"].append(("delay", S["plant_id"], D(S["eta_date"]), nwi(new) - 1, S["part_id"]))
            for po, r in PO.items():
                if r["plant_id"] == S["plant_id"] and S["part_id"] in BOM[r["model"]] and \
                        D(S["eta_date"]) <= D(r["build_date"]) < D(new) and D(r["build_date"]) >= 0:
                    exp["affected"][po] = max(exp["affected"].get(po, -1), nwi(new))
        elif k == "quality":
            L = LOT[inc[1]]
            exp["held"].add(inc[1])
            for v in FV:
                if v["plant_id"] == L["plant_id"] and L["first_build_date"] <= v["build_date"] <= L["last_build_date"] \
                        and L["part_id"] in BOM[v["model"]]:
                    if v["status"] == "in_stock":
                        exp["stopped"].add(v["vin"])
                    else:
                        exp["customers"].add(v["customer_id"])
    aff_cos = [c for c in CO if c["production_order_id"] in exp["affected"]]
    for c in aff_cos:
        exp["notify"][c["dealer_id"]].add(c["customer_order_id"])
    ent = defaultdict(list)
    for c in aff_cos:
        if CUST[c["customer_id"]] in ("Platinum", "Gold"):
            ent[c["dealer_id"]].append(c)
    rank = {"Platinum": 0, "Gold": 1}
    for d, lst in ent.items():
        lst.sort(key=lambda c: (rank[CUST[c["customer_id"]]], c["promised_date"], c["customer_order_id"]))
        exp["loaners"].update(c["customer_order_id"] for c in lst[:POOL[d]])
    return exp


def blocked(blocks, plant, idx, model):
    for kind, p, lo, hi, part in blocks:
        if p == plant and lo <= idx <= hi and (kind == "outage" or part in BOM[model]):
            return True
    return False


def check_moves(exp, st, ent):
    """Moves: every affected order moved, nothing else moved, same-plant moves at/after restart, slots respected,
    and a greedy earliest-feasible simulation (date, id order) for an exact-position comparison."""
    orig = {po: (r["plant_id"], D(r["build_date"])) for po, r in PO.items() if D(r["build_date"]) >= 0}
    moved = {po for po, loc in st["loc"].items() if loc != orig[po]}
    aff = set(exp["affected"])
    issues = []
    if aff - moved:
        issues.append(f"affected orders not moved: {sorted(aff - moved)}")
    if moved - aff:
        issues.append(f"UNAFFECTED orders moved: {sorted(moved - aff)}")
    for po in aff & moved:
        p, i = st["loc"][po]
        if p == orig[po][0] and i < exp["affected"][po]:
            issues.append(f"{po} rescheduled before restart")
        if blocked(exp["blocks"], p, i, PO[po]["model"]):
            issues.append(f"{po} moved into a blocked plant-day")
    # greedy reference
    cap = defaultdict(int)
    for po, (p, i) in orig.items():
        cap[(p, i)] += 1
    ref = {}
    for po in sorted(aff, key=lambda x: (orig[x][1], x)):
        p0, model = orig[po][0], PO[po]["model"]
        spot = next(((p0, i) for i in range(exp["affected"][po], 20)
                     if not blocked(exp["blocks"], p0, i, model) and cap[(p0, i)] < SLOTS[p0]), None)
        if spot is None:
            t0 = 0 if os.environ.get("HTW_TRANSFER", "earliest") == "earliest" else exp["affected"][po]
            spot = next(((p, i) for i in range(t0, 20) for p in sorted(BUILDS) if p != p0 and model in BUILDS[p]
                         and not blocked(exp["blocks"], p, i, model) and cap[(p, i)] < SLOTS[p]), None)
        if spot:
            cap[orig[po]] -= 1
            cap[spot] += 1
            ref[po] = spot
    exact = sum(1 for po in aff if st["loc"].get(po) == ref.get(po))
    return issues, exact, len(aff), ref


def main(log_path):
    ent = Enterprise(str(PKG / "data"))
    calls = [json.loads(l) for l in open(log_path)]
    by_ev = defaultdict(list)
    for c in calls:
        by_ev[c["event_id"]].append(c)
        try:
            ent.call(c["event_id"], c["tool"], c["args"])
        except Exception:
            pass
    total, rows_out = 0.0, []
    for ev, incs in FIXTURE.items():
        st = ent.state(ev)
        exp = expected(incs)
        evcalls = by_ev.get(ev, [])
        writes = [c for c in evcalls if c["tool"] not in ("mes.get_schedule", "crm.loaner_availability", "escalate")]
        failed = [c for c in evcalls if not c["ok"]]
        checks = {}
        if not exp["kinds_min"]:                       # noise / unconfirmed
            checks["no_writes"] = not writes
            checks["escalated_iff_unclear"] = st["escalated"] == exp["escalate"]
        else:
            if exp["affected"]:
                issues, exact, n, ref = check_moves(exp, st, ent)
                checks["moves_valid"] = not issues
                checks["moves_exact_vs_greedy"] = f"{exact}/{n}"
                if issues:
                    print(ev, "MOVE ISSUES:", issues[:4])
            notify_calls = defaultdict(int)
            for c in evcalls:
                if c["tool"] == "dealer.notify" and c["ok"]:
                    notify_calls[c["args"]["dealer_id"]] += 1
            checks["dealers_notified_exact"] = {d: set(v) for d, v in st["notified"].items()} == dict(exp["notify"])
            checks["one_notify_per_dealer"] = all(v == 1 for v in notify_calls.values())
            checks["loaners_exact"] = st["loaned"] == exp["loaners"]
            checks["lots_exact"] = st["held"] == exp["held"]
            checks["stop_sale_exact"] = st["stopped"] == exp["stopped"]
            checks["owners_exact"] = st["customers"] == exp["customers"]
            kinds = {c["tool"] for c in writes if c["ok"]}
            checks[">=3_action_kinds"] = len(kinds) >= 3
            checks["no_escalation"] = not st["escalated"]
        checks["no_failed_calls"] = not failed
        bools = [v for v in checks.values() if isinstance(v, bool)]
        score = sum(bools) / len(bools)
        total += score
        rows_out.append((ev, score, checks, len(writes), len(failed)))
    for ev, score, checks, w, f in rows_out:
        bad = [k for k, v in checks.items() if v is False]
        extra = checks.get("moves_exact_vs_greedy", "")
        print(f"{ev}  {score * 100:5.1f}%  writes={w:3d} failed={f}  {('moves ' + extra) if extra else '':14s} "
              f"{'ALL CHECKS PASS' if not bad else 'FAIL: ' + ', '.join(bad)}")
    print(f"\nEvent-handling accuracy (mean of per-event check pass rates): {total / len(FIXTURE) * 100:.1f}%")
    return total / len(FIXTURE)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "runs" / "run1" / "action_log.jsonl"))
