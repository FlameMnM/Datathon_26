"""Ontology + knowledge graph over the Wacky Wheels company data.

Classes:   Supplier, Part, Model, Plant, Shipment, Lot, ProductionOrder, CustomerOrder, Customer, Dealer, Vehicle, Alias
Relations: see ONTOLOGY below. Every traversal used by the agents returns the records it found AND the typed
path it walked, so each event's trace can cite its graph path (e.g. Shipment -> Part -> Model -> ProductionOrder ...).
"""
import csv
import json
import re
from collections import defaultdict
from datetime import date

from config import DATA, date_to_idx

ONTOLOGY = {
    "classes": {
        "Supplier": "External parts supplier (suppliers.csv)",
        "Part": "Component in a bill of materials (parts.csv)",
        "Model": "Vehicle model (models.csv)",
        "Plant": "Assembly plant with slots_per_day build capacity (plants.csv)",
        "Shipment": "Inbound shipment of a part to a plant with an ETA (inbound_shipments.csv)",
        "Lot": "Part lot consumed at a plant over a build window (part_lots.csv)",
        "ProductionOrder": "Daily build order of one model at one plant (production_orders.csv)",
        "CustomerOrder": "Customer purchase promised from a production order (customer_orders.csv)",
        "Customer": "Buyer / owner with loyalty tier (customers.csv)",
        "Dealer": "Dealership with a loaner pool (dealers.csv, loaners.csv)",
        "Vehicle": "Finished vehicle by VIN, in stock or sold (finished_vehicles.csv)",
        "Alias": "Free-text name variant (aliases.csv)",
    },
    "relations": [
        ("Supplier", "SUPPLIES", "Part"),
        ("Shipment", "SHIPPED_BY", "Supplier"),
        ("Shipment", "CARRIES", "Part"),
        ("Shipment", "DELIVERS_TO", "Plant"),
        ("Model", "HAS_PART", "Part"),
        ("Plant", "BUILDS", "Model"),
        ("ProductionOrder", "BUILT_AT", "Plant"),
        ("ProductionOrder", "OF_MODEL", "Model"),
        ("CustomerOrder", "FULFILLED_BY", "ProductionOrder"),
        ("CustomerOrder", "VIA_DEALER", "Dealer"),
        ("CustomerOrder", "PLACED_BY", "Customer"),
        ("Lot", "OF_PART", "Part"),
        ("Lot", "MADE_BY", "Supplier"),
        ("Lot", "CONSUMED_AT", "Plant"),
        ("Lot", "INSTALLED_IN", "ProductionOrder"),
        ("Vehicle", "BUILT_FROM", "ProductionOrder"),
        ("Vehicle", "AT_DEALER", "Dealer"),
        ("Vehicle", "OWNED_BY", "Customer"),
        ("Alias", "REFERS_TO", "*"),
    ],
}


def _rows(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}                              # id -> {"type": ..., **attrs}
        self.out = defaultdict(list)                 # id -> [(rel, dst)]
        self.inn = defaultdict(list)                 # id -> [(rel, src)]
        self.by_type = defaultdict(list)
        self.alias_index = {}                        # lowercase text -> (type, id)
        self._build()

    # ------------------------------------------------------------------ construction
    def add_node(self, nid, ntype, **attrs):
        if nid not in self.nodes:
            self.by_type[ntype].append(nid)
        self.nodes[nid] = {"type": ntype, **attrs}

    def add_edge(self, src, rel, dst):
        self.out[src].append((rel, dst))
        self.inn[dst].append((rel, src))

    def _build(self):
        for r in _rows("suppliers.csv"):
            self.add_node(r["supplier_id"], "Supplier", name=r["name"], city=r["city"])
        for r in _rows("parts.csv"):
            self.add_node(r["part_id"], "Part", name=r["name"], category=r["category"])
        for r in _rows("models.csv"):
            self.add_node(r["model"], "Model", segment=r["segment"])
        for r in _rows("plants.csv"):
            self.add_node(r["plant_id"], "Plant", name=r["name"], city=r["city"], slots_per_day=int(r["slots_per_day"]))
        for r in _rows("plant_models.csv"):
            self.add_edge(r["plant_id"], "BUILDS", r["model"])
        for r in _rows("bom.csv"):
            self.add_edge(r["model"], "HAS_PART", r["part_id"])
        supplies = set()
        for r in _rows("inbound_shipments.csv"):
            self.add_node(r["shipment_id"], "Shipment", eta_date=r["eta_date"], covers_days=int(r["covers_days"]),
                          supplier_id=r["supplier_id"], part_id=r["part_id"], plant_id=r["plant_id"])
            self.add_edge(r["shipment_id"], "SHIPPED_BY", r["supplier_id"])
            self.add_edge(r["shipment_id"], "CARRIES", r["part_id"])
            self.add_edge(r["shipment_id"], "DELIVERS_TO", r["plant_id"])
            supplies.add((r["supplier_id"], r["part_id"]))
        for r in _rows("dealers.csv"):
            self.add_node(r["dealer_id"], "Dealer", name=r["name"], city=r["city"], region=r["region"])
        for r in _rows("loaners.csv"):
            self.nodes[r["dealer_id"]]["loaner_pool"] = int(r["loaners_available"])
        for r in _rows("customers.csv"):
            self.add_node(r["customer_id"], "Customer", name=r["name"], tier=r["tier"])
        for r in _rows("production_orders.csv"):
            d = date.fromisoformat(r["build_date"])
            self.add_node(r["production_order_id"], "ProductionOrder", plant_id=r["plant_id"], build_date=r["build_date"],
                          idx=date_to_idx(d), model=r["model"], units=int(r["units"]))
            self.add_edge(r["production_order_id"], "BUILT_AT", r["plant_id"])
            self.add_edge(r["production_order_id"], "OF_MODEL", r["model"])
        for r in _rows("customer_orders.csv"):
            self.add_node(r["customer_order_id"], "CustomerOrder", production_order_id=r["production_order_id"],
                          dealer_id=r["dealer_id"], customer_id=r["customer_id"], model=r["model"],
                          promised_date=r["promised_date"])
            self.add_edge(r["customer_order_id"], "FULFILLED_BY", r["production_order_id"])
            self.add_edge(r["customer_order_id"], "VIA_DEALER", r["dealer_id"])
            self.add_edge(r["customer_order_id"], "PLACED_BY", r["customer_id"])
        for r in _rows("part_lots.csv"):
            self.add_node(r["lot_id"], "Lot", part_id=r["part_id"], supplier_id=r["supplier_id"], plant_id=r["plant_id"],
                          first_build_date=r["first_build_date"], last_build_date=r["last_build_date"])
            self.add_edge(r["lot_id"], "OF_PART", r["part_id"])
            self.add_edge(r["lot_id"], "MADE_BY", r["supplier_id"])
            self.add_edge(r["lot_id"], "CONSUMED_AT", r["plant_id"])
            supplies.add((r["supplier_id"], r["part_id"]))
        for s, p in sorted(supplies):
            self.add_edge(s, "SUPPLIES", p)
        # derived: Lot INSTALLED_IN ProductionOrder (same plant, model uses the part, built inside the lot window)
        for lot in self.by_type["Lot"]:
            L = self.nodes[lot]
            for po in self.neighbors(L["plant_id"], "BUILT_AT", reverse=True):
                P = self.nodes[po]
                if L["first_build_date"] <= P["build_date"] <= L["last_build_date"] and \
                        L["part_id"] in self.neighbors(P["model"], "HAS_PART"):
                    self.add_edge(lot, "INSTALLED_IN", po)
        for r in _rows("finished_vehicles.csv"):
            self.add_node(r["vin"], "Vehicle", production_order_id=r["production_order_id"], model=r["model"],
                          plant_id=r["plant_id"], build_date=r["build_date"], dealer_id=r["dealer_id"],
                          status=r["status"], customer_id=r["customer_id"] or None)
            self.add_edge(r["vin"], "BUILT_FROM", r["production_order_id"])
            self.add_edge(r["vin"], "AT_DEALER", r["dealer_id"])
            if r["customer_id"]:
                self.add_edge(r["vin"], "OWNED_BY", r["customer_id"])
        # name index: aliases + ids + official names/cities
        for r in _rows("aliases.csv"):
            t = {"plant": "Plant", "part": "Part", "supplier": "Supplier"}.get(r["entity_type"], r["entity_type"].title())
            self.add_node("ALIAS:" + r["alias"], "Alias", text=r["alias"])
            self.add_edge("ALIAS:" + r["alias"], "REFERS_TO", r["entity_id"])
            self.alias_index[r["alias"].lower()] = (t, r["entity_id"])
        for t in ("Plant", "Part", "Supplier", "Model"):
            for nid in self.by_type[t]:
                n = self.nodes[nid]
                self.alias_index.setdefault(nid.lower(), (t, nid))
                if n.get("name"):
                    self.alias_index.setdefault(n["name"].lower(), (t, nid))
                if t == "Plant":
                    self.alias_index.setdefault(n["city"].split(",")[0].lower(), (t, nid))

    # ------------------------------------------------------------------ primitives
    def neighbors(self, nid, rel, reverse=False):
        edges = self.inn[nid] if reverse else self.out[nid]
        return [x for r, x in edges if r == rel]

    def stats(self):
        n_edges = sum(len(v) for v in self.out.values())
        return {"nodes": len(self.nodes), "edges": n_edges,
                "by_type": {t: len(v) for t, v in sorted(self.by_type.items())}}

    # ------------------------------------------------------------------ name resolution
    def resolve(self, text, want_type):
        """Resolve a free-text mention to an id of want_type. Returns (id, how) or (None, reason)."""
        if not text:
            return None, "not mentioned"
        t = text.strip().lower().strip(".,;:()\"'")
        cands = []
        for key, (typ, nid) in self.alias_index.items():
            if typ != want_type:
                continue
            k = key.lower()
            if t == k:
                return nid, f"exact match '{key}'"
            # containment either way, on word boundaries (handles 'the Huntsville line', 'Orbit Optics', plurals)
            if re.search(r"\b" + re.escape(k) + r"s?\b", t) or re.search(r"\b" + re.escape(t.rstrip("s")) + r"s?\b", k):
                cands.append((len(k), key, nid))
        ids = {c[2] for c in cands}
        if len(ids) == 1:
            best = max(cands)
            return best[2], f"matched '{best[1]}'"
        if len(ids) > 1:
            # prefer the longest key that is contained in the text
            inside = [c for c in cands if re.search(r"\b" + re.escape(c[1]) + r"s?\b", t)]
            if inside and len({c[2] for c in inside if c[0] == max(inside)[0]}) == 1:
                best = max(inside)
                return best[2], f"matched '{best[1]}' (longest of {sorted(ids)})"
            return None, f"ambiguous: {sorted(ids)}"
        return None, f"no {want_type} matches '{text}'"

    def find_mentions(self, text):
        """All alias/name mentions in a text (longest first, non-overlapping) -> list of (type, id, surface)."""
        found, taken = [], []
        low = text.lower()
        for key in sorted(self.alias_index, key=len, reverse=True):
            if len(key) < 3:
                continue
            for m in re.finditer(r"(?<![\w-])" + re.escape(key) + r"s?(?![\w-])", low):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                typ, nid = self.alias_index[key]
                found.append((m.start(), typ, nid, text[m.start():m.end()]))
        return [f[1:] for f in sorted(found)]

    # ------------------------------------------------------------------ traversals (records + path evidence)
    def find_shipments(self, part, plant, supplier=None, eta=None):
        out = []
        for sh in self.neighbors(part, "CARRIES", reverse=True):
            S = self.nodes[sh]
            if S["plant_id"] == plant and (supplier is None or S["supplier_id"] == supplier) and \
                    (eta is None or S["eta_date"] == eta):
                out.append(sh)
        return sorted(out)

    def supplier_delay_impact(self, shipment, new_eta):
        """Shipment -> Part -> Model (BOM) -> ProductionOrder (receiving plant, ETA <= build < new ETA, future)."""
        S = self.nodes[shipment]
        part, plant = S["part_id"], S["plant_id"]
        lo, hi = date_to_idx(date.fromisoformat(S["eta_date"])), date_to_idx(date.fromisoformat(new_eta))
        models = sorted(self.neighbors(part, "HAS_PART", reverse=True))
        built_here = set(self.neighbors(plant, "BUILDS"))
        pos = []
        for m in models:
            for po in self.neighbors(m, "OF_MODEL", reverse=True):
                P = self.nodes[po]
                if P["plant_id"] == plant and lo <= P["idx"] < hi and P["idx"] >= 0:
                    pos.append(po)
        path = [("Shipment", [shipment]), ("CARRIES", None), ("Part", [part]),
                ("HAS_PART (BOM, reverse)", None), ("Model", [m for m in models if m in built_here] or models),
                ("OF_MODEL (reverse) + BUILT_AT " + plant + f" + build in [{S['eta_date']}, {new_eta})", None),
                ("ProductionOrder", sorted(pos))]
        return sorted(pos, key=lambda p: (self.nodes[p]["idx"], p)), path

    def outage_impact(self, plant, start, end):
        """Plant -> ProductionOrder (BUILT_AT plant, start <= build <= end, future)."""
        lo, hi = date_to_idx(start), date_to_idx(end)
        pos = [po for po in self.neighbors(plant, "BUILT_AT", reverse=True)
               if lo <= self.nodes[po]["idx"] <= hi and self.nodes[po]["idx"] >= 0]
        pos.sort(key=lambda p: (self.nodes[p]["idx"], p))
        path = [("Plant", [plant]), (f"BUILT_AT (reverse) + build in [{start}, {end}]", None), ("ProductionOrder", pos)]
        return pos, path

    def downstream_orders(self, pos):
        """ProductionOrder -> CustomerOrder -> Dealer / Customer."""
        cos = sorted(co for po in pos for co in self.neighbors(po, "FULFILLED_BY", reverse=True))
        dealers = sorted({self.nodes[c]["dealer_id"] for c in cos})
        customers = sorted({self.nodes[c]["customer_id"] for c in cos})
        path = [("FULFILLED_BY (reverse)", None), ("CustomerOrder", cos),
                ("VIA_DEALER / PLACED_BY", None), ("Dealer", dealers), ("Customer", customers)]
        return cos, path

    def find_lots(self, part, plant, supplier=None, first=None, last=None):
        lots = []
        for lot in self.neighbors(part, "OF_PART", reverse=True):
            L = self.nodes[lot]
            if L["plant_id"] != plant or (supplier and L["supplier_id"] != supplier):
                continue
            if first and last and (L["last_build_date"] < first or L["first_build_date"] > last):
                continue
            lots.append(lot)
        return sorted(lots)

    def quality_impact(self, lot):
        """Lot -> ProductionOrder (INSTALLED_IN) -> Vehicle (BUILT_FROM) -> Dealer (unsold) / Customer (sold)."""
        L = self.nodes[lot]
        pos = sorted(self.neighbors(lot, "INSTALLED_IN"))
        vins = sorted(v for po in pos for v in self.neighbors(po, "BUILT_FROM", reverse=True)
                      if L["first_build_date"] <= self.nodes[v]["build_date"] <= L["last_build_date"])
        unsold = [v for v in vins if self.nodes[v]["status"] == "in_stock"]
        sold = [v for v in vins if self.nodes[v]["status"] == "sold"]
        path = [("Lot", [lot]), ("INSTALLED_IN", None), ("ProductionOrder", pos), ("BUILT_FROM (reverse)", None),
                ("Vehicle", vins), ("AT_DEALER (in_stock) / OWNED_BY (sold)", None),
                ("Dealer", sorted({self.nodes[v]["dealer_id"] for v in unsold})),
                ("Customer", sorted({self.nodes[v]["customer_id"] for v in sold}))]
        return unsold, sold, path

    # ------------------------------------------------------------------ export
    def export_ontology(self, path_json, path_mmd):
        Path = __import__("pathlib").Path
        Path(path_json).write_text(json.dumps({**ONTOLOGY, "stats": self.stats()}, indent=2))
        lines = ["classDiagram"]
        for c, d in ONTOLOGY["classes"].items():
            lines.append(f"  class {c}")
        for s, r, d in ONTOLOGY["relations"]:
            if d != "*":
                lines.append(f"  {s} --> {d} : {r}")
        Path(path_mmd).write_text("\n".join(lines) + "\n")
