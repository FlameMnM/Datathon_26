# Event summaries - what the control tower did, and why

Run `run1`, model `gemini-3.5-flash-lite` (Google Gemini API (generativelanguage.googleapis.com v1beta)). Generated from the per-event traces by `control_tower/make_summaries.py`; the full machine-readable record of each event is in `runs/run1/traces/E-xx.json`, and every tool call is in `runs/run1/action_log.jsonl`.

| Event | Source | Category | Decision | Write actions | Failed calls |
|---|---|---|---|---|---|
| E-01 | daily digest | Plant outage + Supplier delay | Acted per handbook | 81 | 0 |
| E-02 | ops chat | Unconfirmed report | Escalated to a human, no change | 0 | 0 |
| E-03 | quality bulletin | Quality hold | Acted per handbook | 188 | 0 |
| E-04 | ops chat | Plant outage | Acted per handbook | 59 | 0 |
| E-05 | news wire | Noise | No action (not operational) | 0 | 0 |
| E-06 | news wire | Noise | No action (not operational) | 0 | 0 |
| E-07 | news wire | Noise | No action (not operational) | 0 | 0 |
| E-08 | plant alert | Plant outage | Acted per handbook | 46 | 0 |
| E-09 | supplier email | Supplier delay | Acted per handbook | 42 | 0 |
| E-10 | ops chat | Supplier delay | Acted per handbook | 28 | 0 |
| E-11 | ops chat | Plant outage | Acted per handbook | 62 | 0 |
| E-12 | supplier email | Supplier delay | Acted per handbook | 27 | 0 |
| E-13 | ops chat | Unconfirmed report | Escalated to a human, no change | 0 | 0 |
| E-14 | ops chat | Quality hold | Acted per handbook | 131 | 0 |
| E-15 | quality bulletin | Quality hold | Acted per handbook | 102 | 0 |

---

## E-01 - Plant outage + Supplier delay

> Morning digest, two items. (1) San Antonio update: a software outage on the line. We're dark from next Friday through Monday Mar 22; the line restarts the following working day. (2) Quick heads up from Quanta: the infotainment chip run for Princeton can't make it next Monday. Carrier says next Friday now (severe weather on the route).

*Received 2027-03-08T07:17:00 via daily digest. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** 14 builds rescheduled. 122 customer deliveries affected at 41 dealers; 26 Platinum/Gold customers got a loaner, 4 entitled customers are waiting (dealer pool empty).

**Why.**
- I1 (Plant outage): One of our plants cannot build for a range of days due to a software outage. Grounded: plant 'San Antonio' -> P_TX (exact match 'san antonio'). Dates: 'next Friday' -> 2027-03-19 (relative: 'next Friday' from Monday 2027-03-08); 'Monday Mar 22' -> 2027-03-22 (month/day).
- I2 (Supplier delay): An inbound part shipment from a supplier to one of our plants will arrive later than planned. Grounded: part 'infotainment chip' -> CHIP-IVI (exact match 'infotainment chip'); plant 'Princeton' -> P_IN (exact match 'princeton'); supplier 'Quanta' -> SUP-12 (exact match 'quanta'). Dates: 'next Monday' -> 2027-03-15 (relative: 'next Monday' from Monday 2027-03-08); 'next Friday' -> 2027-03-19 (relative: 'next Friday' from Monday 2027-03-08).
- *SupplyChainPlanner:* [I1] P_TX builds 6 future production orders during the outage: ['PO-0058', 'PO-0059', 'PO-0060', 'PO-0061', 'PO-0062', 'PO-0063']; restart day 2027-03-23
- *SupplyChainPlanner:* [I2] BOM traversal: CHIP-IVI is used by ['NOVA', 'PIONEER', 'SUMMIT', 'TERRA', 'VOYAGER']; at P_IN 8 future production orders build in [2027-03-15, 2027-03-19): ['PO-0316', 'PO-0318', 'PO-0319', 'PO-0321', 'PO-0322', 'PO-0324', 'PO-0325', 'PO-0327']
- *SupplyChainPlanner:* PO-0316 (VOYAGER, was 2027-03-15) -> P_IN 2027-03-19 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0318 (NOVA, was 2027-03-15) -> P_IN 2027-03-19 (same plant, first free slot)

**Graph path(s).**
- I1 P_TX down 2027-03-19 .. 2027-03-22: **Plant** `P_TX` -BUILT_AT (reverse) + build in [2027-03-19, 2027-03-22]-> **ProductionOrder** (6) -FULFILLED_BY (reverse)-> **CustomerOrder** (55) -VIA_DEALER / PLACED_BY-> **Dealer** (27) **Customer** (55)
- I2 SH-0190 (CHIP-IVI -> P_IN) late: 2027-03-15 -> 2027-03-19: **Shipment** `SH-0190` -CARRIES-> **Part** `CHIP-IVI` -HAS_PART (BOM, reverse)-> **Model** (2) -OF_MODEL (reverse) + BUILT_AT P_IN + build in [2027-03-15, 2027-03-19)-> **ProductionOrder** (8) -FULFILLED_BY (reverse)-> **CustomerOrder** (67) -VIA_DEALER / PLACED_BY-> **Dealer** (29) **Customer** (67)

**Actions.** `crm.assign_loaner` x26, `dealer.notify` x41, `mes.reschedule` x14.

**Changed:** 14 production orders moved (14 same-plant reschedules, 0 transfers); 14 now build later (avg 4.6 days). 41 dealers told about exactly their affected orders (one message each); 26 loaners assigned.

**Deliberately left alone:** 54 of 60 future production orders at P_TX were not affected and were not moved. 4 other orders at P_IN in the delay window (2027-03-15 to 2027-03-19) build models without CHIP-IVI (['RANGER']), so they were not moved. 52 of 60 future production orders at P_IN were not affected and were not moved. 92 Standard-tier customers: dealer informed, no loaner (policy rule 5). 4 entitled customers did not get a loaner because the dealer's pool ran out: ['CO-00261', 'CO-00284', 'CO-01836', 'CO-01855'] 9 dealers have no affected orders and were not contacted. No lots held, no vehicles stopped from sale (not part of a delay or outage).

---

## E-02 - Unconfirmed report

> Rumor from a supplier rep: something may be off with a seat material coming up. No dates, no details, nothing confirmed.

*Received 2027-03-08T07:34:00 via ops chat. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** No operational impact. Handed to a human for review.

**Why.**
- I1 (Unconfirmed report): It is an unconfirmed rumor missing specifics and dates.

**Actions.** `escalate` x1.

**Changed:** Nothing. No production order, dealer, customer, loaner, lot or vehicle record was written. One escalation to a human (not an operational change).

**Deliberately left alone:** All 540 production orders, 50 dealers, all loaner pools and all 364 lots are untouched.

---

## E-03 - Quality hold

> Quality alert: Kestrel says the infotainment chip sent to Jackson had a faulty weld. Anything built Thursday Feb 25 through last Monday is suspect.

*Received 2027-03-08T07:51:00 via quality bulletin. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** Lot LOT-0130 held; 110 unsold vehicles stopped from sale, 150 owners contacted.

**Why.**
- I1 (Quality hold): A specific part from a named supplier at a specific plant is defective for a stated date range. Grounded: part 'infotainment chip' -> CHIP-IVI (exact match 'infotainment chip'); plant 'Jackson' -> P_MS (exact match 'jackson'); supplier 'Kestrel' -> SUP-06 (exact match 'kestrel'). Dates: 'Thursday Feb 25' -> 2027-02-25 (month/day); 'last Monday' -> 2027-03-01 (relative: 'last Monday' from Monday 2027-03-08).
- *QualityAgent:* LOT-0130: 260 affected vehicles (110 in stock, 150 sold)

**Graph path(s).**
- I1 LOT-0130 (CHIP-IVI from SUP-06 at P_MS, built 2027-02-25..2027-03-01): **Lot** `LOT-0130` -INSTALLED_IN-> **ProductionOrder** (9) -BUILT_FROM (reverse)-> **Vehicle** (260) -AT_DEALER (in_stock) / OWNED_BY (sold)-> **Dealer** (37) **Customer** (150)

**Actions.** `crm.notify_customer` x150, `dealer.stop_sale` x37, `quality.hold_lot` x1.

**Changed:** Lot LOT-0130 on quality hold; stop-sale on 110 in-stock vehicles at 37 dealers; 150 owners notified.

**Deliberately left alone:** 3 other CHIP-IVI lots at P_MS (LOT-0129, LOT-0131, LOT-0132) are outside the suspect build window and were not held. Every model P_MS built on those days uses CHIP-IVI, so all 260 vehicles from the window are in scope; vehicles from other days and plants were not touched. No production orders were rescheduled and no loaners assigned (not part of a quality hold).

---

## E-04 - Plant outage

> The Kentucky plant update: a software outage on the line. We're dark from next Tuesday through next Thursday; the line restarts the following working day.

*Received 2027-03-08T08:08:00 via ops chat. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** 9 builds rescheduled. 79 customer deliveries affected at 32 dealers; 18 Platinum/Gold customers got a loaner, 4 entitled customers are waiting (dealer pool empty).

**Why.**
- I1 (Plant outage): The plant is unable to build for a specific range of days due to a software outage. Grounded: plant 'Kentucky' -> P_KY (matched 'the kentucky plant'). Dates: 'next Tuesday' -> 2027-03-16 (relative: 'next Tuesday' from Monday 2027-03-08); 'next Thursday' -> 2027-03-18 (relative: 'next Thursday' from Monday 2027-03-08).
- *SupplyChainPlanner:* [I1] P_KY builds 9 future production orders during the outage: ['PO-0139', 'PO-0140', 'PO-0141', 'PO-0142', 'PO-0143', 'PO-0144', 'PO-0145', 'PO-0146', 'PO-0147']; restart day 2027-03-19
- *SupplyChainPlanner:* PO-0139 (PIONEER, was 2027-03-16) -> P_KY 2027-03-19 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0140 (SUMMIT, was 2027-03-16) -> P_KY 2027-03-19 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0141 (VOYAGER, was 2027-03-16) -> P_KY 2027-03-22 (same plant, first free slot)

**Graph path(s).**
- I1 P_KY down 2027-03-16 .. 2027-03-18: **Plant** `P_KY` -BUILT_AT (reverse) + build in [2027-03-16, 2027-03-18]-> **ProductionOrder** (9) -FULFILLED_BY (reverse)-> **CustomerOrder** (79) -VIA_DEALER / PLACED_BY-> **Dealer** (32) **Customer** (79)

**Actions.** `crm.assign_loaner` x18, `dealer.notify` x32, `mes.reschedule` x9.

**Changed:** 9 production orders moved (9 same-plant reschedules, 0 transfers); 9 now build later (avg 5.3 days). 32 dealers told about exactly their affected orders (one message each); 18 loaners assigned.

**Deliberately left alone:** 51 of 60 future production orders at P_KY were not affected and were not moved. 57 Standard-tier customers: dealer informed, no loaner (policy rule 5). 4 entitled customers did not get a loaner because the dealer's pool ran out: ['CO-00718', 'CO-00744', 'CO-00763', 'CO-00774'] 18 dealers have no affected orders and were not contacted. No lots held, no vehicles stopped from sale (not part of a delay or outage).

---

## E-05 - Noise

> Dealer Amarillo, TX asks where to find the new brochure PDF.

*Received 2027-03-08T08:25:00 via news wire. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** No operational impact.

**Why.**
- I1 (Noise): It is about marketing materials rather than our plant operations or supply chain.

**Actions.** none.

**Changed:** Nothing. No production order, dealer, customer, loaner, lot or vehicle record was written.

**Deliberately left alone:** All 540 production orders, 50 dealers, all loaner pools and all 364 lots are untouched.

---

## E-06 - Noise

> Dealer Little Rock, AR asks where to find the new brochure PDF.

*Received 2027-03-08T08:42:00 via news wire. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** No operational impact.

**Why.**
- I1 (Noise): This is a general inquiry about marketing materials and is not related to manufacturing operations.

**Actions.** none.

**Changed:** Nothing. No production order, dealer, customer, loaner, lot or vehicle record was written.

**Deliberately left alone:** All 540 production orders, 50 dealers, all loaner pools and all 364 lots are untouched.

---

## E-07 - Noise

> Dealer Pensacola, FL asks where to find the new brochure PDF.

*Received 2027-03-08T08:59:00 via news wire. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** No operational impact.

**Why.**
- I1 (Noise): This is a general inquiry about brochures, which is unrelated to our internal plant operations.

**Actions.** none.

**Changed:** Nothing. No production order, dealer, customer, loaner, lot or vehicle record was written.

**Deliberately left alone:** All 540 production orders, 50 dealers, all loaner pools and all 364 lots are untouched.

---

## E-08 - Plant outage

> Plant operations alert: Wacky Wheels Lexington (P_KY) lost production to a power failure. Production is stopped from 2027-03-24 through 2027-03-25 inclusive.

*Received 2027-03-08T09:16:00 via plant alert. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** 6 builds rescheduled. 56 customer deliveries affected at 25 dealers; 15 Platinum/Gold customers got a loaner.

**Why.**
- I1 (Plant outage): The plant cannot build for a range of days due to an outage. Grounded: plant 'Lexington' -> P_KY (exact match 'lexington'). Dates: '2027-03-24' -> 2027-03-24 (ISO date); '2027-03-25' -> 2027-03-25 (ISO date).
- *SupplyChainPlanner:* [I1] P_KY builds 6 future production orders during the outage: ['PO-0157', 'PO-0158', 'PO-0159', 'PO-0160', 'PO-0161', 'PO-0162']; restart day 2027-03-26
- *SupplyChainPlanner:* PO-0157 (PIONEER, was 2027-03-24) -> P_KY 2027-03-26 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0158 (SUMMIT, was 2027-03-24) -> P_KY 2027-03-26 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0159 (VOYAGER, was 2027-03-24) -> P_KY 2027-03-29 (same plant, first free slot)

**Graph path(s).**
- I1 P_KY down 2027-03-24 .. 2027-03-25: **Plant** `P_KY` -BUILT_AT (reverse) + build in [2027-03-24, 2027-03-25]-> **ProductionOrder** (6) -FULFILLED_BY (reverse)-> **CustomerOrder** (56) -VIA_DEALER / PLACED_BY-> **Dealer** (25) **Customer** (56)

**Actions.** `crm.assign_loaner` x15, `dealer.notify` x25, `mes.reschedule` x6.

**Changed:** 6 production orders moved (6 same-plant reschedules, 0 transfers); 6 now build later (avg 3.8 days). 25 dealers told about exactly their affected orders (one message each); 15 loaners assigned.

**Deliberately left alone:** 54 of 60 future production orders at P_KY were not affected and were not moved. 41 Standard-tier customers: dealer informed, no loaner (policy rule 5). 25 dealers have no affected orders and were not contacted. No lots held, no vehicles stopped from sale (not part of a delay or outage).

---

## E-09 - Supplier delay

> Quick heads up from Orbit: the surround-view cameras run for Huntsville can't make it Monday Mar 29. Carrier says Thursday Apr 1 now (a customs inspection).

*Received 2027-03-08T09:33:00 via supplier email. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** 6 builds rescheduled. 54 customer deliveries affected at 26 dealers; 10 Platinum/Gold customers got a loaner, 2 entitled customers are waiting (dealer pool empty).

**Why.**
- I1 (Supplier delay): An inbound part shipment from a supplier to a specific plant will arrive later than planned with specific dates and part mentioned. Grounded: part 'surround-view cameras' -> CAM-360 (exact match 'surround-view cameras'); plant 'Huntsville' -> P_AL (exact match 'huntsville'); supplier 'Orbit' -> SUP-11 (exact match 'orbit'). Dates: 'Monday Mar 29' -> 2027-03-29 (month/day); 'Thursday Apr 1' -> 2027-04-01 (month/day).
- *SupplyChainPlanner:* [I1] BOM traversal: CAM-360 is used by ['SUMMIT', 'VOYAGER']; at P_AL 6 future production orders build in [2027-03-29, 2027-04-01): ['PO-0436', 'PO-0438', 'PO-0439', 'PO-0441', 'PO-0442', 'PO-0444']
- *SupplyChainPlanner:* PO-0436 (SUMMIT, was 2027-03-29) -> P_AL 2027-04-01 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0438 (VOYAGER, was 2027-03-29) -> P_AL 2027-04-01 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0439 (SUMMIT, was 2027-03-30) -> P_AL 2027-04-02 (same plant, first free slot)

**Graph path(s).**
- I1 SH-0256 (CAM-360 -> P_AL) late: 2027-03-29 -> 2027-04-01: **Shipment** `SH-0256` -CARRIES-> **Part** `CAM-360` -HAS_PART (BOM, reverse)-> **Model** (2) -OF_MODEL (reverse) + BUILT_AT P_AL + build in [2027-03-29, 2027-04-01)-> **ProductionOrder** (6) -FULFILLED_BY (reverse)-> **CustomerOrder** (54) -VIA_DEALER / PLACED_BY-> **Dealer** (26) **Customer** (54)

**Actions.** `crm.assign_loaner` x10, `dealer.notify` x26, `mes.reschedule` x4, `mes.transfer` x2.

**Changed:** 6 production orders moved (4 same-plant reschedules, 2 transfers); 4 now build later (avg 3.0 days), 2 earlier at another plant. 26 dealers told about exactly their affected orders (one message each); 10 loaners assigned.

**Deliberately left alone:** 3 other orders at P_AL in the delay window (2027-03-29 to 2027-04-01) build models without CAM-360 (['NOVA']), so they were not moved. 54 of 60 future production orders at P_AL were not affected and were not moved. 42 Standard-tier customers: dealer informed, no loaner (policy rule 5). 2 entitled customers did not get a loaner because the dealer's pool ran out: ['CO-02568', 'CO-02644'] 24 dealers have no affected orders and were not contacted. No lots held, no vehicles stopped from sale (not part of a delay or outage).

---

## E-10 - Supplier delay

> Huntsville line is waiting on 60 kWh battery packs from Lumen. Original drop was Monday Mar 29, new ETA is Thursday Apr 1 because of a truck breakdown.

*Received 2027-03-08T09:50:00 via ops chat. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** 3 builds rescheduled. 29 customer deliveries affected at 18 dealers; 7 Platinum/Gold customers got a loaner.

**Why.**
- I1 (Supplier delay): An inbound part shipment will arrive later than planned with specific dates and plant. Grounded: part '60 kWh battery packs' -> BATT-60 (exact match '60 kwh battery packs'); plant 'Huntsville' -> P_AL (exact match 'huntsville'); supplier 'Lumen' -> SUP-18 (exact match 'lumen'). Dates: 'Monday Mar 29' -> 2027-03-29 (month/day); 'Thursday Apr 1' -> 2027-04-01 (month/day).
- *SupplyChainPlanner:* [I1] BOM traversal: BATT-60 is used by ['NOVA']; at P_AL 3 future production orders build in [2027-03-29, 2027-04-01): ['PO-0437', 'PO-0440', 'PO-0443']
- *SupplyChainPlanner:* PO-0437 (NOVA, was 2027-03-29) -> P_AL 2027-04-01 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0440 (NOVA, was 2027-03-30) -> P_AL 2027-04-01 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0443 (NOVA, was 2027-03-31) -> P_AL 2027-04-02 (same plant, first free slot)

**Graph path(s).**
- I1 SH-0248 (BATT-60 -> P_AL) late: 2027-03-29 -> 2027-04-01: **Shipment** `SH-0248` -CARRIES-> **Part** `BATT-60` -HAS_PART (BOM, reverse)-> **Model** `NOVA` -OF_MODEL (reverse) + BUILT_AT P_AL + build in [2027-03-29, 2027-04-01)-> **ProductionOrder** (3) -FULFILLED_BY (reverse)-> **CustomerOrder** (29) -VIA_DEALER / PLACED_BY-> **Dealer** (18) **Customer** (29)

**Actions.** `crm.assign_loaner` x7, `dealer.notify` x18, `mes.reschedule` x3.

**Changed:** 3 production orders moved (3 same-plant reschedules, 0 transfers); 3 now build later (avg 2.3 days). 18 dealers told about exactly their affected orders (one message each); 7 loaners assigned.

**Deliberately left alone:** 6 other orders at P_AL in the delay window (2027-03-29 to 2027-04-01) build models without BATT-60 (['SUMMIT', 'VOYAGER']), so they were not moved. 57 of 60 future production orders at P_AL were not affected and were not moved. 22 Standard-tier customers: dealer informed, no loaner (policy rule 5). 32 dealers have no affected orders and were not contacted. No lots held, no vehicles stopped from sale (not part of a delay or outage).

---

## E-11 - Plant outage

> Jackson update: a power failure. We're dark from Friday Mar 26 through Tuesday Mar 30; the line restarts the following working day.

*Received 2027-03-08T10:07:00 via ops chat. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** 9 builds rescheduled. 79 customer deliveries affected at 36 dealers; 17 Platinum/Gold customers got a loaner, 1 entitled customer is waiting (dealer pool empty).

**Why.**
- I1 (Plant outage): The plant cannot build for a range of days due to a power failure. Grounded: plant 'Jackson' -> P_MS (exact match 'jackson'). Dates: 'Friday Mar 26' -> 2027-03-26 (month/day); 'Tuesday Mar 30' -> 2027-03-30 (month/day).
- *SupplyChainPlanner:* [I1] P_MS builds 9 future production orders during the outage: ['PO-0253', 'PO-0254', 'PO-0255', 'PO-0256', 'PO-0257', 'PO-0258', 'PO-0259', 'PO-0260', 'PO-0261']; restart day 2027-03-31
- *SupplyChainPlanner:* PO-0253 (PIONEER, was 2027-03-26) -> P_MS 2027-03-31 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0254 (NOVA, was 2027-03-26) -> P_MS 2027-03-31 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0255 (TERRA, was 2027-03-26) -> P_MS 2027-04-01 (same plant, first free slot)

**Graph path(s).**
- I1 P_MS down 2027-03-26 .. 2027-03-30: **Plant** `P_MS` -BUILT_AT (reverse) + build in [2027-03-26, 2027-03-30]-> **ProductionOrder** (9) -FULFILLED_BY (reverse)-> **CustomerOrder** (79) -VIA_DEALER / PLACED_BY-> **Dealer** (36) **Customer** (79)

**Actions.** `crm.assign_loaner` x17, `dealer.notify` x36, `mes.reschedule` x6, `mes.transfer` x3.

**Changed:** 9 production orders moved (6 same-plant reschedules, 3 transfers); 6 now build later (avg 4.5 days), 3 earlier at another plant. 36 dealers told about exactly their affected orders (one message each); 17 loaners assigned.

**Deliberately left alone:** 51 of 60 future production orders at P_MS were not affected and were not moved. 61 Standard-tier customers: dealer informed, no loaner (policy rule 5). 1 entitled customer did not get a loaner because the dealer's pool ran out: ['CO-01529'] 14 dealers have no affected orders and were not contacted. No lots held, no vehicles stopped from sale (not part of a delay or outage).

---

## E-12 - Supplier delay

> Subject: Shipment SH-0182 delayed
From: dispatch@orbit.example

Our shipment SH-0182 of CAM-360 (surround-view camera kit) for Wacky Wheels Princeton (P_IN) was due 2027-03-15 and will now arrive 2027-03-18. Reason: port congestion.

*Received 2027-03-08T10:24:00 via supplier email. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** 3 builds rescheduled. 26 customer deliveries affected at 14 dealers; 10 Platinum/Gold customers got a loaner.

**Why.**
- I1 (Supplier delay): An inbound part shipment from a supplier is arriving later than planned with specific details. Grounded: part 'CAM-360' -> CAM-360 (exact match 'cam-360'); plant 'Princeton' -> P_IN (exact match 'princeton'); supplier 'orbit' -> SUP-11 (exact match 'orbit'). Dates: '2027-03-15' -> 2027-03-15 (ISO date); '2027-03-18' -> 2027-03-18 (ISO date).
- *SupplyChainPlanner:* [I1] BOM traversal: CAM-360 is used by ['SUMMIT', 'VOYAGER']; at P_IN 3 future production orders build in [2027-03-15, 2027-03-18): ['PO-0316', 'PO-0319', 'PO-0322']
- *SupplyChainPlanner:* PO-0316 (VOYAGER, was 2027-03-15) -> P_IN 2027-03-18 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0319 (VOYAGER, was 2027-03-16) -> P_IN 2027-03-18 (same plant, first free slot)
- *SupplyChainPlanner:* PO-0322 (VOYAGER, was 2027-03-17) -> P_IN 2027-03-19 (same plant, first free slot)

**Graph path(s).**
- I1 SH-0182 (CAM-360 -> P_IN) late: 2027-03-15 -> 2027-03-18: **Shipment** `SH-0182` -CARRIES-> **Part** `CAM-360` -HAS_PART (BOM, reverse)-> **Model** `VOYAGER` -OF_MODEL (reverse) + BUILT_AT P_IN + build in [2027-03-15, 2027-03-18)-> **ProductionOrder** (3) -FULFILLED_BY (reverse)-> **CustomerOrder** (26) -VIA_DEALER / PLACED_BY-> **Dealer** (14) **Customer** (26)

**Actions.** `crm.assign_loaner` x10, `dealer.notify` x14, `mes.reschedule` x3.

**Changed:** 3 production orders moved (3 same-plant reschedules, 0 transfers); 3 now build later (avg 2.3 days). 14 dealers told about exactly their affected orders (one message each); 10 loaners assigned.

**Deliberately left alone:** 6 other orders at P_IN in the delay window (2027-03-15 to 2027-03-18) build models without CAM-360 (['NOVA', 'RANGER']), so they were not moved. 57 of 60 future production orders at P_IN were not affected and were not moved. 16 Standard-tier customers: dealer informed, no loaner (policy rule 5). 36 dealers have no affected orders and were not contacted. No lots held, no vehicles stopped from sale (not part of a delay or outage).

---

## E-13 - Unconfirmed report

> Someone on the floor mentioned a possible shortage of something electronic sometime soon. Not confirmed by anyone.

*Received 2027-03-08T10:41:00 via ops chat. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** No operational impact. Handed to a human for review.

**Why.**
- I1 (Unconfirmed report): The report is a rumor on the floor, unconfirmed, and lacks specific details such as the exact plant, part, or dates.

**Actions.** `escalate` x1.

**Changed:** Nothing. No production order, dealer, customer, loaner, lot or vehicle record was written. One escalation to a human (not an operational change).

**Deliberately left alone:** All 540 production orders, 50 dealers, all loaner pools and all 364 lots are untouched.

---

## E-14 - Quality hold

> Quality alert: Atlas says the heavy-duty wiring harness sent to the Texas plant had a moisture contamination. Anything built Monday Feb 22 through Wednesday Feb 24 is suspect.

*Received 2027-03-08T10:58:00 via ops chat. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** Lot LOT-0017 held; 86 unsold vehicles stopped from sale, 94 owners contacted.

**Why.**
- I1 (Quality hold): A specific part from a named supplier at a specific plant is defective for a defined date range. Grounded: part 'heavy-duty wiring harness' -> HARN-B (exact match 'heavy-duty wiring harness'); plant 'Texas' -> P_TX (matched 'the texas plant'); supplier 'Atlas' -> SUP-07 (exact match 'atlas'). Dates: 'Monday Feb 22' -> 2027-02-22 (month/day); 'Wednesday Feb 24' -> 2027-02-24 (month/day).
- *QualityAgent:* LOT-0017: 180 affected vehicles (86 in stock, 94 sold)

**Graph path(s).**
- I1 LOT-0017 (HARN-B from SUP-07 at P_TX, built 2027-02-22..2027-02-24): **Lot** `LOT-0017` -INSTALLED_IN-> **ProductionOrder** (6) -BUILT_FROM (reverse)-> **Vehicle** (180) -AT_DEALER (in_stock) / OWNED_BY (sold)-> **Dealer** (36) **Customer** (94)

**Actions.** `crm.notify_customer` x94, `dealer.stop_sale` x36, `quality.hold_lot` x1.

**Changed:** Lot LOT-0017 on quality hold; stop-sale on 86 in-stock vehicles at 36 dealers; 94 owners notified.

**Deliberately left alone:** 3 other HARN-B lots at P_TX (LOT-0018, LOT-0019, LOT-0020) are outside the suspect build window and were not held. 88 vehicles built at P_TX on the same days use models without HARN-B and were not touched. No production orders were rescheduled and no loaners assigned (not part of a quality hold).

---

## E-15 - Quality hold

> QUALITY BULLETIN QB-368: Lot LOT-0191 (CHIP-IVI, supplier SUP-12) installed at P_IN on vehicles built 2027-03-02 to 2027-03-03 has a faulty weld. Place the lot on hold, stop sale of unsold vehicles and contact owners.

*Received 2027-03-08T11:15:00 via quality bulletin. Understood by `gemini-3.5-flash-lite`; rule-based cross-check agrees.*

**Summary.** Lot LOT-0191 held; 40 unsold vehicles stopped from sale, 76 owners contacted.

**Why.**
- I1 (Quality hold): A specific part lot used at a plant has a defect affecting vehicles built in a specific date range. Grounded: part 'CHIP-IVI' -> CHIP-IVI (exact match 'chip-ivi'); plant 'P_IN' -> P_IN (exact match 'p_in'); supplier 'SUP-12' -> SUP-12 (exact match 'sup-12'). Dates: '2027-03-02' -> 2027-03-02 (ISO date); '2027-03-03' -> 2027-03-03 (ISO date).
- *QualityAgent:* LOT-0191: 116 affected vehicles (40 in stock, 76 sold)

**Graph path(s).**
- I1 LOT-0191 (CHIP-IVI from SUP-12 at P_IN, built 2027-03-02..2027-03-03): **Lot** `LOT-0191` -INSTALLED_IN-> **ProductionOrder** (4) -BUILT_FROM (reverse)-> **Vehicle** (116) -AT_DEALER (in_stock) / OWNED_BY (sold)-> **Dealer** (25) **Customer** (76)

**Actions.** `crm.notify_customer` x76, `dealer.stop_sale` x25, `quality.hold_lot` x1.

**Changed:** Lot LOT-0191 on quality hold; stop-sale on 40 in-stock vehicles at 25 dealers; 76 owners notified.

**Deliberately left alone:** 3 other CHIP-IVI lots at P_IN (LOT-0189, LOT-0190, LOT-0192) are outside the suspect build window and were not held. 69 vehicles built at P_IN on the same days use models without CHIP-IVI and were not touched. No production orders were rescheduled and no loaners assigned (not part of a quality hold).
