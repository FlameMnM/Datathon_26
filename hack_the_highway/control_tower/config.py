"""Paths and constants for the Wacky Wheels control tower."""
import os
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                      # hack_the_highway/
PKG = ROOT / "hack_the_highway_student_package" / "hack_the_highway_package"
DATA = PKG / "data"
RUNS = ROOT / "runs"

TODAY = date(2027, 3, 8)          # handbook: today is Monday 2027-03-08 = working-day index 0
FIRST_IDX, LAST_IDX = -10, 19     # two past weeks (built) + four future weeks, Mon-Fri
TIER_RANK = {"Platinum": 0, "Gold": 1, "Standard": 2}
LOANER_TIERS = ("Platinum", "Gold")

# LLM settings (see nlu.py). Model choice is the team's decision; override with HTW_MODEL.
LLM_MODEL = os.environ.get("HTW_MODEL", "claude-opus-5-5")
# Handbook rule 3: if no same-plant slot, transfer "to another plant that builds the model on any day with a free slot".
# earliest = earliest such day from today (literal reading); after_restart = not before the disruption ends.
TRANSFER_POLICY = os.environ.get("HTW_TRANSFER", "earliest")
LLM_BACKEND = os.environ.get("HTW_LLM", "auto")   # auto | claude | rules


def idx_to_date(i):
    return TODAY + timedelta(days=7 * (i // 5) + i % 5)


def date_to_idx(d):
    days = (d - TODAY).days
    if days % 7 > 4:
        raise ValueError(f"{d} is a weekend")
    return 5 * (days // 7) + days % 7


def next_working_idx(d):
    """Index of the first working day on or after date d."""
    while d.weekday() > 4:
        d += timedelta(days=1)
    return date_to_idx(d)
