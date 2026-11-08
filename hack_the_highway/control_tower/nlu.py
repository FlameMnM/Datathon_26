"""Event understanding: LLM extraction (Claude) + deterministic grounding.

Split of work:
  * LLM  : classify the message, split multi-incident messages, copy the mentions and date phrases verbatim.
  * Code : resolve every mention to an id through the knowledge graph (aliases), turn date phrases into ISO dates
           (relative dates are computed from TODAY in code, never by the model), and validate that the incident is
           specific enough to act on. Anything that fails validation becomes 'unconfirmed' -> escalate.
A rule-based extractor implements the same schema; it is the offline fallback and an independent cross-check.
"""
import json
import re
from datetime import date, timedelta

from config import LLM_BACKEND, LLM_MODEL, TODAY

KINDS = ["supplier_delay", "plant_outage", "quality_hold", "noise", "unconfirmed"]
FIELDS = ["supplier", "part", "plant", "shipment_id", "lot_id", "date_1", "date_2"]

SCHEMA = {
    "type": "object",
    "properties": {
        "incidents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": KINDS},
                    "summary": {"type": "string"},
                    "supplier": {"type": "string"}, "part": {"type": "string"}, "plant": {"type": "string"},
                    "shipment_id": {"type": "string"}, "lot_id": {"type": "string"},
                    "date_1": {"type": "string"}, "date_2": {"type": "string"},
                    "confirmed_and_specific": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["kind", "summary", *FIELDS, "confirmed_and_specific", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["incidents"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You read operational messages for the control tower of Wacky Wheels Co., a just-in-time automaker \
(plants: San Antonio TX, Lexington KY, Jackson MS, Princeton IN, Huntsville AL, Greensboro NC).

Split the message into incidents (a digest may contain several) and classify each one:
- supplier_delay: an inbound part shipment to one of our plants will arrive later than planned.
- plant_outage: one of our plants cannot build for a range of days.
- quality_hold: a lot of a part used at one of our plants is defective / suspect.
- noise: not about our operations (brochures, far-away weather, price lists, general news).
- unconfirmed: possibly about us but a rumor, unconfirmed, or missing the specifics needed to act \
(which part, which plant, which dates).

For each incident copy text from the message VERBATIM (do not normalise, do not compute dates):
- supplier, part, plant: the words used in the message ("" if not mentioned).
- shipment_id, lot_id: ids like SH-0182 or LOT-0191 if written ("" otherwise).
- date_1, date_2: the two date phrases exactly as written ("next Friday", "Monday Mar 29", "2027-03-15", \
"last Monday"). supplier_delay: date_1 = original/planned arrival, date_2 = new arrival. plant_outage: \
date_1 = first day down, date_2 = last day down. quality_hold: date_1 = first build date, date_2 = last build \
date. Use "" when absent.
- confirmed_and_specific: false for rumors / hearsay / missing specifics.
- summary: one short sentence. reason: why you chose this kind."""


# ====================================================================== LLM extractor
class ClaudeExtractor:
    name = "claude"

    def __init__(self, model=LLM_MODEL):
        import anthropic  # official SDK
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def extract(self, text):
        a = self.anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Message received {TODAY.isoformat()}:\n\n{text}"}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.APIConnectionError as e:
            raise RuntimeError(f"LLM connection error: {e}") from e
        except a.AuthenticationError as e:
            raise RuntimeError(f"LLM authentication failed: {e.message}") from e
        except a.RateLimitError as e:
            raise RuntimeError(f"LLM rate limited: {e}") from e
        except a.APIStatusError as e:
            raise RuntimeError(f"LLM API error {e.status_code}: {e.message}") from e
        self.usage["calls"] += 1
        self.usage["input_tokens"] += resp.usage.input_tokens
        self.usage["output_tokens"] += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            raise RuntimeError("LLM refused the request")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("LLM output truncated")
        txt = next(b.text for b in resp.content if b.type == "text")
        return json.loads(txt)["incidents"]


# ====================================================================== Gemini extractor (stdlib REST)
GEMINI_KEY_FILE = __import__("pathlib").Path.home() / ".config" / "hack_the_highway" / "gemini_api_key"


def _gemini_schema(s):
    """JSON Schema -> Gemini responseSchema (OpenAPI subset: no additionalProperties)."""
    if isinstance(s, dict):
        return {k: _gemini_schema(v) for k, v in s.items() if k != "additionalProperties"}
    if isinstance(s, list):
        return [_gemini_schema(x) for x in s]
    return s


class GeminiExtractor:
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model=None, fallback_model="gemini-2.5-flash-lite"):
        import os
        self.key = os.environ.get("GEMINI_API_KEY") or (GEMINI_KEY_FILE.read_text().strip() if GEMINI_KEY_FILE.exists() else None)
        if not self.key:
            raise RuntimeError("no Gemini API key (GEMINI_API_KEY or ~/.config/hack_the_highway/gemini_api_key)")
        self.model = model or os.environ.get("HTW_GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.fallback_model = fallback_model
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0}
        self.min_interval = float(os.environ.get("HTW_LLM_MIN_INTERVAL", "4.5"))   # free tier ~15 requests/min
        self._last_call = 0.0

    def _post(self, model, body):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.URL.format(model=model), data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        import re as _re
        import time as _time
        last = None
        for attempt in range(6):                       # retry 429 / 5xx; other 4xx (config errors) surface at once
            wait = self.min_interval - (_time.time() - self._last_call)   # client-side pacing for free-tier RPM
            if wait > 0:
                _time.sleep(wait)
            self._last_call = _time.time()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                raw = e.read()
                last = f"HTTP {e.code}: {raw[:300]!r}"
                if e.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(last) from e
                m = _re.search(rb'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', raw)   # honour the server's requested delay
                delay = float(m.group(1)) + 1 if m else 2 ** (attempt + 1)
            except (urllib.error.URLError, TimeoutError) as e:
                last, delay = f"transport: {e}", 2 ** attempt
            self.usage["retries"] += 1
            _time.sleep(min(delay, 65))
        raise RuntimeError(f"Gemini unavailable after retries ({last})")

    def extract(self, text):
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": f"Message received {TODAY.isoformat()}:\n\n{text}"}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": _gemini_schema(SCHEMA)},
        }
        try:
            resp = self._post(self.model, body)
        except RuntimeError as e:
            if "HTTP 404" in str(e) or "HTTP 400" in str(e):   # model name / schema not accepted -> fallback model
                self.model = self.fallback_model
                resp = self._post(self.model, body)
            else:
                raise
        self.usage["calls"] += 1
        um = resp.get("usageMetadata", {})
        self.usage["input_tokens"] += um.get("promptTokenCount", 0)
        self.usage["output_tokens"] += um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)
        cand = (resp.get("candidates") or [{}])[0]
        if cand.get("finishReason") not in (None, "STOP"):
            raise RuntimeError(f"Gemini finishReason={cand.get('finishReason')}")
        txt = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        out = json.loads(txt)["incidents"]
        for inc in out:                                     # schema-check the model output before trusting it
            if inc.get("kind") not in KINDS:
                raise RuntimeError(f"invalid kind from LLM: {inc.get('kind')}")
            for f in FIELDS + ["summary", "reason"]:
                inc[f] = inc.get(f) or ""
            inc["confirmed_and_specific"] = bool(inc.get("confirmed_and_specific"))
        return out


# ====================================================================== rule-based extractor
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day,?\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b(?:next|last|this|coming)\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day\b"
    r"|\b(?:today|tomorrow|yesterday)\b", re.I)
RUMOR_RE = re.compile(r"\b(rumou?r|not confirmed|unconfirmed|nothing confirmed|no details|hearsay|someone (?:on the floor )?mentioned|might|may be|possibl[ey]|could be)\b", re.I)
QUALITY_RE = re.compile(r"\b(quality|defect\w*|faulty|contaminat\w*|suspect|recall|hold)\b", re.I)
OUTAGE_RE = re.compile(r"\b(outage|dark|power failure|lost production|production is stopped|shut ?down|cannot build|offline|(?:is|are|went|going) down|down (?:through|until|from|for))\b", re.I)
# operational vocabulary: a message that talks about operations but cannot be grounded is unclear (escalate), not noise
OPS_RE = re.compile(r"\b(plant|factory|line|production|shipment|supplier|parts?|lot|batter(?:y|ies)|harness|chip|tires?|seats?|recall|defect\w*|quality|delay\w*|late|outage|offline|down)\b", re.I)
DELAY_RE = re.compile(r"\b(delay\w*|late|can'?t make it|new eta|waiting on|will now arrive|carrier says)\b", re.I)


class RuleExtractor:
    name = "rules"

    def __init__(self, kg):
        self.kg = kg

    def split(self, text):
        parts = [p.strip() for p in re.split(r"\(\d+\)", text) if p.strip()]
        if len(parts) > 1 and len(parts[0].split()) <= 6:     # "Morning digest, two items." header
            parts = parts[1:]
        return parts or [text]

    def extract(self, text):
        out = []
        for chunk in self.split(text):
            mentions = self.kg.find_mentions(chunk)
            first = lambda t: next((s for typ, _, s in mentions if typ == t), "")
            dates = [m.group(0) for m in DATE_RE.finditer(chunk)]
            sh = re.search(r"\bSH-\d{4}\b", chunk)
            lot = re.search(r"\bLOT-\d{4}\b", chunk)
            if RUMOR_RE.search(chunk):
                kind = "unconfirmed"
            elif QUALITY_RE.search(chunk):
                kind = "quality_hold"
            elif OUTAGE_RE.search(chunk):
                kind = "plant_outage"
            elif DELAY_RE.search(chunk):
                kind = "supplier_delay"
            else:
                kind = "noise"
            if kind in ("supplier_delay", "plant_outage", "quality_hold") and not mentions and not (sh or lot):
                kind = "unconfirmed"          # operational but unspecific -> escalate (handbook rule 8)
            elif kind == "noise" and OPS_RE.search(chunk) and not re.search(r"\b(brochure|price ?list|weather|newsletter|marketing)\b", chunk, re.I):
                kind = "unconfirmed"
            out.append({
                "kind": kind, "summary": chunk[:160], "supplier": first("Supplier"), "part": first("Part"),
                "plant": first("Plant"), "shipment_id": sh.group(0) if sh else "", "lot_id": lot.group(0) if lot else "",
                "date_1": dates[0] if dates else "", "date_2": dates[1] if len(dates) > 1 else "",
                "confirmed_and_specific": kind not in ("unconfirmed", "noise"),
                "reason": f"keyword rules: {kind}",
            })
        return out


# ====================================================================== date grounding (code, not LLM)
def resolve_date(phrase, today=TODAY):
    """Turn a date phrase into a date. 'next <weekday>' = that weekday in next calendar week (Mon-Sun);
    'last <weekday>' = that weekday in the previous calendar week; 'this <weekday>' = this week."""
    if not phrase:
        return None, "no date given"
    p = phrase.strip().lower().rstrip(".,")
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", p)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3])), "ISO date"
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})", p)
    if m:
        d = date(today.year, MONTHS[m[1]], int(m[2]))
        wd = re.match(r"(mon|tues|wednes|thurs|fri|satur|sun)day", p)
        note = "month/day"
        if wd and WEEKDAYS[d.weekday()][:3] != wd[1][:3]:
            note += f" (WARNING: weekday '{wd[0]}' does not match {d.strftime('%A')})"
        return d, note
    week_start = today - timedelta(days=today.weekday())
    m = re.fullmatch(r"(next|last|this|coming)\s+(\w+day)", p)
    if m and m[2] in WEEKDAYS:
        wd = WEEKDAYS.index(m[2])
        off = {"next": 7, "last": -7, "this": 0, "coming": 0}[m[1]]
        d = week_start + timedelta(days=off + wd)
        if m[1] == "coming" and d <= today:
            d += timedelta(days=7)
        return d, f"relative: '{phrase}' from {today.strftime('%A %Y-%m-%d')}"
    if p == "today":
        return today, "relative"
    if p == "tomorrow":
        return today + timedelta(days=1), "relative"
    if p == "yesterday":
        return today - timedelta(days=3 if today.weekday() == 0 else 1), "relative (previous working day)"
    return None, f"could not parse date '{phrase}'"


def make_extractor(kg, backend=LLM_BACKEND):
    """auto: Gemini if a key is configured, else Claude if credentials exist, else rules."""
    if backend in ("auto", "gemini"):
        try:
            return GeminiExtractor(), None
        except Exception:
            if backend == "gemini":
                raise
    if backend in ("auto", "claude"):
        import os
        from pathlib import Path
        has_creds = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")) \
            or (Path.home() / ".config" / "anthropic").exists()
        if backend == "auto" and not has_creds:
            return RuleExtractor(kg), "no Anthropic credentials found (ANTHROPIC_API_KEY / ant auth login); using rule-based extractor"
        try:
            ex = ClaudeExtractor()
            return ex, None
        except Exception as e:  # missing SDK or credentials
            if backend == "claude":
                raise
            return RuleExtractor(kg), f"Claude unavailable ({type(e).__name__}: {e}); using rule-based extractor"
    return RuleExtractor(kg), "rule-based extractor selected"
