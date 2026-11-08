"""MCP-style tool client for the enterprise server (POST /act), with validation, audit trail and failure handling."""
import json
import time
import urllib.error
import urllib.request
from datetime import datetime

WRITE_TOOLS = {"mes.reschedule", "mes.transfer", "dealer.notify", "dealer.stop_sale", "crm.assign_loaner",
               "crm.notify_customer", "quality.hold_lot"}


class ToolClient:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        self.catalog = {t["tool"]: t for t in self._get("/tools")}

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return json.loads(r.read())

    def events(self):
        return self._get("/events")

    def session(self, event_id, trace):
        return ToolSession(self, event_id, trace)


class ToolSession:
    """All tool calls for one event. Every call (ok or failed) is appended to the event's audit trail."""

    def __init__(self, client, event_id, trace):
        self.c, self.event_id, self.trace = client, event_id, trace

    def _validate(self, tool, args):
        spec = self.c.catalog.get(tool)
        if spec is None:
            return f"unknown tool {tool}"
        missing = [a for a in spec["args"] if a not in args or args[a] in (None, "", [])]
        return f"missing args {missing}" if missing else None

    def call(self, agent, tool, args, why=""):
        entry = {"t": datetime.now().isoformat(timespec="milliseconds"), "agent": agent, "tool": tool,
                 "args": args, "why": why, "write": tool in WRITE_TOOLS}
        err = self._validate(tool, args)
        if err:                                   # never send a call we know is malformed
            entry.update(ok=False, sent=False, error=f"blocked before sending: {err}")
            self.trace.audit.append(entry)
            return False, {"error": err}
        body = json.dumps({"event_id": self.event_id, "tool": tool, "args": args}).encode()
        for attempt in range(3):                  # retry only transport failures, never a 4xx business error
            try:
                req = urllib.request.Request(self.c.base + "/act", data=body, method="POST",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    res, ok = json.loads(r.read()), True
                break
            except urllib.error.HTTPError as e:
                res, ok = json.loads(e.read() or b"{}"), False
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                res, ok = {"error": f"transport: {e}"}, False
                time.sleep(0.5 * (attempt + 1))
        entry.update(ok=ok, sent=True, result=res)
        self.trace.audit.append(entry)
        return ok, res
