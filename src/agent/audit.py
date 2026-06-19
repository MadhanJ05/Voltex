"""SQLite regulatory audit trail for every agent response."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class AuditLogger:
    def __init__(self, path: str | Path = "data/voltex_audit.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, date TEXT NOT NULL,
                signal_json TEXT NOT NULL, llm_used INTEGER NOT NULL, guardrail_json TEXT NOT NULL,
                latency_ms REAL NOT NULL, alert_json TEXT NOT NULL
            )""")

    def log(self, *, date: str, signal: dict, llm_used: bool, guardrails: dict, latency_ms: float, alert: dict) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO alerts (timestamp,date,signal_json,llm_used,guardrail_json,latency_ms,alert_json) VALUES (?,?,?,?,?,?,?)",
                (datetime.now(UTC).isoformat(), date, json.dumps(signal), int(llm_used), json.dumps(guardrails), latency_ms, json.dumps(alert)),
            )
