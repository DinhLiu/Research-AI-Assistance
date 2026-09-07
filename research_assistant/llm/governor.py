"""Persistent, single-flight admission for every HTTP attempt (local POSIX hosts).

flock serializes admission AND transport across processes; the kernel releases it
on crash. SQLite keeps pacing, cooldown and run reservations across restarts.
Reservations are deliberately never refunded: ambiguous failures can cost quota.
"""
from __future__ import annotations

import fcntl
import math
import os
import random
import sqlite3
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

import httpx


class LlmError(RuntimeError):
    """Terminal for the caller; never turn this into a content repair."""


class RateLimitError(LlmError):
    pass


class LlmBudgetExceeded(LlmError):
    pass


class LlmDeferred(LlmError):
    pass


@dataclass(frozen=True)
class Quota:
    rpm: int
    tpm: int
    rpd: int
    safety: float = 0.8
    min_interval_s: float = 15.0

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (self.rpm, self.tpm, self.rpd)):
            raise ValueError("Quota RPM/TPM/RPD must be positive")
        if not 0 < self.safety <= 1 or not math.isfinite(self.min_interval_s) or self.min_interval_s < 0:
            raise ValueError("Invalid quota safety/interval")

    @classmethod
    def from_env(cls):
        values = [os.environ.get(f"LLM_{name}") for name in ("RPM", "TPM", "RPD")]
        if not all(values):
            raise LlmError("quota_configuration_required: set LLM_RPM, LLM_TPM, LLM_RPD")
        return cls(*(int(value) for value in values))


@dataclass(frozen=True)
class RunBudget:
    run_id: str
    max_attempts: int = 5
    max_tokens: int = 100000
    deadline_s: float = 600
    parent: RunBudget | None = None

    def __post_init__(self):
        if not self.run_id or any(type(n) is not int or n < 0 for n in (self.max_attempts, self.max_tokens)):
            raise ValueError("Invalid run budget")
        if not math.isfinite(self.deadline_s) or self.deadline_s <= 0:
            raise ValueError("Invalid run deadline")


_ACTIVE_BUDGET = ContextVar("llm_run_budget", default=None)


def current_budget():
    return _ACTIVE_BUDGET.get()


@contextmanager
def llm_run(budget: RunBudget):
    """Optional pipeline-wide budget shared by all complete() call sites."""
    token = _ACTIVE_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _ACTIVE_BUDGET.reset(token)


def retry_after(value: str | None, now: float) -> float:
    if not value:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - now
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


def provider_retry_delay(response, now):
    if response is None:
        return 0.0
    wait = retry_after(response.headers.get("Retry-After"), now)
    try:
        details = response.json().get("error", {}).get("details", [])
        for detail in details:
            if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("google.rpc.RetryInfo"):
                value = detail.get("retryDelay", "")
                if isinstance(value, str) and value.endswith("s"):
                    wait = max(wait, retry_after(value[:-1], now))
    except (ValueError, TypeError, AttributeError):
        pass
    return wait


@contextmanager
def file_lock(path: Path, *, clock=time.time, sleep=time.sleep, deadline=float("inf")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        while True:
            if clock() >= deadline:
                raise LlmDeferred("deadline_waiting_for_lock")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                sleep(min(0.1, max(0, deadline - clock())))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Governor:
    def __init__(self, state_dir: Path | None = None, *, clock=time.time, sleep=time.sleep, jitter=random.random):
        self.state_dir = Path(state_dir or os.environ.get("LLM_STATE_DIR", Path(__file__).resolve().parents[2] / "data/llm_state"))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.clock, self.sleep, self.jitter = clock, sleep, jitter
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, attempts INTEGER, tokens INTEGER,
                    max_attempts INTEGER, max_tokens INTEGER, deadline REAL);
                CREATE TABLE IF NOT EXISTS events (group_id TEXT, started REAL, tokens INTEGER);
                CREATE INDEX IF NOT EXISTS event_window ON events(group_id, started);
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, run_id TEXT, is_retry INTEGER,
                    api_seconds REAL, wait_seconds REAL);
                CREATE TABLE IF NOT EXISTS groups (
                    id TEXT PRIMARY KEY, next_start REAL DEFAULT 0,
                    cooldown REAL DEFAULT 0, failures INTEGER DEFAULT 0);
            """)

    @contextmanager
    def _db(self):
        connection = sqlite3.connect(self.state_dir / "quota.sqlite3", timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _budgets(self, budget):
        seen = set()
        while budget:
            if budget.run_id in seen:
                raise ValueError("Cyclic/repeated run budget")
            seen.add(budget.run_id)
            yield budget
            budget = budget.parent

    def start_run(self, budget: RunBudget) -> float:
        with self._db() as db:
            for item in self._budgets(budget):
                db.execute("INSERT OR IGNORE INTO runs VALUES (?,0,0,?,?,?)",
                           (item.run_id, item.max_attempts, item.max_tokens, self.clock() + item.deadline_s))
                # A resume may tighten caps; it cannot renew a deadline or budget.
                db.execute("UPDATE runs SET max_attempts=min(max_attempts,?), max_tokens=min(max_tokens,?) WHERE id=?",
                           (item.max_attempts, item.max_tokens, item.run_id))
            return min(db.execute("SELECT deadline FROM runs WHERE id=?", (item.run_id,)).fetchone()[0]
                       for item in self._budgets(budget))

    def stats(self, run_id: str) -> dict:
        with self._db() as db:
            row = db.execute("SELECT attempts,tokens FROM runs WHERE id=?", (run_id,)).fetchone()
        return {"http_attempts": row[0] if row else 0, "reserved_tokens": row[1] if row else 0}

    def telemetry(self, run_id):
        with self._db() as db:
            row = db.execute("SELECT coalesce(sum(is_retry),0), coalesce(sum(api_seconds),0), coalesce(sum(wait_seconds),0) FROM attempts WHERE run_id=?", (run_id,)).fetchone()
        return {"transport_retries": row[0], "api_seconds": row[1], "queue_wait_seconds": row[2]}

    def request(self, send_once: Callable[[float], httpx.Response], *, group: str, quota: Quota,
                budget: RunBudget, tokens: int, timeout_s: float = 90, retries: int = 1) -> httpx.Response:
        if tokens <= 0 or timeout_s <= 0 or retries < 0:
            raise ValueError("Invalid request limits")
        rpm = max(1, int(quota.rpm * quota.safety))
        tpm = max(1, int(quota.tpm * quota.safety))
        rpd = max(1, int(quota.rpd * quota.safety))
        if tokens > tpm:
            raise LlmBudgetExceeded("request_exceeds_token_window")
        deadline = self.start_run(budget)
        interval = max(quota.min_interval_s, 60 / (quota.rpm * quota.safety))
        for attempt in range(retries + 1):
            waiting_since = self.clock()
            with file_lock(self.state_dir / "transport.lock", clock=self.clock, sleep=self.sleep, deadline=deadline):
                while True:
                    now = self.clock()
                    if now >= deadline:
                        raise LlmDeferred("run_deadline")
                    with self._db() as db:
                        db.execute("BEGIN IMMEDIATE")
                        for item in self._budgets(budget):
                            used, reserved, max_calls, max_tokens = db.execute(
                                "SELECT attempts,tokens,max_attempts,max_tokens FROM runs WHERE id=?", (item.run_id,)).fetchone()
                            if used >= max_calls or reserved + tokens > max_tokens:
                                raise LlmBudgetExceeded("run_budget_exhausted")
                        db.execute("INSERT OR IGNORE INTO groups(id) VALUES (?)", (group,))
                        next_start, cooldown, failures = db.execute(
                            "SELECT next_start,cooldown,failures FROM groups WHERE id=?", (group,)).fetchone()
                        events = db.execute("SELECT started,tokens FROM events WHERE group_id=? AND started>? ORDER BY started",
                                            (group, now - 86400)).fetchall()
                        wait_until = max(next_start, cooldown)
                        minute = [(ts, count) for ts, count in events if ts > now - 60]
                        if len(events) >= rpd:
                            # Rolling 24h is conservative across provider reset timezones.
                            raise LlmDeferred("daily_request_budget")
                        if len(minute) >= rpm:
                            wait_until = max(wait_until, minute[-rpm][0] + 60)
                        total = sum(count for _, count in minute) + tokens
                        for ts, count in minute:
                            if total <= tpm:
                                break
                            wait_until = max(wait_until, ts + 60)
                            total -= count
                        if wait_until <= now:
                            for item in self._budgets(budget):
                                db.execute("UPDATE runs SET attempts=attempts+1,tokens=tokens+? WHERE id=?", (tokens, item.run_id))
                            db.execute("INSERT INTO events VALUES (?,?,?)", (group, now, tokens))
                            audit_id = db.execute("INSERT INTO attempts(run_id,is_retry,wait_seconds) VALUES (?,?,?)",
                                                  (budget.run_id, int(attempt > 0), max(0, now - waiting_since))).lastrowid
                            db.execute("UPDATE groups SET next_start=? WHERE id=?", (now + interval, group))
                            db.execute("DELETE FROM events WHERE started<=?", (now - 86400,))
                            break
                    if wait_until >= deadline:
                        raise LlmDeferred("cooldown_exceeds_deadline")
                    self.sleep(min(30, max(0.001, wait_until - self.clock())))
                # Keep the process lock through transport; no lease expiry can cause overlap.
                error = None
                api_started = self.clock()
                try:
                    response = send_once(min(timeout_s, deadline - self.clock()))
                except (httpx.HTTPError, TimeoutError) as exc:
                    response = None
                    error = type(exc).__name__
                finally:
                    with self._db() as db:
                        db.execute("UPDATE attempts SET api_seconds=? WHERE id=?", (max(0, self.clock() - api_started), audit_id))
                transient = response is None or response.status_code in {408, 429} or response.status_code >= 500
                if response is not None and response.status_code < 400:
                    with self._db() as db:
                        db.execute("UPDATE groups SET failures=0 WHERE id=?", (group,))
                    return response
                if not transient:
                    raise LlmError(f"llm_http_{response.status_code}")
                # Daily/billing quota indicators are terminal, not rapid retries.
                body = response.text.lower() if response is not None else ""
                permanent = any(marker in body for marker in ("insufficient_quota", "billing", "perday", "per_day", "daily quota"))
                now = self.clock()
                delay = max(provider_retry_delay(response, now),
                            min(60, 2 ** (attempt + 1)) + self.jitter())
                failures += 1
                if failures >= 2:
                    delay = max(delay, 60)
                if permanent:
                    delay = max(delay, 86400)
                with self._db() as db:
                    db.execute("UPDATE groups SET failures=?,cooldown=max(cooldown,?) WHERE id=?",
                               (failures, now + delay, group))
                if permanent or failures >= 2 or attempt == retries:
                    label = "quota_exhausted" if permanent else ("circuit_open" if failures >= 2 else "transport_exhausted")
                    raise RateLimitError(f"{label}:{response.status_code if response is not None else error}")
        raise AssertionError("unreachable")
