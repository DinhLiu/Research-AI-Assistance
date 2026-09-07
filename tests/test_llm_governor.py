import threading
from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate

import httpx
import pytest

from research_assistant.llm.governor import Governor, Quota, RunBudget, LlmBudgetExceeded, LlmDeferred, LlmError, RateLimitError


class Clock:
    def __init__(self):
        self.now = 1000.0
    def time(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def setup(tmp_path):
    clock = Clock()
    return clock, Governor(tmp_path, clock=clock.time, sleep=clock.sleep, jitter=lambda: 0)


def test_pacing_retry_and_restart(tmp_path):
    clock, gov = setup(tmp_path)
    starts = []
    def send(timeout):
        starts.append(clock.now)
        return httpx.Response(429, headers={"Retry-After": "20"}) if len(starts) == 1 else httpx.Response(200)
    budget = RunBudget("same")
    quota = Quota(60, 10000, 1000, safety=1)
    gov.request(send, group="g", quota=quota, budget=budget, tokens=100)
    restarted = Governor(tmp_path, clock=clock.time, sleep=clock.sleep)
    restarted.request(send, group="g", quota=quota, budget=budget, tokens=100)
    assert starts == [1000, 1020, 1035]
    assert restarted.stats("same") == {"http_attempts": 3, "reserved_tokens": 300}


def test_tpm_and_idle_do_not_accumulate_permits(tmp_path):
    clock, gov = setup(tmp_path)
    starts = []
    def send(timeout):
        starts.append(clock.now)
        return httpx.Response(200)
    kwargs = dict(group="g", quota=Quota(60, 150, 100, safety=1), budget=RunBudget("run"), tokens=100)
    gov.request(send, **kwargs)
    gov.request(send, **kwargs)
    clock.now += 300
    gov.request(send, **kwargs)
    gov.request(send, **kwargs)
    assert starts == [1000, 1060, 1360, 1420]


def test_zero_budget_and_oversized_request_do_not_send(tmp_path):
    _, gov = setup(tmp_path)
    def fail(_):
        pytest.fail("must not send")
    for budget, tokens in ((RunBudget("zero", 0), 100), (RunBudget("oversize"), 10001)):
        with pytest.raises(LlmBudgetExceeded):
            gov.request(fail, group="g", quota=Quota(10, 10000, 100), budget=budget, tokens=tokens)


def test_http_date_cooldown_deadline_and_no_sleep_after_final(tmp_path):
    clock, gov = setup(tmp_path)
    kwargs = dict(group="g", quota=Quota(60, 10000, 100), budget=RunBudget("run", deadline_s=20), tokens=100)
    with pytest.raises(RateLimitError):
        gov.request(lambda _: httpx.Response(429, headers={"Retry-After": formatdate(1050, usegmt=True)}), retries=0, **kwargs)
    assert clock.now == 1000
    with pytest.raises(LlmDeferred):
        gov.request(lambda _: pytest.fail("cooldown must block"), **kwargs)
    assert clock.now == 1000


def test_terminal_error_and_circuit(tmp_path):
    clock, gov = setup(tmp_path)
    kwargs = dict(group="g", quota=Quota(60, 10000, 100), budget=RunBudget("run"), tokens=100)
    with pytest.raises(LlmError, match="401"):
        gov.request(lambda _: httpx.Response(401), **kwargs)
    assert gov.stats("run")["http_attempts"] == 1
    with pytest.raises(RateLimitError, match="circuit_open"):
        gov.request(lambda _: httpx.Response(503), **kwargs)
    assert gov.stats("run")["http_attempts"] == 3
    assert clock.now == 1030


def test_parent_budget_and_resume_do_not_renew(tmp_path):
    _, gov = setup(tmp_path)
    quota = Quota(100, 10000, 100)
    parent = RunBudget("pipeline", max_attempts=1)
    gov.request(lambda _: httpx.Response(200), group="g", quota=quota, budget=RunBudget("stage3", parent=parent), tokens=100)
    with pytest.raises(LlmBudgetExceeded):
        gov.request(lambda _: pytest.fail("parent cap"), group="g", quota=quota,
                    budget=RunBudget("stage4", parent=RunBudget("pipeline", max_attempts=10)), tokens=100)


def test_parallel_governors_never_overlap_transport(tmp_path):
    # Real clock only for this contention test; transport synchronization does
    # not rely on sleep timing. Use a generous test quota with zero pacing floor.
    active = 0
    maximum = 0
    lock = threading.Lock()
    def send(_):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        with lock:
            active -= 1
        return httpx.Response(200)
    def work(i):
        gov = Governor(tmp_path)
        gov.request(send, group="g", quota=Quota(100000, 100000, 10000, safety=1, min_interval_s=0),
                    budget=RunBudget(str(i)), tokens=1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(8)))
    assert maximum == 1


def test_daily_exhaustion_is_not_retried(tmp_path):
    _, gov = setup(tmp_path)
    with pytest.raises(RateLimitError, match="quota_exhausted"):
        gov.request(lambda _: httpx.Response(429, text="GenerateRequestsPerDay"), group="g",
                    quota=Quota(60, 10000, 100), budget=RunBudget("daily"), tokens=100)
    assert gov.stats("daily")["http_attempts"] == 1


def test_gemini_retry_metadata(tmp_path):
    clock, gov = setup(tmp_path)
    starts = []
    def send(_):
        starts.append(clock.now)
        if len(starts) == 1:
            return httpx.Response(429, json={"error": {"details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "35s"}]}})
        return httpx.Response(200)
    gov.request(send, group="g", quota=Quota(60, 10000, 1000), budget=RunBudget("metadata"), tokens=100)
    assert starts == [1000, 1035]


def test_rpm_sliding_window_and_long_response(tmp_path):
    clock, gov = setup(tmp_path)
    starts = []
    def send(_):
        starts.append(clock.now)
        clock.sleep(40)
        return httpx.Response(200)
    for _ in range(3):
        gov.request(send, group="g", quota=Quota(2, 10000, 1000, safety=1), budget=RunBudget("slow"), tokens=100)
    assert starts == [1000, 1040, 1080]
    assert gov.telemetry("slow")["api_seconds"] == 120


def test_run_deadline_persists_after_restart(tmp_path):
    clock, gov = setup(tmp_path)
    gov.start_run(RunBudget("deadline", deadline_s=10))
    clock.sleep(11)
    with pytest.raises(LlmDeferred):
        gov.request(lambda _: pytest.fail("expired"), group="g", quota=Quota(60, 10000, 1000),
                    budget=RunBudget("deadline", deadline_s=600), tokens=100)


def _process_request(state_dir, barrier, output):
    import os
    import time
    from pathlib import Path
    gov = Governor(Path(state_dir))
    barrier.wait(timeout=10)
    def send(_):
        marker = Path(state_dir) / "in-flight-sentinel"
        # An overlap is an O_EXCL failure, propagated to the parent.
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            time.sleep(0.02)
            return httpx.Response(200)
        finally:
            os.close(fd)
            marker.unlink()
    try:
        gov.request(send, group="g", quota=Quota(100000, 100000, 10000, safety=1, min_interval_s=0),
                    budget=RunBudget(str(os.getpid())), tokens=1)
        output.put("ok")
    except Exception as exc:
        output.put(repr(exc))


def test_processes_share_inflight_lock(tmp_path):
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    barrier, output = ctx.Barrier(3), ctx.Queue()
    processes = [ctx.Process(target=_process_request, args=(str(tmp_path), barrier, output)) for _ in range(3)]
    for proc in processes:
        proc.start()
    try:
        results = [output.get(timeout=15) for _ in processes]
        assert results == ["ok"] * 3
    finally:
        for proc in processes:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        output.close()
