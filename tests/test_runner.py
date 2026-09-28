from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from minimax_batch import manifest as mf
from minimax_batch.runner import RunOptions, Runner

from .conftest import CDN, TEST_KEY, VIDEO_BYTES, make_jobs, nosleep

from rich.console import Console


def run(make_client, jobs, out: Path, clock=None, **opts):
    opts.setdefault("poll_initial", 0.01)
    opts.setdefault("poll_max", 0.01)

    async def go():
        async with make_client() as client:
            manifest = mf.Manifest.load(out)
            kw = {"sleep": nosleep}
            if clock is not None:
                kw["clock"] = clock
            runner = Runner(client, manifest, out, RunOptions(**opts), Console(record=True, width=200), **kw)
            results = await runner.run(jobs)
            return results, runner

    results, runner = asyncio.run(go())
    return {r.name: r for r in results}, runner


def manifest_jobs(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text())["jobs"]


def text_jobs(tmp_path, n=1, **extra):
    return make_jobs([{"name": f"clip-{i}", "prompt": f"prompt {i}", **extra} for i in range(1, n + 1)], tmp_path)


# ---------------------------------------------------------------- happy path


def test_poll_to_success_and_download(tmp_path, server, make_client):
    out = tmp_path / "out"
    jobs = text_jobs(tmp_path)
    results, _ = run(make_client, jobs, out)

    assert results["clip-1"].outcome == "downloaded"
    assert (out / "clip-1.mp4").read_bytes() == VIDEO_BYTES
    assert server.creates == 1 and server.queries == 3 and server.downloads == 1
    assert server.create_bodies[0] == jobs[0].payload

    entry = manifest_jobs(out)["clip-1"]
    assert entry["status"] == "downloaded"
    assert entry["task_id"] == "task-1"
    assert entry["file"] == "clip-1.mp4"
    assert entry["params_hash"] == jobs[0].params_hash
    assert entry["error"] is None
    # no leftover temp files
    assert sorted(p.name for p in out.iterdir() if not p.name.startswith(".minimax")) == ["clip-1.mp4", "manifest.json"]


def test_download_request_does_not_carry_api_key(tmp_path, server, make_client):
    run(make_client, text_jobs(tmp_path), tmp_path / "out")
    dl = [r for r in server.requests if r.url.host == CDN]
    assert dl and all("authorization" not in r.headers for r in dl)
    assert all(TEST_KEY not in str(r.url) for r in server.requests)


def test_concurrent_jobs_all_recorded(tmp_path, server, make_client):
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path, n=6), out, concurrency=3)
    assert all(r.outcome == "downloaded" for r in results.values())
    assert server.max_active_tasks == 3  # reached, never exceeded
    assert server.max_inflight_requests <= 3
    entries = manifest_jobs(out)
    assert len(entries) == 6 and {e["task_id"] for e in entries.values()} == {f"task-{i}" for i in range(1, 7)}


# ---------------------------------------------------------------- task failures


def test_failed_task(tmp_path, server, make_client):
    server.next_statuses = [["queued", "failed"]]
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out)
    r = results["clip-1"]
    assert r.outcome == "failed" and "model error" in r.detail
    assert manifest_jobs(out)["clip-1"]["status"] == "failed"
    assert not (out / "clip-1.mp4").exists()


def test_cancelled_task(tmp_path, server, make_client):
    server.next_statuses = [["running", "cancelled"]]
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out)
    assert results["clip-1"].outcome == "cancelled"
    assert manifest_jobs(out)["clip-1"]["status"] == "cancelled"


def test_failed_job_is_resubmitted_on_rerun(tmp_path, server, make_client):
    server.next_statuses = [["failed"]]
    out = tmp_path / "out"
    jobs = text_jobs(tmp_path)
    run(make_client, jobs, out)
    results, _ = run(make_client, jobs, out)
    assert results["clip-1"].outcome == "downloaded"
    assert server.creates == 2


def test_422_rejected_not_retried(tmp_path, server, make_client):
    server.create_script = [422]
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out)
    assert results["clip-1"].outcome == "failed"
    assert "unprocessable_entity_error" in results["clip-1"].detail
    assert server.creates == 1
    assert manifest_jobs(out)["clip-1"]["status"] == "failed"


# ---------------------------------------------------------------- retries


def test_429_on_create_is_retried(tmp_path, server, make_client):
    server.create_script = [429, 429]
    results, _ = run(make_client, text_jobs(tmp_path), tmp_path / "out")
    assert results["clip-1"].outcome == "downloaded"
    assert server.creates == 3
    assert len(server.tasks) == 1  # only one task actually created


def test_429_retry_exhaustion(tmp_path, server, make_client):
    server.create_script = [429] * 10

    def client(**kw):
        return make_client(max_retries=2, **kw)

    results, _ = run(client, text_jobs(tmp_path), tmp_path / "out")
    assert results["clip-1"].outcome == "failed"
    assert server.creates == 3


def test_5xx_and_network_errors_on_query_retried(tmp_path, server, make_client):
    server.query_script = [500, httpx.ConnectError("boom"), httpx.ReadTimeout("slow"), 503]
    results, _ = run(make_client, text_jobs(tmp_path), tmp_path / "out")
    assert results["clip-1"].outcome == "downloaded"
    assert server.queries == 3 + 4


def test_retry_uses_backoff_with_retry_after(tmp_path, server, make_client):
    delays: list[float] = []

    async def rec(d):
        delays.append(d)

    server.create_script = [
        lambda req: httpx.Response(429, headers={"retry-after": "7"}, json={"type": "error", "error": {}}),
        429,
    ]

    async def go():
        async with make_client(sleep=rec, rng=lambda: 1.0, backoff_base=2.0) as c:
            return await c.create_task({"x": 1})

    assert asyncio.run(go()) == "task-1"
    assert delays == [7.0, 4.0]  # Retry-After honoured, then base*2**1 with jitter=1.0


def test_ambiguous_network_error_on_create_is_not_retried(tmp_path, server, make_client):
    server.create_script = [httpx.ReadTimeout("read timed out")]
    out = tmp_path / "out"
    jobs = text_jobs(tmp_path)
    results, _ = run(make_client, jobs, out)
    assert results["clip-1"].outcome == "submit_unknown"
    assert server.creates == 1
    assert manifest_jobs(out)["clip-1"]["status"] == "submit_unknown"

    # re-run refuses to resubmit without --retry-unknown
    results, _ = run(make_client, jobs, out)
    assert results["clip-1"].outcome == "blocked" and server.creates == 1
    results, _ = run(make_client, jobs, out, retry_unknown=True)
    assert results["clip-1"].outcome == "downloaded" and server.creates == 2


def test_connect_error_on_create_is_retried(tmp_path, server, make_client):
    server.create_script = [httpx.ConnectError("refused")]
    results, _ = run(make_client, text_jobs(tmp_path), tmp_path / "out")
    assert results["clip-1"].outcome == "downloaded"
    assert server.creates == 2


# ---------------------------------------------------------------- fatal errors


def test_401_aborts_remaining_jobs(tmp_path, server, make_client):
    server.create_script = [401]
    out = tmp_path / "out"
    results, runner = run(make_client, text_jobs(tmp_path, n=4), out, concurrency=1)
    outcomes = [results[f"clip-{i}"].outcome for i in range(1, 5)]
    assert outcomes == ["error", "aborted", "aborted", "aborted"]
    assert server.creates == 1  # never retried, nothing else submitted
    assert "authorized_error" in runner.abort_reason
    entries = manifest_jobs(out)
    assert {n: e["status"] for n, e in entries.items()} == {f"clip-{i}": "not_submitted" for i in range(1, 5)}
    assert all(e["task_id"] is None for e in entries.values())
    assert "batch aborted" in entries["clip-2"]["error"]

    # not_submitted jobs are resubmitted on the next run
    results, _ = run(make_client, text_jobs(tmp_path, n=4), out, concurrency=1)
    assert all(r.outcome == "downloaded" for r in results.values())


def test_402_stops_submissions_but_resumes_existing_tasks(tmp_path, server, make_client):
    out = tmp_path / "out"
    old, new = make_jobs([{"name": "old", "prompt": "a"}, {"name": "new", "prompt": "b"}], tmp_path)
    server.add_task("task-old", ["running", "succeeded"])
    m = mf.Manifest.load(out)
    m.update("old", task_id="task-old", status="running", params_hash=old.params_hash)
    server.create_script = [402]

    results, runner = run(make_client, [new, old], out, concurrency=1)
    assert results["new"].outcome == "error"
    assert "insufficient_balance_error" in runner.abort_reason
    assert results["old"].outcome == "downloaded"  # free: already paid for
    assert server.creates == 1


# ---------------------------------------------------------------- resume


def test_resume_polls_existing_task_without_resubmitting(tmp_path, server, make_client):
    out = tmp_path / "out"
    [job] = text_jobs(tmp_path)
    server.add_task("task-abc", ["running", "succeeded"])
    mf.Manifest.load(out).update("clip-1", task_id="task-abc", status="running", params_hash=job.params_hash)

    results, _ = run(make_client, [job], out)
    assert results["clip-1"].outcome == "downloaded"
    assert server.creates == 0
    assert manifest_jobs(out)["clip-1"]["task_id"] == "task-abc"


def test_rerun_skips_downloaded_jobs_with_no_network(tmp_path, server, make_client):
    out = tmp_path / "out"
    jobs = text_jobs(tmp_path, n=2)
    run(make_client, jobs, out)
    before = len(server.requests)
    results, _ = run(make_client, jobs, out)
    assert all(r.outcome == "skipped" for r in results.values())
    assert len(server.requests) == before


def test_missing_file_refetched_from_existing_task(tmp_path, server, make_client):
    out = tmp_path / "out"
    jobs = text_jobs(tmp_path)
    run(make_client, jobs, out)
    (out / "clip-1.mp4").unlink()
    server.tasks["task-1"].statuses = ["succeeded"]
    results, _ = run(make_client, jobs, out)
    assert results["clip-1"].outcome == "downloaded"
    assert server.creates == 1


def test_changed_params_blocked_unless_rerun_changed(tmp_path, server, make_client):
    out = tmp_path / "out"
    run(make_client, text_jobs(tmp_path), out)
    changed = text_jobs(tmp_path, duration=10)
    results, _ = run(make_client, changed, out)
    assert results["clip-1"].outcome == "blocked" and server.creates == 1
    results, _ = run(make_client, changed, out, rerun_changed=True)
    assert results["clip-1"].outcome == "downloaded" and server.creates == 2
    assert manifest_jobs(out)["clip-1"]["params_hash"] == changed[0].params_hash


def test_existing_unmanaged_file_is_not_overwritten(tmp_path, server, make_client):
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip-1.mp4").write_bytes(b"mine")
    results, _ = run(make_client, text_jobs(tmp_path), out)
    assert results["clip-1"].outcome == "blocked"
    assert server.creates == 0
    assert (out / "clip-1.mp4").read_bytes() == b"mine"


def test_timeout_then_resume_same_task(tmp_path, server, make_client):
    out = tmp_path / "out"
    jobs = text_jobs(tmp_path)
    server.next_statuses = [["queued"] * 5 + ["running", "succeeded"]]
    t = [0.0]

    def clock():
        t[0] += 100.0
        return t[0]

    results, _ = run(make_client, jobs, out, clock=clock, job_timeout=250.0)
    assert results["clip-1"].outcome == "timeout"
    assert manifest_jobs(out)["clip-1"]["status"] == "timeout"
    assert server.creates == 1

    results, _ = run(make_client, jobs, out, job_timeout=10_000.0)
    assert results["clip-1"].outcome == "downloaded"
    assert server.creates == 1  # never paid twice


def test_poll_backoff_grows_and_caps(tmp_path, server, make_client):
    server.next_statuses = [["queued"] * 8 + ["succeeded"]]
    delays: list[float] = []

    async def rec(d):
        delays.append(d)

    async def go():
        async with make_client() as client:
            runner = Runner(client, mf.Manifest.load(tmp_path / "o"), tmp_path / "o",
                            RunOptions(poll_initial=5, poll_max=30, poll_factor=1.5), Console(record=True),
                            sleep=rec)
            return await runner.run(text_jobs(tmp_path))

    asyncio.run(go())
    assert delays[0] == 5
    assert delays == sorted(delays) and max(delays) == 30


# ---------------------------------------------------------------- downloads


def test_expired_download_url_is_refreshed_once(tmp_path, server, make_client):
    server.expired_urls = {f"https://{CDN}/task-1/v1.mp4"}
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out)
    assert results["clip-1"].outcome == "downloaded"
    assert server.downloads == 2
    assert server.queries == 4  # 3 polls + 1 refresh
    assert (out / "clip-1.mp4").read_bytes() == VIDEO_BYTES


def test_download_expired_twice_leaves_job_resumable(tmp_path, server, make_client):
    server.expired_urls = {f"https://{CDN}/task-1/v1.mp4", f"https://{CDN}/task-1/v2.mp4"}
    out = tmp_path / "out"
    jobs = text_jobs(tmp_path)
    results, _ = run(make_client, jobs, out)
    assert results["clip-1"].outcome == "error"
    entry = manifest_jobs(out)["clip-1"]
    assert entry["status"] == "succeeded" and "download failed" in entry["error"]
    assert not (out / "clip-1.mp4").exists()

    server.expired_urls = set()
    results, _ = run(make_client, jobs, out)
    assert results["clip-1"].outcome == "downloaded" and server.creates == 1


def test_truncated_download_retried_and_no_partial_file(tmp_path, server, make_client):
    server.download_script = [
        lambda req: httpx.Response(200, content=b"short", headers={"content-length": "9999"}),
    ]
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out)
    assert results["clip-1"].outcome == "downloaded"
    assert server.downloads == 2
    assert not list(out.glob("*.part")) and not list(out.glob(".*.part"))


# ---------------------------------------------------------------- manifest


def test_corrupt_manifest_is_not_overwritten(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.json").write_text("{not json")
    with pytest.raises(mf.ManifestError):
        mf.Manifest.load(out)
    assert (out / "manifest.json").read_text() == "{not json"


def test_run_lock_is_exclusive(tmp_path):
    out = tmp_path / "out"
    with mf.run_lock(out):
        with pytest.raises(mf.ManifestError):
            with mf.run_lock(out):
                pass
    with mf.run_lock(out):  # released
        pass
