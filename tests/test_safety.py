"""Money-safety, security and robustness regression tests (Phase 2 review items)."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from minimax_batch import api, cli, spec
from minimax_batch import manifest as mf
from minimax_batch.api import MiniMaxClient, UnsafeDownloadError, check_download_url, sanitize
from minimax_batch.batch import BatchFileError, ValidationError, load_batch_file, redact_payload, resolve_jobs
from minimax_batch.runner import RunOptions, Runner

from .conftest import API, CDN, PNG_BYTES, TEST_KEY, VIDEO_BYTES, make_jobs, nosleep
from .test_runner import manifest_jobs, run, text_jobs

ROOT = Path(__file__).resolve().parents[1]


def statuses(out: Path) -> dict[str, str]:
    return {n: e["status"] for n, e in manifest_jobs(out).items()}


# ============================================================ A. double billing


def test_submitting_entry_is_blocked_until_retry_unknown(tmp_path, server, make_client):
    """Simulates a SIGKILL mid-submit: the entry was left as 'submitting'."""
    out = tmp_path / "out"
    [job] = text_jobs(tmp_path)
    mf.Manifest.load(out).update("clip-1", status="submitting", task_id=None, params_hash=job.params_hash)
    results, _ = run(make_client, [job], out)
    assert results["clip-1"].outcome == "blocked" and "interrupted while submitting" in results["clip-1"].detail
    assert server.creates == 0
    results, _ = run(make_client, [job], out, retry_unknown=True)
    assert results["clip-1"].outcome == "downloaded" and server.creates == 1


def test_manifest_says_submitting_while_create_is_in_flight_and_cancel_leaves_submit_unknown(
    tmp_path, server, make_client
):
    server.create_delay = 30.0
    out = tmp_path / "out"
    jobs = text_jobs(tmp_path)
    seen_during_flight = {}

    async def go():
        async with make_client() as client:
            runner = Runner(client, mf.Manifest.load(out), out, RunOptions(poll_initial=0.01),
                            Console(record=True), sleep=nosleep)
            task = asyncio.create_task(runner.run(jobs))
            while server.inflight_requests == 0:
                await asyncio.sleep(0.005)
            seen_during_flight.update(statuses(out))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(go())
    assert seen_during_flight == {"clip-1": "submitting"}
    entry = manifest_jobs(out)["clip-1"]
    assert entry["status"] == "submit_unknown" and "CancelledError" in entry["error"]

    server.create_delay = 0.0
    results, _ = run(make_client, jobs, out)
    assert results["clip-1"].outcome == "blocked"
    assert server.creates == 0  # the cancelled POST never reached _handle; no second POST either


def test_task_id_printed_before_manifest_write(tmp_path, server, make_client, monkeypatch):
    out = tmp_path / "out"
    jobs = text_jobs(tmp_path)
    real_update = mf.Manifest.update

    def flaky(self, name, **fields):
        if fields.get("status") == mf.SUBMITTED:
            raise OSError("disk full")
        return real_update(self, name, **fields)

    monkeypatch.setattr(mf.Manifest, "update", flaky)
    console = Console(record=True, width=300)

    async def go():
        async with make_client() as client:
            return await Runner(client, mf.Manifest.load(out), out, RunOptions(), console, sleep=nosleep).run(jobs)

    [result] = asyncio.run(go())
    assert result.outcome == "error"
    assert "task=task-1" in console.export_text()
    monkeypatch.undo()
    assert statuses(out) == {"clip-1": "submitting"}  # blocks resubmission on the next run
    results, _ = run(make_client, jobs, out)
    assert results["clip-1"].outcome == "blocked" and server.creates == 1


@pytest.mark.parametrize(
    "response",
    [
        lambda req: httpx.Response(200, text="<html>gateway</html>"),
        lambda req: httpx.Response(200, json={"status": "ok"}),
        lambda req: httpx.Response(200, json={"task_id": ""}),
        lambda req: httpx.Response(200, json=[1, 2]),
    ],
    ids=["non-json", "no-task-id", "empty-task-id", "json-list"],
)
def test_2xx_create_without_task_id_is_submit_unknown(tmp_path, server, make_client, response):
    server.create_script = [response]
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out)
    assert results["clip-1"].outcome == "submit_unknown"
    assert server.creates == 1
    assert statuses(out) == {"clip-1": "submit_unknown"}


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_5xx_on_create_is_not_retried(tmp_path, server, make_client, code):
    server.create_script = [code]
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out)
    assert results["clip-1"].outcome == "submit_unknown"
    assert server.creates == 1
    assert statuses(out) == {"clip-1": "submit_unknown"}


def test_2xx_error_envelope_with_429_is_retried(tmp_path, server, make_client):
    env = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down", "http_code": 429}}
    server.create_script = [lambda req: httpx.Response(200, json=env)]
    server.query_script = [lambda req: httpx.Response(200, json=env)]
    results, _ = run(make_client, text_jobs(tmp_path), tmp_path / "out")
    assert results["clip-1"].outcome == "downloaded"
    assert server.creates == 2 and len(server.tasks) == 1


def test_2xx_error_envelope_with_400_is_definitive(tmp_path, server, make_client):
    env = {"type": "error", "error": {"type": "bad_request_error", "message": "bad", "http_code": "400"}}
    server.create_script = [lambda req: httpx.Response(200, json=env)]
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out)
    assert results["clip-1"].outcome == "failed" and server.creates == 1
    assert statuses(out) == {"clip-1": "failed"}


def test_rerun_changed_keeps_previous_task_ids(tmp_path, server, make_client):
    out = tmp_path / "out"
    run(make_client, text_jobs(tmp_path), out)
    changed = text_jobs(tmp_path, duration=10)
    run(make_client, changed, out, rerun_changed=True)
    entry = manifest_jobs(out)["clip-1"]
    assert entry["task_id"] == "task-2" and entry["previous_task_ids"] == ["task-1"]


@pytest.mark.parametrize("status", ["running", "queued", "submitted", "timeout", "submit_unknown", "submitting"])
def test_rerun_changed_refuses_live_tasks_without_retry_unknown(tmp_path, server, make_client, status):
    out = tmp_path / "out"
    mf.Manifest.load(out).update("clip-1", status=status, task_id="task-old", params_hash="old-hash")
    changed = text_jobs(tmp_path, duration=10)
    results, _ = run(make_client, changed, out, rerun_changed=True)
    assert results["clip-1"].outcome == "blocked" and server.creates == 0
    results, _ = run(make_client, changed, out, rerun_changed=True, retry_unknown=True)
    assert results["clip-1"].outcome == "downloaded" and server.creates == 1
    assert manifest_jobs(out)["clip-1"]["previous_task_ids"] == ["task-old"]


def test_changed_params_on_failed_job_just_resubmits(tmp_path, server, make_client):
    out = tmp_path / "out"
    mf.Manifest.load(out).update("clip-1", status="failed", task_id=None, params_hash="old-hash")
    results, _ = run(make_client, text_jobs(tmp_path), out)
    assert results["clip-1"].outcome == "downloaded"


# ------------------------------------------------------------ real process, real signals


class _SlowApi(BaseHTTPRequestHandler):
    posts = 0
    create_delay = 3.0

    def log_message(self, *a):  # silence
        pass

    def _json(self, code, body):
        data = json.dumps(body).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        type(self).posts += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(type(self).create_delay)
        self._json(200, {"task_id": f"slow-{type(self).posts}"})

    def do_GET(self):
        port = self.server.server_address[1]
        if self.path.startswith("/v2/query/"):
            self._json(200, {"task": {"id": "x", "status": "succeeded",
                                      "content": {"url": f"http://127.0.0.1:{port}/video.mp4"}}})
        else:
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(VIDEO_BYTES)))
                self.end_headers()
                self.wfile.write(VIDEO_BYTES)
            except (BrokenPipeError, ConnectionResetError):
                pass


@pytest.fixture
def slow_server():
    handler = type("H", (_SlowApi,), {"posts": 0})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, handler
    srv.shutdown()
    srv.server_close()


def _cli(tmp_path, port, *extra):
    env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
    env["MINIMAX_API_KEY"] = TEST_KEY
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [sys.executable, "-m", "minimax_batch.cli", "run", "b.yaml", "--out", "o",
           "--base-url", f"http://127.0.0.1:{port}", "--allow-custom-host", "--allow-insecure-downloads", *extra]
    return subprocess.Popen(cmd, cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGKILL], ids=["SIGINT", "SIGKILL"])
def test_signal_mid_submit_never_double_posts(tmp_path, slow_server, sig):
    srv, handler = slow_server
    port = srv.server_address[1]
    (tmp_path / "b.yaml").write_text("jobs:\n  - name: slow\n    prompt: hello\n")

    proc = _cli(tmp_path, port)
    deadline = time.time() + 15
    while handler.posts == 0 and time.time() < deadline:
        time.sleep(0.02)
    assert handler.posts == 1
    time.sleep(0.3)
    proc.send_signal(sig)
    output, _ = proc.communicate(timeout=15)

    entry = json.loads((tmp_path / "o" / "manifest.json").read_text())["jobs"]["slow"]
    if sig == signal.SIGINT:
        assert proc.returncode == 130, output
        assert "slow" in output and "may have been charged" in output
        assert entry["status"] == "submit_unknown"
    else:
        assert entry["status"] == "submitting"

    rerun = _cli(tmp_path, port)
    output2, _ = rerun.communicate(timeout=30)
    assert rerun.returncode == 1, output2
    assert "blocked" in output2
    assert handler.posts == 1  # exactly one POST across both runs
    assert TEST_KEY not in output + output2


# ============================================================ B. real API quirk


NOT_FOUND_BODY = {"type": "error", "error": {"type": "server_error", "message": "record not found (1000)",
                                            "http_code": "500"}}


def not_found(req):
    return httpx.Response(500, json=NOT_FOUND_BODY)


def test_query_record_not_found_marks_task_lost_not_resubmittable(tmp_path, server, make_client):
    out = tmp_path / "out"
    [job] = text_jobs(tmp_path)
    mf.Manifest.load(out).update("clip-1", status="running", task_id="ghost", params_hash=job.params_hash)
    server.query_script = [not_found] * 10
    results, _ = run(make_client, [job], out)
    assert results["clip-1"].outcome == "task_lost"
    assert "NOT resubmitted" in results["clip-1"].detail
    assert server.queries == 1  # not retried
    assert statuses(out) == {"clip-1": "task_lost"}

    server.query_script = []
    results, _ = run(make_client, [job], out)  # plain re-run must not pay again
    assert results["clip-1"].outcome == "blocked" and server.creates == 0
    results, _ = run(make_client, [job], out, retry_unknown=True)
    assert results["clip-1"].outcome == "downloaded" and server.creates == 1
    assert manifest_jobs(out)["clip-1"]["previous_task_ids"] == ["ghost"]


def test_deleted_mp4_with_purged_task_is_not_rebilled(tmp_path, server, make_client):
    out = tmp_path / "out"
    jobs = text_jobs(tmp_path)
    run(make_client, jobs, out)
    (out / "clip-1.mp4").unlink()
    server.query_script = [not_found] * 10
    results, _ = run(make_client, jobs, out)
    assert results["clip-1"].outcome == "task_lost"
    server.query_script = []
    results, _ = run(make_client, jobs, out)
    assert results["clip-1"].outcome == "blocked"
    assert server.creates == 1


def test_fresh_task_not_found_gets_grace_period(tmp_path, server, make_client):
    server.query_script = [not_found, not_found]
    results, _ = run(make_client, text_jobs(tmp_path), tmp_path / "out")
    assert results["clip-1"].outcome == "downloaded"
    assert server.creates == 1


def test_fresh_task_not_found_beyond_grace_is_task_lost(tmp_path, server, make_client):
    server.query_script = [not_found] * 50
    t = [0.0]

    def clock():
        t[0] += 10.0
        return t[0]

    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out, clock=clock, not_found_grace=60.0)
    assert results["clip-1"].outcome == "task_lost"
    assert 2 <= server.queries <= 8  # retried during the grace period, then gave up
    assert statuses(out) == {"clip-1": "task_lost"} and server.creates == 1


def test_not_found_requires_5xx_and_exact_message(tmp_path, server, make_client):
    out = tmp_path / "out"
    [job] = text_jobs(tmp_path)
    mf.Manifest.load(out).update("clip-1", status="running", task_id="t1", params_hash=job.params_hash)
    server.query_script = [lambda r: httpx.Response(400, json={"type": "error", "error": {
        "type": "bad_request_error", "message": "record not found (1000)", "http_code": 400}})]
    results, _ = run(make_client, [job], out)
    assert results["clip-1"].outcome == "error" and statuses(out) == {"clip-1": "running"}


def test_401_with_1000_code_is_still_auth_error(tmp_path, server, make_client):
    out = tmp_path / "out"
    [old, new] = make_jobs([{"name": "old", "prompt": "a"}, {"name": "new", "prompt": "b"}], tmp_path)
    mf.Manifest.load(out).update("old", status="running", task_id="t1", params_hash=old.params_hash)
    server.query_script = [lambda r: httpx.Response(401, json={"type": "error", "error": {
        "type": "authorized_error", "message": "invalid api key (1000)", "http_code": 401}})]
    results, runner = run(make_client, [old, new], out, concurrency=1)
    assert results["old"].outcome == "error" and "authorized_error" in runner.abort_reason
    assert results["new"].outcome == "aborted" and server.creates == 0
    assert statuses(out) == {"old": "running", "new": "not_submitted"}


def test_2xx_error_envelope_without_http_code_is_submit_unknown(tmp_path, server, make_client):
    server.create_script = [lambda r: httpx.Response(200, json={"type": "error", "error": {"message": "hm"}})]
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out)
    assert results["clip-1"].outcome == "submit_unknown" and server.creates == 1


def test_create_500_message_is_not_duplicated(tmp_path, server, make_client):
    server.create_script = [500]
    results, _ = run(make_client, text_jobs(tmp_path), tmp_path / "out")
    detail = results["clip-1"].detail
    assert detail.count("HTTP 500") == 1 and detail.count("request_id") == 1, detail


def test_abort_keeps_possibly_billed_states(tmp_path, server, make_client):
    out = tmp_path / "out"
    a, b, c = make_jobs([{"name": n, "prompt": n} for n in ("a", "b", "c")], tmp_path)
    m = mf.Manifest.load(out)
    m.update("a", status="submit_unknown", task_id=None, params_hash=a.params_hash)
    m.update("b", status="task_lost", task_id="lost-1", params_hash=b.params_hash)
    server.create_script = [401]
    results, _ = run(make_client, [a, b, c], out, concurrency=1, retry_unknown=True)
    entries = manifest_jobs(out)
    assert entries["a"]["status"] == "submit_unknown"
    assert entries["b"]["status"] == "task_lost" and entries["b"]["task_id"] == "lost-1"
    assert entries["c"]["status"] == "not_submitted"
    assert server.creates == 1


def test_base_url_recorded_and_mismatch_blocks_only_resume(tmp_path, server, make_client):
    out = tmp_path / "out"
    [job] = text_jobs(tmp_path)
    run(make_client, [job], out, base_url="https://api.test")
    assert manifest_jobs(out)["clip-1"]["base_url"] == "https://api.test"

    m = mf.Manifest.load(out)
    m.update("clip-1", status="running")
    results, _ = run(make_client, [job], out, base_url="https://api.minimax.io")
    assert results["clip-1"].outcome == "blocked" and "same --base-url" in results["clip-1"].detail
    results, _ = run(make_client, [job], out, base_url="https://API.test/")  # same host: resumes
    assert results["clip-1"].outcome == "downloaded" and server.creates == 1

    # uncertain entry: plain run blocked; with --retry-unknown a new submit may go to another host
    m.update("clip-1", status="submit_unknown", task_id=None)
    results, _ = run(make_client, [job], out, base_url="https://api.minimax.io")
    assert results["clip-1"].outcome == "blocked" and server.creates == 1
    results, _ = run(make_client, [job], out, base_url="https://api.minimax.io", retry_unknown=True)
    assert results["clip-1"].outcome == "downloaded" and server.creates == 2
    entry = manifest_jobs(out)["clip-1"]
    assert entry["base_url"] == "https://api.minimax.io"
    assert entry["previous_tasks"][-1] == {"task_id": None, "base_url": "https://api.test",
                                           "status": "submit_unknown"}


def test_rerun_changed_live_task_on_other_host_records_old_host(tmp_path, server, make_client):
    out = tmp_path / "out"
    # legacy manifest shape: plain previous_task_ids, no previous_tasks
    mf.Manifest.load(out).update("clip-1", status="running", task_id="old-2", params_hash="old-hash",
                                 base_url="https://api.test", previous_task_ids=["old-1"])
    changed = text_jobs(tmp_path, duration=10)
    results, _ = run(make_client, changed, out, base_url="https://api.minimax.io", rerun_changed=True)
    assert results["clip-1"].outcome == "blocked" and server.creates == 0  # still needs --retry-unknown
    results, _ = run(make_client, changed, out, base_url="https://api.minimax.io", rerun_changed=True,
                     retry_unknown=True)
    assert results["clip-1"].outcome == "downloaded" and server.creates == 1
    entry = manifest_jobs(out)["clip-1"]
    assert entry["previous_task_ids"] == ["old-1", "old-2"]
    assert entry["previous_tasks"] == [{"task_id": "old-2", "base_url": "https://api.test", "status": "running"}]


def test_retry_of_uncertain_job_hitting_429s_stays_uncertain(tmp_path, server, make_client):
    out = tmp_path / "out"
    [job] = text_jobs(tmp_path)
    mf.Manifest.load(out).update("clip-1", status="submit_unknown", task_id=None, params_hash=job.params_hash)
    server.create_script = [429] * 20

    def client(**kw):
        return make_client(max_retries=2, **kw)

    results, _ = run(client, [job], out, retry_unknown=True)
    assert results["clip-1"].outcome == "failed" and "429" in results["clip-1"].detail
    assert server.creates == 3
    assert statuses(out) == {"clip-1": "submit_unknown"}
    results, _ = run(client, [job], out)  # next plain run is still blocked
    assert results["clip-1"].outcome == "blocked" and server.creates == 3


# ============================================================ C. security: downloads


def dl_client(handler, **kw):
    kw.setdefault("sleep", nosleep)
    kw.setdefault("max_retries", 1)
    return MiniMaxClient(TEST_KEY, API, transport=httpx.MockTransport(handler), **kw)


def do_download(handler, dest, url=f"https://{CDN}/v.mp4", **kw):
    async def go():
        async with dl_client(handler, **kw) as c:
            return await c.download(url, dest)

    return asyncio.run(go())


def test_download_rejects_oversized_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "MAX_VIDEO_BYTES", 100)
    with pytest.raises(UnsafeDownloadError, match="limit"):
        do_download(lambda r: httpx.Response(200, content=VIDEO_BYTES), tmp_path / "v.mp4")
    assert list(tmp_path.iterdir()) == []


def test_download_aborts_stream_over_limit_without_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "MAX_VIDEO_BYTES", 100)

    async def body():
        for _ in range(10):
            yield VIDEO_BYTES[:50]

    with pytest.raises(UnsafeDownloadError, match="exceeded"):
        do_download(lambda r: httpx.Response(200, content=body()), tmp_path / "v.mp4")
    assert list(tmp_path.iterdir()) == []


def test_download_rejects_non_mp4(tmp_path):
    with pytest.raises(UnsafeDownloadError, match="not an MP4"):
        do_download(lambda r: httpx.Response(200, content=b"<html>login page</html>"), tmp_path / "v.mp4")
    assert list(tmp_path.iterdir()) == []


def test_download_attempt_timeout(tmp_path):
    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, content=VIDEO_BYTES)

    t0 = time.monotonic()
    with pytest.raises(api.ApiError, match="timed out"):
        do_download(slow, tmp_path / "v.mp4", download_attempt_timeout=0.05)
    assert time.monotonic() - t0 < 2


@pytest.mark.parametrize(
    "url",
    ["http://cdn.test/v.mp4", "https://10.0.0.5/v.mp4", "https://169.254.169.254/latest", "https://[::1]/v.mp4",
     "https://localhost/v.mp4", "https://foo.localhost/v.mp4", "https://intranet/v.mp4", "file:///etc/passwd"],
)
def test_unsafe_download_urls_rejected(url):
    with pytest.raises(UnsafeDownloadError):
        check_download_url(url)


@pytest.mark.parametrize(
    "host", ["127.1", "0x7f.1", "10.1", "017700000001", "2130706433", "cdn.0x10", "a.b.123", "0x7f000001"]
)
def test_legacy_ip_forms_rejected(host):
    with pytest.raises(UnsafeDownloadError):
        check_download_url(f"https://{host}/v.mp4")


def test_safe_download_url_and_test_escape():
    check_download_url("https://video-cdn.minimax.io/a/b.mp4?sig=1")
    check_download_url("http://127.0.0.1:8080/v.mp4", allow_insecure=True)
    with pytest.raises(UnsafeDownloadError):
        check_download_url("file:///x", allow_insecure=True)


@pytest.mark.parametrize("target", ["http://cdn.test/v.mp4", "https://169.254.169.254/x", "https://127.0.0.1/x"])
def test_redirect_hops_are_checked(tmp_path, target):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.host == CDN:
            return httpx.Response(302, headers={"location": target})
        return httpx.Response(200, content=VIDEO_BYTES)

    with pytest.raises(UnsafeDownloadError):
        do_download(handler, tmp_path / "v.mp4")
    assert seen == [f"https://{CDN}/v.mp4"]  # the unsafe hop was never requested


def test_redirect_loop_capped(tmp_path):
    with pytest.raises(UnsafeDownloadError, match="redirected too many"):
        do_download(lambda r: httpx.Response(302, headers={"location": f"https://{CDN}/again"}), tmp_path / "v")


def test_api_client_does_not_follow_redirects(tmp_path, server, make_client):
    out = tmp_path / "out"
    [job] = text_jobs(tmp_path)
    mf.Manifest.load(out).update("clip-1", status="running", task_id="t1", params_hash=job.params_hash)
    server.query_script = [lambda req: httpx.Response(302, headers={"location": "https://evil.test/steal"})]
    results, _ = run(make_client, [job], out)
    assert results["clip-1"].outcome == "error" and "redirect" in results["clip-1"].detail
    assert all(r.url.host != "evil.test" for r in server.requests)


# ============================================================ C. security: base url, secrets, .env


@pytest.mark.parametrize(
    "url, allow, ok",
    [
        ("https://api.minimax.io", False, True),
        ("https://api-uw.minimax.io", False, True),
        ("https://api.minimaxi.com", False, True),
        ("https://evil.test", False, False),
        ("https://evil.test", True, True),
        ("https://api.minimax.io.evil.test", False, False),
        ("http://api.minimax.io", False, False),
        ("http://example.com", True, False),
        ("http://127.0.0.1:9000", True, True),
        ("http://localhost:9000", False, False),
        ("https://user:pw@api.minimax.io", False, False),
        ("ftp://api.minimax.io", True, False),
        ("http://localhost:8000", True, True),
        ("http://127.0.0.2:8000", True, True),
        ("http://[::1]:8000", True, True),
        ("http://127.evil.test", True, False),
        ("http://127.0.0.1.nip.io", True, False),
        ("https://evil.test\\@api.minimax.io", False, False),
        ("https://evil.test\\@api.minimax.io", True, False),
        ("https://api.minimax.io%2f.evil.test/", False, False),
        ("https://api.minimax.io%2f.evil.test/", True, False),
        ("https://api.minimax.io /x", False, False),
        ("https://API.MiniMax.io:443/", False, True),
    ],
)
def test_check_base_url(url, allow, ok):
    assert (cli.check_base_url(url, allow) is None) is ok


def test_api_error_messages_are_scrubbed(tmp_path, server, make_client):
    nasty = f"bad key {TEST_KEY} header Bearer abc.def.ghi \x1b[31mred\x07 \x9b"
    server.create_script = [lambda req: httpx.Response(400, json={"type": "error", "error": {
        "type": "bad_request_error", "message": nasty, "http_code": 400}})]
    out = tmp_path / "out"
    results, _ = run(make_client, text_jobs(tmp_path), out)
    err = manifest_jobs(out)["clip-1"]["error"]
    for text in (err, results["clip-1"].detail):
        assert TEST_KEY not in text and "abc.def.ghi" not in text
        assert "\x1b" not in text and "\x07" not in text and "\x9b" not in text
        assert "Bearer ***" in text


def test_sanitize_unit():
    api.register_secret("sk-unit-secret")
    assert sanitize("x sk-unit-secret y\r\n\x00z") == "x *** y z"


def test_dotenv_permission_warning(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("MINIMAX_API_KEY=abc\n")
    warnings: list[str] = []
    env.chmod(0o644)
    assert cli.load_api_key(tmp_path, warn=warnings.append) == "abc"
    assert warnings and "chmod 600" in warnings[0]
    warnings.clear()
    env.chmod(0o600)
    assert cli.load_api_key(tmp_path, warn=warnings.append) == "abc"
    assert warnings == []


# ============================================================ C. security: images & YAML


def test_image_outside_batch_dir_rejected(tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(PNG_BYTES)
    (batch_dir / "link.png").symlink_to(outside)
    for ref in ("../secret.png", str(outside), "link.png"):
        with pytest.raises(ValidationError) as ei:
            make_jobs([{"name": "a", "prompt": "p", "first_frame": ref}], batch_dir)
        assert "outside the batch file's directory" in ei.value.errors[0]
    from minimax_batch.batch import RawBatch

    [job] = resolve_jobs(RawBatch({}, [{"name": "a", "prompt": "p", "first_frame": "link.png"}], batch_dir),
                         allow_external_images=True)
    assert job.payload["ratio"] == "adaptive"


def test_image_magic_bytes_checked(tmp_path):
    (tmp_path / "fake.png").write_bytes(b"#!/bin/sh\necho not an image\n")
    with pytest.raises(ValidationError) as ei:
        make_jobs([{"name": "a", "prompt": "p", "first_frame": "fake.png"}], tmp_path)
    assert "magic bytes" in ei.value.errors[0]
    jpeg = tmp_path / "actually-jpeg.png"
    jpeg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
    [job] = make_jobs([{"name": "a", "prompt": "p", "first_frame": "actually-jpeg.png"}], tmp_path)
    assert job.payload["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_data_uri_magic_checked(tmp_path):
    import base64

    bad = "data:image/png;base64," + base64.b64encode(b"hello world, not an image").decode()
    with pytest.raises(ValidationError):
        make_jobs([{"name": "a", "prompt": "p", "first_frame": bad}], tmp_path)


def test_redaction_reveals_no_image_bytes(tmp_path, png_file):
    [job] = make_jobs([{"name": "img", "prompt": "p", "first_frame": "frame.png"}], tmp_path)
    full = job.payload["content"][1]["image_url"]["url"].split(",", 1)[1]
    red = redact_payload(job.payload)["content"][1]["image_url"]["url"]
    assert full[:8] not in red
    assert "sha256:" in red and f"<{len(full)} base64 chars" in red


def test_yaml_alias_rejected(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text("defaults: &d\n  duration: 5\njobs:\n  - name: a\n    prompt: *d\n")
    with pytest.raises(BatchFileError, match="aliases"):
        load_batch_file(p)


def test_yaml_non_string_prompt_rejected(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text("jobs:\n  - name: a\n    prompt: 12345\n")
    with pytest.raises(ValidationError) as ei:
        resolve_jobs(load_batch_file(p))
    assert any("prompt must be text" in e for e in ei.value.errors)


def test_yaml_unquoted_ratio_hint(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text("jobs:\n  - name: a\n    prompt: hi\n    ratio: 16:9\n")
    with pytest.raises(ValidationError) as ei:
        resolve_jobs(load_batch_file(p))
    assert any("quote ratios in YAML" in e for e in ei.value.errors)


def test_error_values_truncated(tmp_path):
    with pytest.raises(ValidationError) as ei:
        make_jobs([{"name": "a", "prompt": "p", "model": "M" * 5000}], tmp_path)
    assert len(ei.value.errors[0]) < 250


# ============================================================ D. robustness


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0", "abc"])
def test_timeout_must_be_finite_positive(value, capsys):
    with pytest.raises(SystemExit) as ei:
        cli.build_parser().parse_args(["run", "b.yaml", "--timeout", value])
    assert ei.value.code == 2


def test_short_timeout_message_in_seconds(tmp_path, server, make_client):
    server.next_statuses = [["queued"] * 10]
    t = [0.0]

    def clock():
        t[0] += 10.0
        return t[0]

    results, _ = run(make_client, text_jobs(tmp_path), tmp_path / "out", clock=clock, job_timeout=30.0)
    assert results["clip-1"].outcome == "timeout" and "after 30s" in results["clip-1"].detail


def test_lock_reports_non_contention_errors_honestly(tmp_path, monkeypatch):
    import fcntl

    def boom(fd, op):
        if op & fcntl.LOCK_EX:
            raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", boom)
    with pytest.raises(mf.ManifestError, match="could not lock"):
        with mf.run_lock(tmp_path / "o"):
            pass


def test_manifest_write_fsyncs_parent_dir(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mf, "_fsync_dir", lambda d: calls.append(d))
    m = mf.Manifest.load(tmp_path)
    m.update("a", status="x")
    assert calls == [tmp_path]


def test_resumed_job_error_reports_manifest_task_id(tmp_path, server, make_client):
    out = tmp_path / "out"
    [job] = text_jobs(tmp_path)
    mf.Manifest.load(out).update("clip-1", status="running", task_id="task-keep", params_hash=job.params_hash)
    server.add_task("task-keep", ["running"])
    server.query_script = [401]
    results, _ = run(make_client, [job], out)
    assert results["clip-1"].outcome == "error" and results["clip-1"].task_id == "task-keep"
    assert statuses(out) == {"clip-1": "running"}  # task untouched, resumable


def test_connection_refused_is_logged_as_connection_error(tmp_path, server, make_client):
    server.create_script = [httpx.ConnectError("refused")] * 3
    def client(**kw):
        return make_client(max_retries=2, **kw)

    results, runner = run(client, text_jobs(tmp_path), tmp_path / "out")
    r = results["clip-1"]
    assert r.outcome == "failed" and "connection error" in r.detail
    text = runner.console.export_text()
    assert "connection error (not submitted)" in text and "rejected" not in text


def test_dry_run_blocked_exits_1_and_corrupt_manifest_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    b = tmp_path / "b.csv"
    b.write_text("name,prompt\none,hello\n")
    out = tmp_path / "o"
    out.mkdir()
    (out / "one.mp4").write_bytes(b"unmanaged")
    assert cli.main(["run", str(b), "--out", str(out), "--dry-run"]) == 1
    assert "blocked" in capsys.readouterr().out
    (out / "manifest.json").write_text("{broken")
    assert cli.main(["run", str(b), "--out", str(out), "--dry-run"]) == 2


def test_ctrl_c_message_names_uncertain_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINIMAX_API_KEY", TEST_KEY)
    b = tmp_path / "b.csv"
    b.write_text("name,prompt\nalpha,hello\nbeta,world\n")

    class InterruptingRunner:
        def __init__(self, client, manifest, out_dir, opts, console):
            self.manifest = manifest
            self.abort_reason = ""

        async def run(self, jobs):
            self.manifest.update("alpha", status="submitting")
            self.manifest.update("beta", status="running", task_id="t-b")
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "Runner", InterruptingRunner)
    code = cli.main(["run", str(b), "--out", str(tmp_path / "o"), "--base-url", "https://api.test",
                     "--allow-custom-host"])
    out = capsys.readouterr().out
    assert code == 130
    assert "1 job(s) were interrupted mid-submit" in out and "alpha" in out
    assert "--retry-unknown" in out


def test_banner_counts_only_new_submissions(tmp_path, server, monkeypatch, capsys):
    import functools

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINIMAX_API_KEY", TEST_KEY)
    monkeypatch.setattr(cli, "MiniMaxClient",
                        functools.partial(MiniMaxClient, transport=httpx.MockTransport(server.handler), sleep=nosleep))
    monkeypatch.setattr(cli, "Runner", functools.partial(Runner, sleep=nosleep))
    b = tmp_path / "b.csv"
    b.write_text("name,prompt,duration\none,hello,7\ntwo,world,9\n")
    args = ["run", str(b), "--out", str(tmp_path / "o"), "--base-url", "https://api.test", "--allow-custom-host"]
    assert cli.main(args) == 0
    assert "2 to submit (16s of video billable)" in capsys.readouterr().out
    assert cli.main(args) == 0
    assert "0 to submit (0s of video billable)" in capsys.readouterr().out


def test_status_output_is_sanitized(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"])
    out = tmp_path / "o"
    mf.Manifest.load(out).update(
        "job[red]", status="failed\x1b[2J", task_id="[bold]tid\x07", model="\x9bm",
        error=f"boom {TEST_KEY} Bearer abc.def \x1b]0;title\x07",
    )
    api.register_secret(TEST_KEY)
    assert cli.main(["status", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "\x1b" not in text and "\x07" not in text and "\x9b" not in text
    assert TEST_KEY not in text and "abc.def" not in text
    assert "[bold]tid" in text and "job[red]" in text  # markup shown literally, not interpreted
