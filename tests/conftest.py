"""Shared fixtures: an in-memory fake of the MiniMax V2 video API."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from minimax_batch.api import MiniMaxClient
from minimax_batch.batch import RawBatch, resolve_jobs

API = "https://api.test"
CDN = "cdn.test"
TEST_KEY = "sk-test-SECRET-key-123"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"x" * 2048


async def nosleep(_: float) -> None:
    await asyncio.sleep(0)


def error_body(code: int, etype: str, message: str) -> dict:
    return {"type": "error", "error": {"type": etype, "message": message, "http_code": code}, "request_id": "req-1"}


@dataclass
class FakeTask:
    statuses: list[str]
    idx: int = 0
    url_version: int = 0
    error: dict | None = None

    def current(self) -> str:
        return self.statuses[min(self.idx, len(self.statuses) - 1)]


@dataclass
class FakeServer:
    """Scriptable fake. Queue responses with ``create_script`` / ``query_script``.

    Script entries are either an int HTTP status (with an error body), an
    Exception instance to raise, or a callable(request) -> httpx.Response.
    """

    default_statuses: list[str] = field(default_factory=lambda: ["queued", "running", "succeeded"])
    tasks: dict[str, FakeTask] = field(default_factory=dict)
    create_script: list[Any] = field(default_factory=list)
    query_script: list[Any] = field(default_factory=list)
    download_script: list[Any] = field(default_factory=list)
    expired_urls: set[str] = field(default_factory=set)
    requests: list[httpx.Request] = field(default_factory=list)
    create_bodies: list[dict] = field(default_factory=list)
    next_statuses: list[list[str]] = field(default_factory=list)
    create_delay: float = 0.0
    # concurrency tracking: a "job" is active from create until its video is downloaded
    active_tasks: set[str] = field(default_factory=set)
    max_active_tasks: int = 0
    inflight_requests: int = 0
    max_inflight_requests: int = 0

    # ---- counters
    def count(self, method: str, path_prefix: str, host: str | None = None) -> int:
        return sum(
            1
            for r in self.requests
            if r.method == method and r.url.path.startswith(path_prefix) and (host is None or r.url.host == host)
        )

    @property
    def creates(self) -> int:
        return self.count("POST", "/v2/video_generation")

    @property
    def queries(self) -> int:
        return self.count("GET", "/v2/query/video_generation/")

    @property
    def downloads(self) -> int:
        return self.count("GET", "/", host=CDN)

    def add_task(self, task_id: str, statuses: list[str], error: dict | None = None) -> None:
        self.tasks[task_id] = FakeTask(list(statuses), error=error)

    # ---- transport
    def _scripted(self, script: list[Any], request: httpx.Request) -> httpx.Response | None:
        if not script:
            return None
        item = script.pop(0)
        if item is None:
            return None
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(request)
        codes = {400: "bad_request_error", 401: "authorized_error", 402: "insufficient_balance_error",
                 422: "unprocessable_entity_error", 429: "rate_limit_error"}
        return httpx.Response(item, json=error_body(item, codes.get(item, "server_error"), f"scripted {item}"))

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.inflight_requests += 1
        self.max_inflight_requests = max(self.max_inflight_requests, self.inflight_requests)
        try:
            await asyncio.sleep(0)  # yield so concurrent jobs genuinely interleave
            if request.method == "POST" and self.create_delay:
                await asyncio.sleep(self.create_delay)
            return self._handle(request)
        finally:
            self.inflight_requests -= 1

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == CDN:
            r = self._scripted(self.download_script, request)
            if r is not None:
                return r
            if str(request.url) in self.expired_urls:
                return httpx.Response(403, text="AccessDenied: Request has expired")
            self.active_tasks.discard(request.url.path.split("/")[1])
            return httpx.Response(200, content=VIDEO_BYTES, headers={"content-length": str(len(VIDEO_BYTES))})

        assert request.headers.get("authorization") == f"Bearer {TEST_KEY}"
        path = request.url.path
        if request.method == "POST" and path == "/v2/video_generation":
            r = self._scripted(self.create_script, request)
            if r is not None:
                return r
            body = json.loads(request.content)
            self.create_bodies.append(body)
            task_id = f"task-{len(self.tasks) + 1}"
            statuses = self.next_statuses.pop(0) if self.next_statuses else self.default_statuses
            self.add_task(task_id, statuses)
            self.active_tasks.add(task_id)
            self.max_active_tasks = max(self.max_active_tasks, len(self.active_tasks))
            return httpx.Response(200, json={"task_id": task_id})

        m = re.fullmatch(r"/v2/query/video_generation/(.+)", path)
        if request.method == "GET" and m:
            r = self._scripted(self.query_script, request)
            if r is not None:
                return r
            task_id = m.group(1)
            task = self.tasks.get(task_id)
            if task is None:
                return httpx.Response(400, json=error_body(400, "bad_request_error", "task not found"))
            status = task.current()
            task.idx += 1
            payload: dict[str, Any] = {"id": task_id, "status": status, "resolution": "768P", "duration": 6,
                                       "ratio": "16:9"}
            if status == "succeeded":
                task.url_version += 1
                payload["content"] = {"url": f"https://{CDN}/{task_id}/v{task.url_version}.mp4", "prompt": "p"}
                payload["usage"] = {"duration": 6}
            if status == "failed":
                payload["error"] = task.error or {"code": "generation_failed", "message": "model error"}
            return httpx.Response(200, json={"task": payload})

        return httpx.Response(404, json=error_body(404, "not_found", "no route"))


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def make_client(server: FakeServer) -> Callable[..., MiniMaxClient]:
    def factory(**kw: Any) -> MiniMaxClient:
        kw.setdefault("sleep", nosleep)
        return MiniMaxClient(TEST_KEY, API, transport=httpx.MockTransport(server.handler), **kw)

    return factory


def make_jobs(jobs: list[dict], base_dir: Path, defaults: dict | None = None, cli: dict | None = None):
    return resolve_jobs(RawBatch(defaults=defaults or {}, jobs=jobs, base_dir=base_dir), cli)


@pytest.fixture
def png_file(tmp_path: Path) -> Path:
    p = tmp_path / "frame.png"
    p.write_bytes(PNG_BYTES)
    return p
