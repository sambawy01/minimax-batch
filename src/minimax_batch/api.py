"""Thin async client for the MiniMax V2 video generation API.

Retry policy
------------
* Idempotent GETs (task queries, downloads): 429, 5xx and network errors are
  retried with exponential backoff + jitter (``Retry-After`` honoured).
* ``POST /v2/video_generation`` costs money, so it is only retried when the
  server *definitely* did not create a task: HTTP 429 (also when delivered as a
  2xx error envelope) and connection-establishment failures (the request never
  left the machine). Anything else that leaves the outcome uncertain - 5xx,
  read timeouts, dropped connections, a 2xx without a usable ``task_id`` -
  raises :class:`AmbiguousSubmitError` so the caller never pays twice.
* Other 4xx (400/401/402/422...) are never retried.

Security
--------
* The API key is only placed in the ``Authorization`` header of requests to the
  API base URL; the API client never follows redirects.
* Every error message is scrubbed of the key / bearer tokens and of control
  characters (see :func:`sanitize`).
* Video downloads use a separate client without the auth header. Download URLs
  must be https on every redirect hop, must not point at IP literals /
  localhost, are size-capped and must look like an MP4 before being kept.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import random
import re
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote

import httpx

from . import spec

SleepFn = Callable[[float], Awaitable[None]]

_CONNECT_PHASE_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_TASK_NOT_FOUND_RE = re.compile(r"record not found", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_SECRETS: set[str] = set()


def register_secret(secret: str) -> None:
    """Remember a secret so :func:`sanitize` removes it from any string."""
    if secret and len(secret) >= 4:
        _SECRETS.add(secret)


def sanitize(text: Any, limit: int = 1000) -> str:
    """Scrub secrets and control characters from API/network-sourced text."""
    s = str(text)
    for secret in _SECRETS:
        s = s.replace(secret, "***")
    s = _BEARER_RE.sub("Bearer ***", s)
    s = _CONTROL_RE.sub("", s.replace("\n", " ").replace("\t", " "))
    return s if len(s) <= limit else s[:limit] + "..."


class ApiError(Exception):
    def __init__(
        self,
        message: str,
        *,
        http_code: int | None = None,
        error_type: str | None = None,
        request_id: str | None = None,
        retryable: bool = False,
    ):
        self.message = sanitize(message)
        self.http_code = http_code
        self.error_type = sanitize(error_type, 100) if error_type else None
        self.request_id = sanitize(request_id, 100) if request_id else None
        self.retryable = retryable
        super().__init__(str(self))

    def __str__(self) -> str:
        parts = []
        if self.http_code:
            parts.append(f"HTTP {self.http_code}")
        if self.error_type:
            parts.append(self.error_type)
        head = " ".join(parts)
        s = f"{head}: {self.message}" if head else self.message
        if self.request_id:
            s += f" (request_id={self.request_id})"
        return s


class AuthError(ApiError):
    """401 - bad / missing API key. Fatal for the whole batch."""


class InsufficientBalanceError(ApiError):
    """402 - account out of credit. Fatal for further submissions."""


class AmbiguousSubmitError(ApiError):
    """A create request may or may not have produced a (billed) task."""


class ConnectionFailedError(ApiError):
    """Could not connect at all (retries exhausted); the request was never sent."""


class TaskNotFoundError(ApiError):
    """The query endpoint does not know this task_id (MiniMax: 500 'record not found (1000)')."""


class UnsafeDownloadError(ApiError):
    """A download URL (or redirect hop) failed the safety checks, or the body is not acceptable."""


class DownloadExpiredError(Exception):
    """The time-limited download URL is no longer valid (401/403/404/410)."""

    def __init__(self, status: int):
        self.status = status
        super().__init__(f"download URL rejected with HTTP {status} (likely expired)")


FatalApiErrors = (AuthError, InsufficientBalanceError)


def _error_class(http_code: int | None) -> type[ApiError]:
    if http_code == 401:
        return AuthError
    if http_code == 402:
        return InsufficientBalanceError
    return ApiError


def _is_retryable_status(code: int | None) -> bool:
    return code is not None and (code == 429 or code >= 500)


_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain")


def check_download_url(url: httpx.URL | str, allow_insecure: bool = False) -> None:
    """Raise UnsafeDownloadError unless ``url`` is an https URL on a public hostname."""
    u = httpx.URL(str(url))
    if allow_insecure:
        if u.scheme not in ("http", "https"):
            raise UnsafeDownloadError(f"download URL scheme {u.scheme!r} not allowed")
        return
    if u.scheme != "https":
        raise UnsafeDownloadError(f"download URL must be https (got {u.scheme or 'no scheme'})")
    host = (u.host or "").strip("[]").lower().rstrip(".")
    if not host:
        raise UnsafeDownloadError("download URL has no host")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise UnsafeDownloadError(f"download URL host is an IP literal ({sanitize(host, 64)}); refusing")
    if _is_legacy_ip_form(host):
        raise UnsafeDownloadError(f"download URL host looks like an IP address ({sanitize(host, 64)}); refusing")
    if host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES) or "." not in host:
        raise UnsafeDownloadError(f"download URL host {sanitize(host, 64)!r} is not a public hostname; refusing")


def _is_legacy_ip_form(host: str) -> bool:
    """Shorthand / hex / octal IPv4 forms (127.1, 0x7f.1, 10.1, 017700000001) that resolvers accept."""
    try:
        socket.inet_aton(host)
        return True
    except (OSError, ValueError):
        pass
    last = host.rsplit(".", 1)[-1]
    return last.isdigit() or last.startswith("0x")


def _looks_like_mp4(path: Path) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(12)
    return len(head) >= 8 and head[4:8] == b"ftyp"


class MiniMaxClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = spec.DEFAULT_BASE_URL,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        timeout: float = 60.0,
        download_attempt_timeout: float = spec.DOWNLOAD_ATTEMPT_TIMEOUT,
        allow_insecure_downloads: bool = False,
        sleep: SleepFn = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        register_secret(api_key)
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._sleep = sleep
        self._rng = rng
        self._dl_attempt_timeout = download_attempt_timeout
        self._allow_insecure_downloads = allow_insecure_downloads
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
            follow_redirects=False,  # never carry the bearer token to another location
            transport=transport,
        )

        async def _check_hop(request: httpx.Request) -> None:  # runs for every redirect hop
            check_download_url(request.url, self._allow_insecure_downloads)

        # No auth header: download URLs live on a storage host, not the API.
        self._dl = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=30.0),
            follow_redirects=True,
            max_redirects=5,
            event_hooks={"request": [_check_hop]},
            transport=transport,
        )

    async def __aenter__(self) -> "MiniMaxClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()
        await self._dl.aclose()

    # ------------------------------------------------------------------ public

    async def create_task(self, payload: dict[str, Any]) -> str:
        body = await self._request("POST", spec.CREATE_PATH, json=payload, idempotent=False)
        task_id = body.get("task_id")
        if not isinstance(task_id, (str, int)) or isinstance(task_id, bool) or not str(task_id).strip():
            raise AmbiguousSubmitError(
                f"create returned success without a usable task_id (keys: {sorted(body)[:10]}); "
                "a task may have been created - check the MiniMax console"
            )
        return sanitize(str(task_id).strip(), 200)

    async def query_task(self, task_id: str) -> dict[str, Any]:
        path = spec.QUERY_PATH.format(task_id=quote(task_id, safe=""))
        body = await self._request("GET", path, idempotent=True)
        task = body.get("task")
        if not isinstance(task, dict):
            raise ApiError(f"query response missing 'task' object (keys: {sorted(body)[:10]})")
        return task

    async def download(self, url: str, dest: Path) -> int:
        """Stream ``url`` to ``dest`` atomically. Returns bytes written."""
        check_download_url(url, self._allow_insecure_downloads)
        dest.parent.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.part")
            try:
                async with asyncio.timeout(self._dl_attempt_timeout):
                    written = await self._download_once(url, tmp)
                if not _looks_like_mp4(tmp):
                    raise UnsafeDownloadError("downloaded file is not an MP4 (no 'ftyp' box); discarded")
                os.replace(tmp, dest)
                return written
            except httpx.TooManyRedirects as exc:
                raise UnsafeDownloadError("download URL redirected too many times (max 5)") from exc
            except _RetryDownload as exc:
                if attempt >= self._max_retries:
                    raise exc.error from None
                await self._backoff(attempt, exc.retry_after)
                attempt += 1
            except (httpx.TransportError, TimeoutError) as exc:
                if attempt >= self._max_retries:
                    kind = "timed out" if isinstance(exc, TimeoutError) else f"network error: {exc}"
                    raise ApiError(f"download {kind}", retryable=True) from exc
                await self._backoff(attempt, None)
                attempt += 1
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

    async def _download_once(self, url: str, tmp: Path) -> int:
        async with self._dl.stream("GET", url) as resp:
            if resp.status_code in (401, 403, 404, 410):
                raise DownloadExpiredError(resp.status_code)
            if resp.status_code >= 400:
                err = ApiError(
                    f"download failed: {resp.reason_phrase}",
                    http_code=resp.status_code,
                    retryable=_is_retryable_status(resp.status_code),
                )
                if err.retryable:
                    raise _RetryDownload(err, resp.headers.get("retry-after"))
                raise err
            if resp.status_code != 200:
                raise ApiError(f"unexpected download status {resp.status_code}", http_code=resp.status_code)
            expected_raw = resp.headers.get("content-length")
            expected = int(expected_raw) if expected_raw and expected_raw.isdigit() else None
            if expected is not None and expected > spec.MAX_VIDEO_BYTES:
                raise UnsafeDownloadError(
                    f"download is {expected} bytes, over the {spec.MAX_VIDEO_BYTES} byte limit; refusing"
                )
            written = 0
            with open(tmp, "wb") as fh:
                async for chunk in resp.aiter_bytes(1 << 20):
                    written += len(chunk)
                    if written > spec.MAX_VIDEO_BYTES:
                        raise UnsafeDownloadError(
                            f"download exceeded the {spec.MAX_VIDEO_BYTES} byte limit; aborted"
                        )
                    fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())
        if expected is not None and expected != written:
            raise httpx.ReadError(f"truncated download: got {written} of {expected} bytes")
        if written == 0:
            raise httpx.ReadError("download returned an empty body")
        return written

    # ------------------------------------------------------------------ internals

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay: float | None = None
        if retry_after:
            try:
                delay = min(float(retry_after), 120.0)
            except ValueError:
                delay = None
            if delay is not None and not delay >= 0:  # nan / negative
                delay = None
        if delay is None:
            ceiling = min(self._backoff_cap, self._backoff_base * (2**attempt))
            delay = ceiling / 2 + self._rng() * ceiling / 2  # "equal jitter"
        await self._sleep(max(delay, 0.0))

    async def _request(
        self, method: str, path: str, *, json: Any = None, idempotent: bool
    ) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                resp = await self._http.request(method, path, json=json)
            except httpx.TransportError as exc:
                connect_phase = isinstance(exc, _CONNECT_PHASE_ERRORS)
                if not idempotent and not connect_phase:
                    raise AmbiguousSubmitError(
                        f"network error after submitting ({type(exc).__name__}); the task may or may "
                        "not have been created - check the MiniMax console before resubmitting"
                    ) from exc
                if attempt >= self._max_retries:
                    cls = ConnectionFailedError if connect_phase else ApiError
                    raise cls(f"connection error: {type(exc).__name__}: {exc}", retryable=True) from exc
                await self._backoff(attempt, None)
                attempt += 1
                continue

            body = self._parse_body(resp)
            err = self._extract_error(resp, body, query=method == "GET")
            if err is None:
                return body
            definitive_reject = err.http_code is not None and 400 <= err.http_code < 500 and err.http_code != 429
            if not idempotent and err.http_code != 429 and not definitive_reject:
                # 5xx, 3xx, 2xx-without-JSON...: the task may exist. Never resubmit blindly.
                detail = f"{err.error_type}: {err.message}" if err.error_type else err.message
                raise AmbiguousSubmitError(
                    f"create failed with an uncertain outcome ({detail}); a task may have been created - "
                    "check the MiniMax console before resubmitting",
                    http_code=err.http_code,
                    request_id=err.request_id,
                )
            if err.retryable and attempt < self._max_retries:
                await self._backoff(attempt, resp.headers.get("retry-after"))
                attempt += 1
                continue
            raise err

    @staticmethod
    def _parse_body(resp: httpx.Response) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError:
            return {"_raw": resp.text[:300]}
        return data if isinstance(data, dict) else {"_raw": str(data)[:300]}

    @staticmethod
    def _extract_error(resp: httpx.Response, body: dict[str, Any], *, query: bool = False) -> ApiError | None:
        status = resp.status_code
        if 300 <= status < 400:
            return ApiError(f"unexpected redirect from API ({status}); refusing to follow", http_code=status)
        if status < 400:
            if "_raw" in body:
                # 2xx but not JSON: a transient server/proxy problem (GET) / ambiguous (POST)
                return ApiError(f"non-JSON response: {body['_raw']!r}", http_code=status, retryable=True)
            if body.get("type") != "error":
                return None
        info = body.get("error") if isinstance(body.get("error"), dict) else {}
        code: int | None = status
        if status < 400:  # error envelope delivered with a 2xx status
            try:
                code = int(info.get("http_code"))
            except (TypeError, ValueError):
                code = None  # unknown severity: callers must treat it as uncertain
        message = info.get("message") or body.get("_raw") or resp.reason_phrase or "unknown error"
        if query and code is not None and code >= 500 and _TASK_NOT_FOUND_RE.search(str(message)):
            return TaskNotFoundError(str(message), http_code=code, error_type=info.get("type"),
                                     request_id=body.get("request_id"), retryable=False)
        return _error_class(code)(
            str(message),
            http_code=code,
            error_type=info.get("type"),
            request_id=body.get("request_id"),
            retryable=_is_retryable_status(code),
        )


class _RetryDownload(Exception):
    def __init__(self, error: ApiError, retry_after: str | None):
        self.error = error
        self.retry_after = retry_after
