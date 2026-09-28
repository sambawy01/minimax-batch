"""Batch orchestration: submit, poll, download, persist - with bounded concurrency.

Money-safety invariants
-----------------------
* A job is only submitted when the manifest has no live task for it.
* Before the create call the entry is persisted as ``submitting``; only a
  returned task_id or a definitive 4xx rejection replaces that. If the process
  dies (SIGKILL, power loss) mid-submit, the next run sees ``submitting`` and
  refuses to resubmit without ``--retry-unknown``. Any other uncertain outcome
  (5xx, network error after send, cancellation, Ctrl-C) ends as
  ``submit_unknown``, which is treated the same way.
* The task_id is printed to the console *before* it is written to the
  manifest, so it is recoverable even if that write fails.
* On 401/402 no *new* submissions are made; jobs that already have a task_id
  keep polling/downloading (that costs nothing).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from rich.console import Console
from rich.markup import escape

from . import manifest as mf
from . import spec
from .api import (
    AmbiguousSubmitError,
    ApiError,
    ConnectionFailedError,
    DownloadExpiredError,
    FatalApiErrors,
    MiniMaxClient,
    TaskNotFoundError,
    sanitize,
)
from .batch import Job

OK_OUTCOMES = frozenset({"downloaded", "skipped"})


@dataclass
class RunOptions:
    concurrency: int = 3
    poll_initial: float = 5.0
    poll_max: float = 30.0
    poll_factor: float = 1.5
    job_timeout: float = 20 * 60.0
    rerun_changed: bool = False
    retry_unknown: bool = False
    base_url: str | None = None  # recorded per entry; resuming against another API host is refused
    not_found_grace: float = 60.0  # seconds a *fresh* task may be "not found" before task_lost


@dataclass
class JobResult:
    name: str
    outcome: str  # downloaded|skipped|failed|cancelled|timeout|aborted|blocked|submit_unknown|task_lost|error
    detail: str = ""
    task_id: str | None = None
    file: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome in OK_OUTCOMES


# Statuses whose task may still be running / billed: abandoning them needs both flags.
_LIVE_STATUSES = frozenset({mf.SUBMITTED, mf.QUEUED, mf.RUNNING, mf.TIMEOUT}) | mf.UNCERTAIN


def _fmt_secs(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 60 else f"{seconds / 60:.0f} min"


def _norm_base(url: str | None) -> str:
    return str(url or "").strip().rstrip("/").lower()


def plan_action(job: Job, entry: dict[str, Any] | None, out_dir: Path, opts: RunOptions) -> tuple[str, str]:
    """Decide what to do for a job given its manifest entry. Pure; no I/O except stat.

    Returns (action, reason) where action is one of: skip, resume, submit, blocked.
    """
    action, reason = _plan_core(job, entry, out_dir, opts)
    # Resuming polls the *old* task, which only exists on the host that created it. A new
    # submission (already gated by --retry-unknown / --rerun-changed) may go anywhere; the
    # old task and its host are kept in previous_tasks.
    if entry and action == "resume":
        recorded = entry.get("base_url")
        if (recorded and opts.base_url
                and _norm_base(recorded) != _norm_base(opts.base_url)):
            return "blocked", (
                f"its task belongs to {recorded}, not {opts.base_url}; pass the same --base-url "
                "to resume it"
            )
    return action, reason


def _plan_core(job: Job, entry: dict[str, Any] | None, out_dir: Path, opts: RunOptions) -> tuple[str, str]:
    dest = out_dir / job.filename
    if entry is None:
        if dest.exists():
            return "blocked", f"{dest} exists but is not in the manifest; delete or rename it to regenerate"
        return "submit", "new job"

    status = entry.get("status")
    task_id = entry.get("task_id")
    old_hash = entry.get("params_hash")
    changed = bool(old_hash) and old_hash != job.params_hash

    if changed and status not in mf.RESUBMITTABLE:
        if not opts.rerun_changed:
            return "blocked", (
                "parameters changed since the last run; pass --rerun-changed to generate it again "
                "(this is charged again)"
            )
        if status in _LIVE_STATUSES and not opts.retry_unknown:
            return "blocked", (
                f"parameters changed but the previous task is {status} and may still be billed; pass "
                "--retry-unknown together with --rerun-changed to abandon it and submit a new one"
            )
        return "submit", "parameters changed (--rerun-changed)"

    if status in mf.UNCERTAIN:
        if not opts.retry_unknown:
            what = {
                mf.SUBMITTING: "a previous run was interrupted while submitting",
                mf.TASK_LOST: "the server no longer knows its previous (paid) task",
            }.get(status, "a previous submit had an uncertain outcome")
            return "blocked", (
                f"{what} and may have been charged; check the MiniMax console, then pass "
                "--retry-unknown to submit again"
            )
        return "submit", f"previous status {status} (--retry-unknown)"

    if status == mf.DOWNLOADED:
        if dest.exists():
            return "skip", "already downloaded"
        if task_id:
            return "resume", "output file missing; re-fetching from the existing task"
        return "submit", "output file missing and no task_id recorded"
    if status in mf.RESUMABLE and task_id:
        return "resume", f"resuming existing task ({status})"
    return "submit", f"previous status: {status or 'none'}"


class _JobStop(Exception):
    """Internal: stop processing a job with a given result."""

    def __init__(self, result: JobResult):
        self.result = result


class Runner:
    def __init__(
        self,
        client: MiniMaxClient,
        manifest: mf.Manifest,
        out_dir: Path,
        opts: RunOptions | None = None,
        console: Console | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.client = client
        self.manifest = manifest
        self.out_dir = Path(out_dir)
        self.opts = opts or RunOptions()
        self.console = console or Console(stderr=True)
        self._sleep = sleep
        self._clock = clock
        self._abort: asyncio.Event | None = None
        self._abort_reason = ""
        self._width = 10

    # ------------------------------------------------------------------ public

    async def run(self, jobs: list[Job]) -> list[JobResult]:
        self._abort = asyncio.Event()
        self._width = max((len(j.name) for j in jobs), default=10)
        sem = asyncio.Semaphore(max(1, self.opts.concurrency))
        return list(await asyncio.gather(*(self._guarded(job, sem) for job in jobs)))

    @property
    def abort_reason(self) -> str:
        return self._abort_reason

    # ------------------------------------------------------------------ internals

    def _log(self, job: Job, msg: str, style: str = "") -> None:
        name = escape(job.name.ljust(self._width))
        text = escape(sanitize(msg, 2000))
        self.console.print(f"[bold]{name}[/bold]  " + (f"[{style}]{text}[/{style}]" if style else text))

    def _trigger_abort(self, exc: Exception) -> None:
        assert self._abort is not None
        if not self._abort.is_set():
            self._abort_reason = sanitize(exc)
            self._abort.set()
            self.console.print(f"[bold red]Aborting new submissions:[/bold red] {escape(self._abort_reason)}")

    def _task_id_of(self, job: Job) -> str | None:
        entry = self.manifest.get(job.name)
        return entry.get("task_id") if entry else None

    async def _guarded(self, job: Job, sem: asyncio.Semaphore) -> JobResult:
        async with sem:
            try:
                return await self._process(job)
            except _JobStop as stop:
                return stop.result
            except FatalApiErrors as exc:
                self._trigger_abort(exc)
                msg = sanitize(exc)
                self._log(job, f"error: {msg}", "red")
                if self.manifest.get(job.name) is not None:
                    self.manifest.update(job.name, error=msg)
                return JobResult(job.name, "error", msg, self._task_id_of(job))
            except Exception as exc:  # noqa: BLE001 - one job must not kill the batch
                msg = sanitize(f"{type(exc).__name__}: {exc}")
                self._log(job, f"error: {msg}", "red")
                try:
                    if self.manifest.get(job.name) is not None:
                        self.manifest.update(job.name, error=msg)
                except Exception:  # noqa: BLE001 - the manifest itself may be what failed
                    pass
                return JobResult(job.name, "error", msg, self._task_id_of(job))

    async def _process(self, job: Job) -> JobResult:
        entry = self.manifest.get(job.name)
        action, reason = plan_action(job, entry, self.out_dir, self.opts)

        if action == "skip":
            self._log(job, f"skip ({reason})", "dim")
            return JobResult(job.name, "skipped", reason, entry.get("task_id") if entry else None,
                             entry.get("file") if entry else None)
        if action == "blocked":
            self._log(job, f"blocked: {reason}", "yellow")
            return JobResult(job.name, "blocked", reason, entry.get("task_id") if entry else None)

        fresh = action == "submit"
        if fresh:
            task_id = await self._submit(job, entry or {})
        else:
            task_id = entry["task_id"]  # type: ignore[index]
            self._log(job, f"{reason} task={task_id}", "cyan")

        task = await self._poll(job, task_id, first_delay=self.opts.poll_initial if fresh else 0.0, fresh=fresh)
        status = str(task.get("status", "")).lower()

        if status == spec.STATUS_SUCCEEDED:
            return await self._download(job, task_id, task)

        err = task.get("error") or {}
        if not isinstance(err, dict):
            err = {"message": err}
        detail = sanitize(": ".join(str(x) for x in (err.get("code"), err.get("message")) if x) or status)
        mstatus = mf.CANCELLED if status == spec.STATUS_CANCELLED else mf.FAILED
        self.manifest.update(job.name, status=mstatus, error=detail)
        self._log(job, f"{mstatus}: {detail}", "red")
        return JobResult(job.name, mstatus, detail, task_id)

    async def _submit(self, job: Job, entry: dict[str, Any]) -> str:
        assert self._abort is not None
        # previous_task_ids: plain ids (kept for backward compatibility).
        # previous_tasks: [{task_id, base_url, status}] - includes uncertain attempts with no id.
        previous = list(entry.get("previous_task_ids") or [])
        if entry.get("task_id") and entry["task_id"] not in previous:
            previous.append(entry["task_id"])
        previous_tasks = list(entry.get("previous_tasks") or [])
        if entry.get("task_id") or entry.get("status") in _LIVE_STATUSES:
            previous_tasks.append({"task_id": entry.get("task_id"), "base_url": entry.get("base_url"),
                                   "status": entry.get("status")})
        base_fields = {
            "params_hash": job.params_hash,
            "model": job.params["model"],
            "resolution": job.params["resolution"],
            "duration": job.params["duration"],
            "ratio": job.params["ratio"],
            "previous_task_ids": previous,
            "previous_tasks": previous_tasks,
            "task_id": None,
            "file": None,
            "base_url": self.opts.base_url,
        }
        prior_status = entry.get("status")

        def mark_not_submitted(error: str) -> None:
            """This attempt created nothing. An entry that was already possibly-billed keeps
            its previous state (it must not be downgraded to a resubmittable one)."""
            if prior_status in mf.UNCERTAIN:
                self.manifest.update(
                    job.name, status=prior_status, task_id=entry.get("task_id"),
                    previous_task_ids=list(entry.get("previous_task_ids") or []),
                    previous_tasks=list(entry.get("previous_tasks") or []),
                    params_hash=entry.get("params_hash"), base_url=entry.get("base_url"), error=error,
                )
            else:
                self.manifest.update(job.name, status=mf.NOT_SUBMITTED, error=error, **base_fields)

        if self._abort.is_set():
            mark_not_submitted(sanitize(f"not submitted: batch aborted ({self._abort_reason})"))
            self._log(job, "not submitted (batch aborted)", "yellow")
            raise _JobStop(JobResult(job.name, "aborted", self._abort_reason))

        self._log(job, f"submitting ({job.params['model']}, {job.params['resolution']}, "
                       f"{job.params['duration']}s, {job.params['ratio']})")
        # Persist intent BEFORE spending money: a crash from here on leaves "submitting".
        self.manifest.update(job.name, status=mf.SUBMITTING, error=None, **base_fields)
        try:
            task_id = await self.client.create_task(job.payload)
        except AmbiguousSubmitError as exc:
            msg = sanitize(exc)
            self.manifest.update(job.name, status=mf.SUBMIT_UNKNOWN, error=msg)
            self._log(job, f"submit outcome unknown: {msg}", "red")
            raise _JobStop(JobResult(job.name, "submit_unknown", msg))
        except FatalApiErrors as exc:
            mark_not_submitted(sanitize(exc))
            raise
        except ConnectionFailedError as exc:
            msg = sanitize(exc)
            if prior_status in mf.UNCERTAIN:
                mark_not_submitted(msg)  # never reached the server: nothing changed
            else:
                self.manifest.update(job.name, status=mf.FAILED, error=msg)
            self._log(job, f"connection error (not submitted): {msg}", "red")
            raise _JobStop(JobResult(job.name, "failed", msg))
        except ApiError as exc:
            msg = sanitize(exc)
            if exc.http_code is not None and 400 <= exc.http_code < 500:
                # Definitive for *this* attempt; an earlier possibly-billed attempt stays uncertain.
                if prior_status in mf.UNCERTAIN:
                    mark_not_submitted(msg)
                else:
                    self.manifest.update(job.name, status=mf.FAILED, error=msg)
                self._log(job, f"rejected by API: {msg}", "red")
                raise _JobStop(JobResult(job.name, "failed", msg))
            self.manifest.update(job.name, status=mf.SUBMIT_UNKNOWN, error=msg)
            self._log(job, f"submit outcome unknown: {msg}", "red")
            raise _JobStop(JobResult(job.name, "submit_unknown", msg))
        except BaseException as exc:  # CancelledError, KeyboardInterrupt, bugs: outcome unknown
            self.manifest.update(
                job.name, status=mf.SUBMIT_UNKNOWN,
                error=sanitize(f"interrupted during submit ({type(exc).__name__}); a task may have "
                               "been created - check the MiniMax console"),
            )
            raise

        # Paid for. Print first (recoverable from the console), then persist.
        self._log(job, f"submitted task={task_id}", "green")
        self.manifest.update(job.name, status=mf.SUBMITTED, task_id=task_id, error=None,
                             submitted_at=mf.utc_now())
        return task_id

    async def _poll(self, job: Job, task_id: str, first_delay: float, fresh: bool = False) -> dict[str, Any]:
        start = self._clock()
        deadline = start + self.opts.job_timeout
        # A task we just created may not be queryable for a moment (eventual consistency).
        not_found_until = start + self.opts.not_found_grace if fresh else start
        delay = self.opts.poll_initial
        last_status = None
        if first_delay > 0:
            await self._sleep(first_delay)
            delay = min(delay * self.opts.poll_factor, self.opts.poll_max)
        while True:
            try:
                task = await self.client.query_task(task_id)
            except FatalApiErrors:
                raise
            except TaskNotFoundError as exc:
                if self._clock() < not_found_until:
                    if last_status != "not-found":
                        self._log(job, f"task {task_id} not visible yet; retrying", "dim")
                        last_status = "not-found"
                    await self._sleep(delay)
                    delay = min(delay * self.opts.poll_factor, self.opts.poll_max)
                    continue
                msg = sanitize(
                    f"task {task_id} is no longer known to the server ({exc}). It was already submitted "
                    "(and may have been billed), so it is NOT resubmitted automatically; pass "
                    "--retry-unknown to generate it again"
                )
                self.manifest.update(job.name, status=mf.TASK_LOST, error=msg)
                self._log(job, f"task lost: {msg}", "red")
                raise _JobStop(JobResult(job.name, "task_lost", msg, task_id))
            except ApiError as exc:
                msg = sanitize(exc)
                self.manifest.update(job.name, error=f"poll failed: {msg}")
                self._log(job, f"poll failed: {msg} (re-run to resume task {task_id})", "red")
                raise _JobStop(JobResult(job.name, "error", msg, task_id))

            status = sanitize(task.get("status", ""), 40).lower()
            if status in spec.TERMINAL_TASK_STATUSES:
                return task
            if status != last_status:
                elapsed = self._clock() - start
                self._log(job, f"{status or 'unknown status'} ({elapsed:.0f}s)")
                if status in (spec.STATUS_QUEUED, spec.STATUS_RUNNING):
                    self.manifest.update(job.name, status=status)
                last_status = status
            if self._clock() >= deadline:
                msg = (f"still {status or 'pending'} after {_fmt_secs(self.opts.job_timeout)}; "
                       f"re-run to keep waiting on task {task_id}")
                self.manifest.update(job.name, status=mf.TIMEOUT, error=msg)
                self._log(job, f"timeout: {msg}", "yellow")
                raise _JobStop(JobResult(job.name, "timeout", msg, task_id))
            await self._sleep(delay)
            delay = min(delay * self.opts.poll_factor, self.opts.poll_max)

    async def _download(self, job: Job, task_id: str, task: dict[str, Any]) -> JobResult:
        dest = self.out_dir / job.filename
        self.manifest.update(job.name, status=mf.SUCCEEDED, error=None, usage=task.get("usage"))
        url = (task.get("content") or {}).get("url")
        try:
            try:
                if not url:
                    raise DownloadExpiredError(0)
                size = await self.client.download(str(url), dest)
            except DownloadExpiredError as exc:
                self._log(job, f"{exc}; requesting a fresh URL", "yellow")
                fresh = await self.client.query_task(task_id)
                url = (fresh.get("content") or {}).get("url")
                if str(fresh.get("status", "")).lower() != spec.STATUS_SUCCEEDED or not url:
                    raise ApiError("could not obtain a fresh download URL") from exc
                size = await self.client.download(str(url), dest)
        except FatalApiErrors:
            raise
        except (ApiError, DownloadExpiredError, OSError) as exc:
            msg = sanitize(f"download failed: {exc}")
            self.manifest.update(job.name, error=msg)
            self._log(job, f"{msg} (re-run to retry; the video is already paid for)", "red")
            return JobResult(job.name, "error", msg, task_id)

        self.manifest.update(job.name, status=mf.DOWNLOADED, file=job.filename, bytes=size, error=None)
        self._log(job, f"downloaded -> {dest} ({size / 1e6:.1f} MB)", "bold green")
        return JobResult(job.name, "downloaded", "", task_id, job.filename)
