"""Persistent per-output-directory state (``<out>/manifest.json``).

The manifest is the source of truth for "have we already paid for this job?".
It maps job name -> {task_id, status, file, error, params_hash, ...}. Every
update rewrites the file atomically (temp file + fsync + os.replace), guarded by
a lock so concurrent job workers can never interleave writes. A separate
advisory lock file prevents two ``minimax-batch run`` processes from sharing an
output directory at the same time.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".minimax-batch.lock"
MANIFEST_VERSION = 1

# Job statuses stored in the manifest
SUBMITTED = "submitted"  # create returned a task_id; not yet polled
QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"  # generation done, not (yet) downloaded
DOWNLOADED = "downloaded"  # final success state
FAILED = "failed"
CANCELLED = "cancelled"
TIMEOUT = "timeout"  # we gave up polling; task may still finish - resumable
SUBMITTING = "submitting"  # written *before* the create call; if seen on start-up the
#                            previous run died mid-submit and may have been charged
SUBMIT_UNKNOWN = "submit_unknown"  # create outcome uncertain; may have been charged
NOT_SUBMITTED = "not_submitted"  # batch aborted (401/402) before this job was submitted
TASK_LOST = "task_lost"  # server no longer knows our (paid-for) task_id; never auto-resubmitted

# Statuses with a live task_id that should be resumed (polled), never resubmitted.
RESUMABLE = frozenset({SUBMITTED, QUEUED, RUNNING, SUCCEEDED, TIMEOUT})
# Possibly-billed submissions whose outcome is unknown.
MID_SUBMIT = frozenset({SUBMITTING, SUBMIT_UNKNOWN})
# Blocked unless --retry-unknown: re-submitting could pay twice.
UNCERTAIN = MID_SUBMIT | {TASK_LOST}
# Nothing paid for is at risk: a re-run will submit a fresh task for these.
RESUBMITTABLE = frozenset({FAILED, CANCELLED, NOT_SUBMITTED})


class ManifestError(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self._data = data
        self._lock = threading.Lock()

    @classmethod
    def load(cls, out_dir: Path) -> "Manifest":
        path = Path(out_dir) / MANIFEST_NAME
        if not path.exists():
            return cls(path, {"version": MANIFEST_VERSION, "jobs": {}})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # Never silently discard: it holds task_ids we've already paid for.
            raise ManifestError(f"cannot read {path}: {exc}. Fix or move it aside before re-running.") from exc
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
            raise ManifestError(f"{path} is not a valid minimax-batch manifest")
        return cls(path, data)

    def get(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._data["jobs"].get(name)
            return copy.deepcopy(entry) if entry is not None else None

    def entries(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._data["jobs"])

    def update(self, name: str, **fields: Any) -> dict[str, Any]:
        """Merge ``fields`` into the entry for ``name`` and persist atomically."""
        with self._lock:
            entry = self._data["jobs"].setdefault(name, {"created_at": utc_now()})
            entry.update(fields)
            entry["updated_at"] = utc_now()
            self._write_locked()
            return copy.deepcopy(entry)

    def _write_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, sort_keys=True)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
            _fsync_dir(self.path.parent)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise


def _fsync_dir(directory: Path) -> None:
    """Make the rename durable (POSIX). Best effort elsewhere."""
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextlib.contextmanager
def run_lock(out_dir: Path) -> Iterator[None]:
    """Exclusive advisory lock so two runs can't share one output dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / LOCK_NAME
    fh = open(lock_path, "a+")
    try:
        try:
            import fcntl
        except ImportError:  # Windows: no advisory lock (documented in README)
            yield
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise ManifestError(
                    f"another minimax-batch run is using {out_dir} (lock: {lock_path})"
                ) from exc
            raise ManifestError(
                f"could not lock {lock_path}: {exc} (does the filesystem support flock?)"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()
