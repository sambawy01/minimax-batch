"""Batch file loading, parameter resolution, validation and payload building.

Everything here is pure / local: no network calls are made, so the whole batch
can be validated (and rejected) before a single credit is spent.
"""

from __future__ import annotations

import base64
import binascii
import copy
import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import spec

OVERRIDE_KEYS = ("model", "resolution", "duration", "ratio", "prompt_expansion")
IMAGE_KEYS = ("first_frame", "last_frame", "reference_images")
JOB_KEYS = frozenset(("name", "prompt", *IMAGE_KEYS, *OVERRIDE_KEYS))

_DATA_URI_RE = re.compile(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)
_ALLOWED_DATA_SUBTYPES = {"jpeg", "jpg", "png", "webp", "heic", "heif"}


class BatchFileError(Exception):
    """The batch file could not be read or has the wrong overall structure."""


class ValidationError(Exception):
    """One or more jobs failed validation. ``errors`` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s)")


@dataclass
class RawBatch:
    defaults: dict[str, Any]
    jobs: list[dict[str, Any]]
    base_dir: Path  # relative image paths resolve against this


@dataclass(frozen=True)
class Job:
    index: int  # 1-based position in the batch file
    name: str  # slugified, unique; used as output filename and manifest key
    params: dict[str, Any]  # resolved model/resolution/duration/ratio/prompt_expansion
    payload: dict[str, Any]  # exact request body for POST /v2/video_generation
    params_hash: str
    warnings: tuple[str, ...] = field(default=())

    @property
    def filename(self) -> str:
        return f"{self.name}.mp4"


# --------------------------------------------------------------------------- loading


def load_batch_file(path: str | Path) -> RawBatch:
    path = Path(path)
    if not path.is_file():
        raise BatchFileError(f"batch file not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            defaults, jobs = _load_yaml(path)
        elif suffix == ".csv":
            defaults, jobs = {}, _load_csv(path)
        else:
            raise BatchFileError(f"unsupported batch file type {_r(suffix)} (use .yaml, .yml or .csv)")
    except (yaml.YAMLError, csv.Error, UnicodeDecodeError) as exc:
        raise BatchFileError(f"could not parse {path}: {exc}") from exc
    if not jobs:
        raise BatchFileError(f"no jobs found in {path}")
    return RawBatch(defaults=defaults, jobs=jobs, base_dir=path.resolve().parent)


def _load_yaml(path: Path) -> tuple[dict, list[dict]]:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_NoAliasLoader)  # noqa: S506 - safe subclass
    if isinstance(data, list):
        return {}, data
    if not isinstance(data, dict):
        raise BatchFileError("YAML batch must be a mapping with 'jobs:' (and optional 'defaults:') or a list of jobs")
    unknown = set(data) - {"defaults", "jobs"}
    if unknown:
        raise BatchFileError(f"unknown top-level YAML key(s): {', '.join(sorted(unknown))}")
    defaults = data.get("defaults") or {}
    jobs = data.get("jobs") or []
    if not isinstance(defaults, dict):
        raise BatchFileError("'defaults' must be a mapping")
    if not isinstance(jobs, list):
        raise BatchFileError("'jobs' must be a list")
    return defaults, jobs


def _load_csv(path: Path) -> list[dict]:
    jobs: list[dict] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return []
        for row in reader:
            job: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:  # more cells than headers
                    raise BatchFileError(f"CSV row {reader.line_num} has more cells than header columns")
                key = key.strip()
                if value is None:
                    continue
                value = value.strip()
                if value == "":
                    continue  # blank cell = not set
                if key == "reference_images":
                    job[key] = [v.strip() for v in value.split(";") if v.strip()]
                else:
                    job[key] = value
            if job:  # skip fully blank lines
                jobs.append(job)
    return jobs


# --------------------------------------------------------------------------- helpers


def _r(value: Any, limit: int = 60) -> str:
    """Truncated repr for error messages."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 4] + "...'"


_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs"}
_HEIF_GENERIC = {b"mif1", b"msf1"}


def sniff_image_mime(head: bytes) -> str | None:
    """Identify an allowed image type from its magic bytes."""
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in _HEIF_BRANDS:
            return "image/heic"
        if brand in _HEIF_GENERIC:
            return "image/heif"
    return None


class _NoAliasLoader(yaml.SafeLoader):
    """SafeLoader that refuses YAML aliases (defends against alias-expansion bombs)."""

    def compose_node(self, parent, index):  # type: ignore[override]
        if self.check_event(yaml.AliasEvent):
            event = self.peek_event()
            raise yaml.composer.ComposerError(
                None, None, "YAML aliases (*name) are not allowed in batch files", event.start_mark
            )
        return super().compose_node(parent, index)


def slugify(value: str, max_len: int = 80) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:max_len].strip("-")


def _normalize_resolution(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


def _parse_duration(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        return int(value)
    return None


def _encode_image(value: Any, base_dir: Path, allow_external: bool = False) -> tuple[str | None, str | None]:
    """Return (url_for_api, error). Local files become base64 data URIs."""
    if not isinstance(value, str) or not value.strip():
        return None, "must be a non-empty string (local path, http(s) URL or data URI)"
    value = value.strip()
    if value.startswith(("http://", "https://")):
        return value, None
    if value.startswith("data:"):
        m = _DATA_URI_RE.match(value)
        if not m or m.group(1).lower() not in _ALLOWED_DATA_SUBTYPES:
            return None, "data URI must be data:image/<jpeg|png|webp|heic|heif>;base64,..."
        data = m.group(2)
        approx = len(data) * 3 // 4
        if approx > spec.MAX_IMAGE_BYTES:
            return None, f"image is ~{approx / 1e6:.1f}MB, exceeds 30MB limit"
        try:
            head = base64.b64decode(data[:24], validate=True)
        except (ValueError, binascii.Error):
            return None, "data URI does not contain valid base64"
        if sniff_image_mime(head) is None:
            return None, "data URI content is not a JPEG/PNG/WEBP/HEIC/HEIF image"
        return value, None

    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    ext = p.suffix.lower().lstrip(".")
    if ext not in spec.IMAGE_MIME:
        return None, f"unsupported image type {_r(p.suffix or '(none)')} for {_r(value)} (allowed: jpg, jpeg, png, webp, heic, heif)"
    try:
        resolved = p.resolve(strict=True)
    except (OSError, RuntimeError):
        return None, f"image file not found: {_r(str(p), 200)}"
    if not resolved.is_file():
        return None, f"image path is not a regular file: {_r(str(p), 200)}"
    if not allow_external:
        try:
            resolved.relative_to(base_dir.resolve())
        except ValueError:
            return None, (
                f"image {_r(value, 120)} resolves outside the batch file's directory; "
                "move it next to the batch file or pass --allow-external-images"
            )
    size = resolved.stat().st_size
    if size > spec.MAX_IMAGE_BYTES:
        return None, f"image {resolved.name} is {size / 1e6:.1f}MB, exceeds 30MB limit"
    if size == 0:
        return None, f"image {resolved.name} is empty"
    raw = resolved.read_bytes()
    mime = sniff_image_mime(raw[:16])
    if mime is None:
        return None, f"image {resolved.name} is not a valid JPEG/PNG/WEBP/HEIC/HEIF file (magic bytes)"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", None


def params_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy of payload with base64 image data replaced by its length + short sha256."""
    out = copy.deepcopy(payload)
    for item in out.get("content", []):
        if item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url", "")
            if url.startswith("data:") and "," in url:
                head, data = url.split(",", 1)
                digest = hashlib.sha256(data.encode("ascii", "replace")).hexdigest()[:12]
                item["image_url"]["url"] = f"{head},<{len(data)} base64 chars, sha256:{digest}>"
    return out


# --------------------------------------------------------------------------- resolution


def resolve_jobs(
    raw: RawBatch, cli_overrides: dict[str, Any] | None = None, *, allow_external_images: bool = False
) -> list[Job]:
    """Merge defaults, validate every job and build payloads.

    Precedence (lowest -> highest): built-in defaults < YAML ``defaults:`` <
    CLI flags < per-job fields. Raises ValidationError listing *all* problems.
    """
    errors: list[str] = []
    cli_overrides = {k: v for k, v in (cli_overrides or {}).items() if v is not None}

    unknown_defaults = set(raw.defaults) - set(OVERRIDE_KEYS)
    if unknown_defaults:
        errors.append(
            f"defaults: unknown key(s) {', '.join(sorted(unknown_defaults))} "
            f"(allowed: {', '.join(OVERRIDE_KEYS)})"
        )

    base = dict(spec.BUILTIN_DEFAULTS)
    base.update({k: v for k, v in raw.defaults.items() if k in OVERRIDE_KEYS and v is not None})
    base.update(cli_overrides)

    jobs: list[Job] = []
    seen: dict[str, int] = {}
    for idx, rawjob in enumerate(raw.jobs, start=1):
        job, job_errors = _resolve_one(idx, rawjob, base, raw.base_dir, allow_external_images)
        errors.extend(job_errors)
        if job is None:
            continue
        if job.name in seen:
            errors.append(
                f"job #{idx} ({job.name}): name collides with job #{seen[job.name]} after slugify; names must be unique"
            )
            continue
        seen[job.name] = idx
        jobs.append(job)

    if errors:
        raise ValidationError(errors)
    return jobs


def _resolve_one(
    idx: int, rawjob: Any, base: dict, base_dir: Path, allow_external: bool
) -> tuple[Job | None, list[str]]:
    if not isinstance(rawjob, dict):
        return None, [f"job #{idx}: must be a mapping, got {type(rawjob).__name__}"]

    raw_name = rawjob.get("name")
    label = f"job #{idx}" + (f" ({str(raw_name)[:60]})" if raw_name else "")
    errs: list[str] = []
    warnings: list[str] = []

    def err(msg: str) -> None:
        errs.append(f"{label}: {msg}")

    unknown = set(rawjob) - JOB_KEYS
    if unknown:
        err(f"unknown field(s) {', '.join(sorted(unknown))} (allowed: {', '.join(sorted(JOB_KEYS))})")

    # name
    name = ""
    if not isinstance(raw_name, (str, int)) or not str(raw_name).strip():
        err("'name' is required")
    else:
        name = slugify(str(raw_name))
        if not name:
            err(f"name {_r(raw_name)} has no usable characters for a filename")

    # merge
    params = dict(base)
    for key in OVERRIDE_KEYS:
        if key in rawjob and rawjob[key] is not None:
            params[key] = rawjob[key]

    model = params["model"]
    model_spec = spec.MODELS.get(model) if isinstance(model, str) else None
    if model_spec is None:
        err(f"model {_r(model)} is invalid (allowed: {', '.join(spec.MODELS)})")

    resolution = _normalize_resolution(params["resolution"])
    params["resolution"] = resolution
    if model_spec and resolution not in model_spec["resolutions"]:
        err(f"resolution {_r(resolution)} not supported by {model} (allowed: {', '.join(model_spec['resolutions'])})")

    duration = _parse_duration(params["duration"])
    if duration is None:
        err(f"duration {_r(params['duration'])} must be an integer number of seconds")
    else:
        params["duration"] = duration
        if model_spec:
            lo, hi = model_spec["duration"]
            if not lo <= duration <= hi:
                err(f"duration {duration}s out of range for {model} ({lo}-{hi}s)")

    pe = params.get("prompt_expansion")
    if pe is not None:
        if not isinstance(pe, str) or pe.strip().lower() not in spec.PROMPT_EXPANSION_MODES:
            err(f"prompt_expansion {_r(pe)} invalid (allowed: {', '.join(spec.PROMPT_EXPANSION_MODES)})")
        else:
            params["prompt_expansion"] = pe.strip().lower()

    # prompt
    prompt = rawjob.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        err(f"prompt must be text, got {type(prompt).__name__} (quote it in YAML)")
        prompt = ""
    prompt = (prompt or "").strip()

    # images
    frames: dict[str, str] = {}
    for role in ("first_frame", "last_frame"):
        if rawjob.get(role) is not None:
            url, e = _encode_image(rawjob[role], base_dir, allow_external)
            if e:
                err(f"{role}: {e}")
            else:
                frames[role] = url
    refs: list[str] = []
    raw_refs = rawjob.get("reference_images")
    if raw_refs is not None:
        if isinstance(raw_refs, str):
            raw_refs = [raw_refs]
        if not isinstance(raw_refs, list):
            err("reference_images must be a list")
        elif len(raw_refs) > spec.MAX_REFERENCE_IMAGES:
            err(f"at most {spec.MAX_REFERENCE_IMAGES} reference_images allowed, got {len(raw_refs)}")
        else:
            for i, r in enumerate(raw_refs, start=1):
                url, e = _encode_image(r, base_dir, allow_external)
                if e:
                    err(f"reference_images[{i}]: {e}")
                else:
                    refs.append(url)

    has_frame = rawjob.get("first_frame") is not None or rawjob.get("last_frame") is not None
    if not prompt and not has_frame:
        err("'prompt' is required for text-to-video jobs")
    if len(prompt) > spec.MAX_PROMPT_CHARS:
        err(f"prompt is {len(prompt)} chars, max {spec.MAX_PROMPT_CHARS}")

    # ratio
    ratio = params["ratio"]
    if isinstance(ratio, int) and not isinstance(ratio, bool):
        err(f"ratio loaded as the number {ratio}: quote ratios in YAML (e.g. ratio: \"16:9\")")
    elif not isinstance(ratio, str) or ratio.strip() not in spec.RATIOS:
        err(f"ratio {_r(ratio)} invalid (allowed: {', '.join(spec.RATIOS)})")
    else:
        ratio = ratio.strip()
        if has_frame:
            if ratio != "adaptive":
                if rawjob.get("ratio") is not None:
                    warnings.append(f"ratio {ratio} overridden to 'adaptive' (required when a first/last frame is given)")
                ratio = "adaptive"
        elif ratio == "adaptive":
            err("ratio 'adaptive' is only allowed when first_frame/last_frame is given; set e.g. 16:9")
        params["ratio"] = ratio

    if errs:
        return None, errs

    content: list[dict[str, Any]] = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    for role in ("first_frame", "last_frame"):
        if role in frames:
            content.append({"type": "image_url", "image_url": {"url": frames[role]}, "role": role})
    for url in refs:
        content.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})

    payload: dict[str, Any] = {
        "model": params["model"],
        "content": content,
        "resolution": params["resolution"],
        "duration": params["duration"],
        "ratio": params["ratio"],
    }
    if params.get("prompt_expansion"):
        payload["extra"] = {"prompt_expansion_mode": params["prompt_expansion"]}

    body_size = len(json.dumps(payload).encode("utf-8"))
    if body_size > spec.MAX_BODY_BYTES:
        return None, [f"{label}: request body is {body_size / 1e6:.1f}MB, exceeds 64MB limit"]

    return (
        Job(
            index=idx,
            name=name,
            params=params,
            payload=payload,
            params_hash=params_hash(payload),
            warnings=tuple(warnings),
        ),
        [],
    )
