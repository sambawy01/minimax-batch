"""Command-line entry point: ``minimax-batch run|status``."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import os
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import httpx
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__, spec
from . import manifest as mf
from .api import MiniMaxClient, sanitize
from .batch import BatchFileError, Job, ValidationError, load_batch_file, redact_payload, resolve_jobs
from .runner import RunOptions, Runner, plan_action

ENV_KEY = "MINIMAX_API_KEY"

EXIT_OK = 0
EXIT_JOB_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


# --------------------------------------------------------------------------- env


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_api_key(cwd: Path | None = None, warn: Callable[[str], None] | None = None) -> str | None:
    """``MINIMAX_API_KEY`` from the environment, else from ``.env`` in cwd."""
    key = os.environ.get(ENV_KEY, "").strip()
    if key:
        return key
    path = (cwd or Path.cwd()) / ".env"
    if not path.is_file():
        return None
    try:
        mode = path.stat().st_mode
    except OSError:
        mode = 0
    if os.name == "posix" and mode & (stat.S_IRGRP | stat.S_IROTH) and warn is not None:
        warn(f"{path} is readable by other users; consider: chmod 600 {path}")
    key = _parse_dotenv(path).get(ENV_KEY, "").strip()
    return key or None


def check_base_url(url: str, allow_custom_host: bool) -> str | None:
    """Return an error message if the API key must not be sent to ``url``.

    Only MiniMax hosts are accepted by default; any other host (localhost included)
    needs ``allow_custom_host``. Plain http is only allowed for loopback hosts. The URL
    is parsed with httpx so the check sees exactly the host httpx will connect to.
    """
    if "\\" in url or any(ord(c) < 0x21 or ord(c) == 0x7F for c in url):
        return "invalid --base-url: backslashes, spaces and control characters are not allowed"
    try:
        u = httpx.URL(url)
    except (httpx.InvalidURL, TypeError, ValueError):
        return "invalid --base-url"
    host = (u.host or "").lower()
    if u.scheme not in ("http", "https") or not host:
        return "invalid --base-url (need http(s)://host)"
    if "%" in host or "%" in u.netloc.decode("ascii", "replace").split("@")[-1]:
        return "invalid --base-url: percent-encoding is not allowed in the host"
    if u.userinfo:
        return "--base-url must not contain credentials"
    try:
        loopback = ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if u.scheme == "http" and not loopback:
        return "--base-url must use https (plain http is only allowed for localhost / loopback)"
    trusted = host in spec.TRUSTED_API_HOSTS or host.endswith(spec.TRUSTED_API_HOST_SUFFIXES)
    if not trusted and not allow_custom_host:
        return (f"refusing to send your API key to {host}; only MiniMax hosts are allowed by default. "
                "Pass --allow-custom-host if this host is intended (e.g. a local mock server)")
    return None


# --------------------------------------------------------------------------- parser


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _positive_float(value: str) -> float:
    try:
        n = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if not math.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("must be a finite number > 0")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="minimax-batch",
        description="Batch-generate video clips with the MiniMax (Hailuo) V2 video API.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="validate and run a batch file (YAML or CSV)")
    run.add_argument("batch", type=Path, help="batch file (.yaml/.yml/.csv)")
    run.add_argument("--out", type=Path, default=Path("output"), help="output directory (default: output/)")
    run.add_argument("--concurrency", type=_positive_int, default=3, help="jobs in flight at once (default: 3)")
    run.add_argument("--dry-run", action="store_true", help="validate and print request payloads; no network calls")
    g = run.add_argument_group("defaults (override the batch file's defaults; per-job fields still win)")
    g.add_argument("--model", choices=list(spec.MODELS))
    g.add_argument("--resolution", help="768P or 2K (MiniMax-H3); 480P or 768P (MiniMax-H3-Max)")
    g.add_argument("--duration", type=int, help="seconds: 4-15 (MiniMax-H3), 5-15 (MiniMax-H3-Max)")
    g.add_argument("--ratio", choices=list(spec.RATIOS))
    g.add_argument("--prompt-expansion", choices=list(spec.PROMPT_EXPANSION_MODES))
    run.add_argument("--timeout", type=_positive_float, default=20.0, metavar="MINUTES",
                     help="per-job polling timeout in minutes (default: 20)")
    run.add_argument("--base-url", default=spec.DEFAULT_BASE_URL, help=f"API base URL (default: {spec.DEFAULT_BASE_URL})")
    run.add_argument("--allow-custom-host", action="store_true",
                     help="allow a --base-url host other than api.minimax.io / *.minimax.io / api.minimaxi.com")
    run.add_argument("--allow-external-images", action="store_true",
                     help="allow local image paths outside the batch file's directory")
    run.add_argument("--rerun-changed", action="store_true",
                     help="regenerate jobs whose parameters changed since the last run (charged again)")
    run.add_argument("--retry-unknown", action="store_true",
                     help="resubmit jobs whose previous submit had an uncertain outcome (may be charged twice)")
    # Testing only: allow http / IP-literal download URLs (e.g. a local fake server).
    run.add_argument("--allow-insecure-downloads", action="store_true", help=argparse.SUPPRESS)

    st = sub.add_parser("status", help="summarise the manifest in an output directory")
    st.add_argument("--out", type=Path, default=Path("output"), help="output directory (default: output/)")
    st.add_argument("--json", action="store_true", help="print the raw manifest entries as JSON")
    return p


# --------------------------------------------------------------------------- commands


def _load_jobs(args: argparse.Namespace, console: Console) -> list[Job] | None:
    try:
        raw = load_batch_file(args.batch)
        overrides = {
            "model": args.model,
            "resolution": args.resolution,
            "duration": args.duration,
            "ratio": args.ratio,
            "prompt_expansion": args.prompt_expansion,
        }
        jobs = resolve_jobs(raw, overrides, allow_external_images=args.allow_external_images)
    except BatchFileError as exc:
        console.print(f"[bold red]Batch file error:[/bold red] {escape(str(exc))}")
        return None
    except ValidationError as exc:
        console.print(f"[bold red]Validation failed - no jobs were submitted ({len(exc.errors)} error(s)):[/bold red]")
        for e in exc.errors:
            console.print(f"  - {escape(e)}")
        return None
    for job in jobs:
        for w in job.warnings:
            console.print(f"[yellow]warning[/yellow] {escape(job.name)}: {escape(w)}")
    return jobs


def _plan(jobs: list[Job], manifest: mf.Manifest, args: argparse.Namespace) -> list[tuple[Job, str, str]]:
    opts = RunOptions(rerun_changed=args.rerun_changed, retry_unknown=args.retry_unknown,
                      base_url=args.base_url)
    return [(job, *plan_action(job, manifest.get(job.name), args.out, opts)) for job in jobs]


def _dry_run(jobs: list[Job], args: argparse.Namespace, console: Console) -> int:
    out = console
    try:
        manifest = mf.Manifest.load(args.out)
    except mf.ManifestError as exc:
        console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
        return EXIT_USAGE
    endpoint = args.base_url.rstrip("/") + spec.CREATE_PATH
    plan = _plan(jobs, manifest, args)
    for job, action, reason in plan:
        out.rule(f"[bold]{escape(job.name)}[/bold]  (job #{job.index})")
        style = "yellow" if action == "blocked" else "cyan"
        out.print(f"action: [{style}]{escape(action)}[/{style}] - {escape(reason)}")
        out.print(f"POST {escape(endpoint)}")
        out.print_json(json.dumps(redact_payload(job.payload), ensure_ascii=False))
    out.rule()
    counts = Counter(a for _, a, _ in plan)
    billable = sum(j.params["duration"] for j, a, _ in plan if a == "submit")
    out.print(
        f"[bold]Dry run:[/bold] {len(jobs)} job(s) valid; {counts['submit']} would be submitted "
        f"({billable}s of video would be billed), {counts['resume']} resumed, {counts['skip']} skipped, "
        f"{counts['blocked']} blocked. No network calls were made."
    )
    if counts["blocked"]:
        out.print("[yellow]Some jobs are blocked; a real run would exit with status 1.[/yellow]")
        return EXIT_JOB_FAILED
    return EXIT_OK


def _run(args: argparse.Namespace, console: Console) -> int:
    jobs = _load_jobs(args, console)
    if jobs is None:
        return EXIT_USAGE
    if args.dry_run:
        return _dry_run(jobs, args, console)

    problem = check_base_url(args.base_url, args.allow_custom_host)
    if problem:
        console.print(f"[bold red]Error:[/bold red] {escape(problem)}")
        return EXIT_USAGE

    api_key = load_api_key(warn=lambda m: console.print(f"[yellow]warning[/yellow] {escape(m)}"))
    if not api_key:
        console.print(f"[bold red]Missing API key:[/bold red] set {ENV_KEY} in the environment or in ./.env")
        return EXIT_USAGE

    out_dir: Path = args.out
    opts = RunOptions(
        concurrency=args.concurrency,
        job_timeout=args.timeout * 60.0,
        rerun_changed=args.rerun_changed,
        retry_unknown=args.retry_unknown,
        base_url=args.base_url,
    )

    async def go(manifest: mf.Manifest) -> tuple[list, str]:
        async with MiniMaxClient(api_key, args.base_url,
                                 allow_insecure_downloads=args.allow_insecure_downloads) as client:
            runner = Runner(client, manifest, out_dir, opts, console)
            results = await runner.run(jobs)
            return results, runner.abort_reason

    try:
        with mf.run_lock(out_dir):
            manifest = mf.Manifest.load(out_dir)
            plan = _plan(jobs, manifest, args)
            counts = Counter(a for _, a, _ in plan)
            billable = sum(j.params["duration"] for j, a, _ in plan if a == "submit")
            console.print(
                f"[bold]{len(jobs)} job(s)[/bold]: {counts['submit']} to submit ({billable}s of video billable), "
                f"{counts['resume']} to resume, {counts['skip']} already done, {counts['blocked']} blocked; "
                f"concurrency {args.concurrency}, output -> {escape(str(out_dir))}"
            )
            try:
                results, abort_reason = asyncio.run(go(manifest))
            except KeyboardInterrupt:
                _report_interrupt(manifest, jobs, console)
                return EXIT_INTERRUPTED
    except mf.ManifestError as exc:
        console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
        return EXIT_USAGE

    table = Table(title="Batch summary")
    table.add_column("job")
    table.add_column("outcome")
    table.add_column("task_id")
    table.add_column("detail", overflow="fold")
    for r in results:
        style = "green" if r.ok else "red"
        table.add_row(escape(r.name), f"[{style}]{escape(r.outcome)}[/{style}]", escape(r.task_id or ""),
                      escape(r.file or r.detail))
    console.print(table)
    counts = Counter(r.outcome for r in results)
    console.print(escape(", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))))
    if abort_reason:
        console.print(f"[bold red]Batch aborted:[/bold red] {escape(abort_reason)}")
    return EXIT_OK if all(r.ok for r in results) else EXIT_JOB_FAILED


def _report_interrupt(manifest: mf.Manifest, jobs: list[Job], console: Console) -> None:
    entries = manifest.entries()
    names = {j.name for j in jobs}
    uncertain = sorted(n for n, e in entries.items() if n in names and e.get("status") in mf.MID_SUBMIT)
    console.print("\n[yellow]Interrupted.[/yellow] Submitted task IDs are saved in the manifest; "
                  "re-run the same command to resume them without paying again.")
    if uncertain:
        console.print(
            f"[bold red]{len(uncertain)} job(s) were interrupted mid-submit and may have been charged:[/bold red] "
            + escape(", ".join(uncertain))
        )
        console.print("Check the MiniMax console for these; a re-run will skip them unless you pass "
                      "--retry-unknown.")


def _status(args: argparse.Namespace, console: Console) -> int:
    path = args.out / mf.MANIFEST_NAME
    if not path.exists():
        console.print(f"No manifest at {escape(str(path))}")
        return EXIT_JOB_FAILED
    try:
        manifest = mf.Manifest.load(args.out)
    except mf.ManifestError as exc:
        console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
        return EXIT_USAGE
    entries = manifest.entries()
    if args.json:
        print(json.dumps(entries, indent=2, sort_keys=True))
        return EXIT_OK
    table = Table(title=escape(f"Manifest: {path}"))
    for col in ("job", "status", "task_id", "model", "dur", "file", "error"):
        table.add_column(col, overflow="fold")
    styles = {"downloaded": "green", "failed": "red", "cancelled": "red", "submit_unknown": "red",
              "submitting": "red", "task_lost": "red", "timeout": "yellow", "not_submitted": "yellow"}
    for name, e in sorted(entries.items()):
        status = sanitize(e.get("status", "?"), 40)
        style = styles.get(status, "cyan")
        file = sanitize(e.get("file") or "", 300)
        if file and not (args.out / file).exists():
            file += " (missing)"
        table.add_row(*(escape(sanitize(v, 300)) for v in (name,)),
                      f"[{style}]{escape(status)}[/{style}]",
                      *(escape(sanitize(e.get(k) or "", 200)) for k in ("task_id", "model", "duration")),
                      escape(file), escape(sanitize(e.get("error") or "", 200)))
    console.print(table)
    counts = Counter(sanitize(e.get("status", "?"), 40) for e in entries.values())
    console.print(escape(f"{len(entries)} job(s): " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    if args.command == "run":
        return _run(args, console)
    return _status(args, console)


if __name__ == "__main__":
    sys.exit(main())
