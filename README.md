# minimax-batch

**Batch-generate MiniMax Hailuo H3 video clips from a YAML/CSV file via the official MiniMax API: resumable, and never bills a clip twice.**

[![CI](https://github.com/sambawy01/minimax-batch/actions/workflows/ci.yml/badge.svg)](https://github.com/sambawy01/minimax-batch/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **Unofficial project.** `minimax-batch` is a community tool. It is not
> affiliated with, endorsed by, or sponsored by MiniMax. It calls the MiniMax
> API with **your own API key**, and **you pay MiniMax for all usage** it
> generates.

## Features

- **Batch files in YAML or CSV.** List many text-to-video or image-to-video
  jobs, with batch-level defaults, CLI overrides and per-job overrides.
- **Checks everything before spending anything.** Model, resolution,
  duration, ratio, prompt length, image type/size/magic bytes and request
  size are all validated locally. If any job is invalid, the whole batch
  stops with a list of every error, and nothing is charged.
- **`--dry-run`** prints the exact request payloads (image data redacted)
  and the number of billable seconds. It makes no network calls and needs no
  API key.
- **Concurrent** submit, poll and download, with a configurable limit.
  Polling backs off from about 5 s to a 30 s cap.
- **Resumable.** A `manifest.json` records every task ID the moment it
  exists. Re-running the same command resumes polling instead of
  resubmitting.
- **Never double-bills.** Any paid request whose outcome is uncertain
  (timeout after sending, 5xx, crash or Ctrl-C mid-request) is flagged and
  never automatically resubmitted.
- **Careful with your key and your disk.**
  - The key only goes to MiniMax hosts, and it is scrubbed from all errors
    and logs.
  - Downloads must be https to public hosts, are capped at 2 GiB and must
    be MP4s.
  - Local images must stay inside the batch file's directory.
- **Retries** `429`, `5xx` and network errors with exponential backoff and
  jitter. `400`, `401`, `402` and `422` are never retried. A `401` or `402`
  stops new submissions.

## Quick start

Requires Python 3.11 or newer.

```bash
# Install the CLI (from GitHub; also works with pipx)
uv tool install git+https://github.com/sambawy01/minimax-batch
# or: pipx install git+https://github.com/sambawy01/minimax-batch
# once published to PyPI:  uv tool install minimax-batch  /  pipx install minimax-batch

# Provide your MiniMax API key
export MINIMAX_API_KEY=...            # or put it in ./.env (see .env.example)

# ALWAYS preview first: validates, shows payloads and billable seconds, no network
minimax-batch run batch.yaml --dry-run

# Generate for real, then check on it
minimax-batch run batch.yaml --out output/
minimax-batch status --out output/
```

To run from source:

```bash
git clone https://github.com/sambawy01/minimax-batch && cd minimax-batch
uv sync
uv run minimax-batch run examples/batch.yaml --dry-run
```

**API key.** The key is read from the `MINIMAX_API_KEY` environment
variable. If that isn't set, it's read from a `.env` file in the current
directory. You get a warning if `.env` is readable by other users; fix that
with `chmod 600 .env`. `.env` is git-ignored.

## Batch file format

### YAML

```yaml
defaults:                    # optional; same keys as the per-job overrides
  model: MiniMax-H3
  resolution: 768P
  duration: 6
  ratio: "16:9"              # always quote ratios in YAML
  prompt_expansion: balanced

jobs:
  - name: Neon City Flyover  # -> neon-city-flyover.mp4
    prompt: A slow cinematic drone flyover of a rain-soaked neon city at night.

  - name: coffee-pour-vertical
    prompt: Macro shot of espresso pouring into a white cup.
    ratio: "9:16"
    duration: 5

  - name: sunrise-timelapse  # image-to-video: ratio becomes "adaptive" automatically
    prompt: The sky transitions from sunrise to a clear blue morning.
    first_frame: images/sunrise_start.png
    last_frame: images/sunrise_end.png
    model: MiniMax-H3-Max
    resolution: 480P
```

A bare list of jobs, with no `defaults:` or `jobs:` keys, also works.

A few YAML gotchas:

- Unquoted `16:9` is read by YAML as the number 969. The validator catches
  this and tells you to add quotes.
- YAML aliases (`*name`) are rejected.
- A `prompt` that isn't text is rejected.

### CSV

The first row holds the column names. A blank cell means "not set", so the
default applies. Separate multiple `reference_images` with `;`. CSV files
have no `defaults:` section; use CLI flags instead.

```csv
name,prompt,first_frame,last_frame,model,resolution,duration,ratio,prompt_expansion
Mountain Lake Dawn,"Mist over a still mountain lake at dawn.",,,MiniMax-H3,768P,6,16:9,balanced
city-crowd-vertical,"Top-down shot of a busy crosswalk.",,,,,5,9:16,
```

Complete examples are in [`examples/batch.yaml`](examples/batch.yaml) and
[`examples/batch.csv`](examples/batch.csv).

### Field reference

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Output filename, slugified (`"Neon City!"` → `neon-city.mp4`). Must be unique after slugifying; this is also the job's key in the manifest |
| `prompt` | for text-to-video | Up to 7000 characters. Optional when a first or last frame is given |
| `first_frame` | no | Local path (sent as a base64 data URI), `http(s)` URL (sent as-is), or `data:image/...;base64,` URI |
| `last_frame` | no | Same as `first_frame` |
| `reference_images` | no | Up to 9 images (YAML list, or `;`-separated in CSV) |
| `model` | no | `MiniMax-H3` or `MiniMax-H3-Max` |
| `resolution` | no | See the model limits below |
| `duration` | no | Whole seconds; see the model limits below |
| `ratio` | no | `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16` or `adaptive` |
| `prompt_expansion` | no | `disabled`, `balanced` or `quality` (sent as `extra.prompt_expansion_mode`) |

**Image rules:**

- Allowed types are jpg, jpeg, png, webp, heic and heif.
- The type is checked from the file's magic bytes, not just the extension.
- Each image can be at most 30 MB, and the whole request body at most 64 MB.
- Relative paths are resolved against the batch file's directory.
- After following symlinks, local files must stay inside that directory
  unless you pass `--allow-external-images`. This stops a batch file you
  were sent from uploading arbitrary files from your disk.

**Ratio rules:**

- A job with a `first_frame` or `last_frame` is always sent with
  `ratio: adaptive`. If the job itself set a different ratio, you get a
  warning.
- A text-only job must not use `adaptive`.

**Precedence**, from lowest to highest: built-in defaults (`MiniMax-H3`,
`768P`, 6 s, `16:9`, no prompt expansion), then YAML `defaults:`, then CLI
flags, then per-job fields.

### Model limits

| Model | Resolutions | Duration |
|---|---|---|
| `MiniMax-H3` | `768P`, `2K` | 4–15 s |
| `MiniMax-H3-Max` | `480P`, `768P` | 5–15 s |

## Cost and safety

MiniMax **bills per second of generated video**. Run `--dry-run` first: it
reports how many seconds a real run would submit, counting only new
submissions, not jobs being resumed or skipped.

`<out>/manifest.json` makes runs resumable:

- A job is written as `submitting` **before** the paid create request is
  sent.
- The returned task ID is printed and then saved immediately.
- Every write is atomic.
- A lock file stops two runs from using the same output directory. (There
  is no lock on Windows, so don't run two batches into one `--out`
  directory there.)

When you re-run a batch, each job is handled according to its recorded
state:

| Manifest state | What a re-run does |
|---|---|
| `downloaded` and the file exists | Skipped, with no network calls |
| `submitted`, `queued`, `running`, `succeeded` or `timeout` | Polls the **existing** task, then downloads. No new charge |
| `downloaded` but the file was deleted | Gets a fresh URL from the existing task and downloads again |
| `failed`, `cancelled` or `not_submitted` | Submitted again as a new task |
| **`submitting`**, **`submit_unknown`** or **`task_lost`** | **Blocked** unless you pass `--retry-unknown` |
| Parameters changed since the last run | Blocked unless you pass `--rerun-changed`. If the old task may still be running, you also need `--retry-unknown`. The old task ID is kept in `previous_task_ids` |
| A `<name>.mp4` exists that the manifest doesn't know about | Blocked. The file is never overwritten |
| The job's task was created on a different `--base-url` | Resuming (polling or downloading) that task is blocked. Pass the same `--base-url` as before. A *new* submission may go to any host, as long as the usual gates are passed: `--retry-unknown`, plus `--rerun-changed` if the parameters changed. The old task ID and its host are kept in `previous_tasks` (`{task_id, base_url, status}`); plain IDs also stay in `previous_task_ids` |

**`submit_unknown` and `--retry-unknown`.** The create request is the only
call that costs money, so it is only retried when MiniMax *definitely* did
not create a task. That means a `429`, or a failure to connect at all.

In every other case the job is marked `submit_unknown` and left alone,
because a task may exist and may be billed. Those cases are:

- a `5xx` response
- a timeout or dropped connection after the request was sent
- a success reply without a task ID
- Ctrl-C during the request

If the process was killed outright mid-request, the job stays `submitting`,
which is treated the same way.

To recover, check the MiniMax console. If no task was created, re-run with
`--retry-unknown`. That flag **can bill twice** if a task *was* created.

**`task_lost`.** Sometimes the server no longer recognises a task ID we
already paid for. MiniMax reports this as an HTTP 5xx with
`record not found`. For example, a finished task may have been purged after
you deleted its `.mp4`. The job is then marked `task_lost` and treated like
`submit_unknown`: it is **not** resubmitted automatically, and generating it
again needs `--retry-unknown`.

A task that was *just* created can briefly be "not found". For that reason,
a freshly submitted task is retried for about 60 seconds before it is
declared lost.

If a `401` or `402` stops a batch while a `submitting`, `submit_unknown` or
`task_lost` job is being retried, that job keeps its previous state.

Download URLs are time-limited. If a download is refused (401/403/404/410),
the task is re-queried once for a fresh URL. A download that fails anyway
leaves the job `succeeded`, and the next run downloads it without
resubmitting.

## CLI reference

```
minimax-batch run BATCH [options]      validate + run a .yaml/.yml/.csv batch
minimax-batch status [--out DIR] [--json]
minimax-batch --version
```

| `run` option | Default | Meaning |
|---|---|---|
| `--out DIR` | `output/` | Where videos and `manifest.json` go |
| `--concurrency N` | `3` | Jobs in flight at once |
| `--dry-run` | off | Validate and print payloads. No network calls, no key needed |
| `--model` | | Default model (overrides YAML `defaults:`) |
| `--resolution` | | Default resolution |
| `--duration` | | Default duration in seconds |
| `--ratio` | | Default ratio |
| `--prompt-expansion` | | Default prompt expansion mode |
| `--timeout MIN` | `20` | How long to poll each job before giving up (a finite number > 0). The task is kept and a re-run resumes it |
| `--base-url URL` | `https://api.minimax.io` | API base URL. Only MiniMax hosts (`api.minimax.io`, `*.minimax.io`, `api.minimaxi.com`) are accepted. Any other host, **including localhost**, needs `--allow-custom-host`. Plain `http` is allowed only for loopback hosts. It is recorded per job, and resuming needs the same value |
| `--allow-custom-host` | off | Allow a non-MiniMax `--base-url` host, such as a local mock server. **Your API key is sent to that host** |
| `--allow-external-images` | off | Allow local images outside the batch file's directory |
| `--rerun-changed` | off | Regenerate jobs whose parameters changed. **Charged again** |
| `--retry-unknown` | off | Resubmit `submitting` / `submit_unknown` / `task_lost` jobs. **May bill twice** |

**Exit codes**

| Code | Meaning |
|---|---|
| `0` | Every job was downloaded or skipped |
| `1` | At least one job failed, timed out, was blocked or was aborted. `--dry-run` also exits `1` if any job would be blocked |
| `2` | A usage, validation or configuration error, or an unreadable manifest. Nothing was submitted |
| `130` | Interrupted. Any jobs interrupted mid-submit are listed by name |

## Troubleshooting

- **`401 authorized_error`:** the API key is missing or wrong. New
  submissions stop at once, and the affected jobs are recorded as
  `not_submitted` so a re-run submits them. Jobs that were already
  `submitting`, `submit_unknown` or `task_lost` keep that state.
- **`402 insufficient_balance_error`:** your account is out of credit. New
  submissions stop, but jobs that already have a task keep polling and
  downloading, since that costs nothing. Top up and re-run.
- **`422 unprocessable_entity_error`:** the prompt or image was rejected as
  sensitive content. The job is marked `failed` and never retried. Edit it
  and re-run.
- **HTTP 500 `record not found (1000)` while polling:** MiniMax returns this
  for a task ID it doesn't know.
  - A task submitted moments ago is retried for about 60 s.
  - Otherwise the job is marked `task_lost` straight away.
  - It is not resubmitted automatically, because the original may have been
    billed. Use `--retry-unknown` to generate it again.
  - Only a 5xx response containing "record not found" is treated this way.
    For example, a `401` whose message includes "(1000)" is still an auth
    error.
- **`blocked: ... may have been charged`:** see
  [Cost and safety](#cost-and-safety). Check the MiniMax console before
  using `--retry-unknown`.
- **`another minimax-batch run is using ...`:** another run holds the lock
  on that output directory. Wait for it, or use a different `--out`.
- **Coding Plan keys (`sk-cp-...`):** these authenticate against the video
  API, but in testing a create was refused with
  `402 insufficient_balance_error: insufficient balance (1008)`: the Coding
  Plan subscription does not appear to cover video generation. Nothing is
  billed and the job is recorded as `not_submitted`. Top up pay-as-you-go
  balance on the MiniMax platform (or use a standard API key), then re-run
  the same command.

## Development

```bash
uv sync
uv run pytest
```

The test suite runs against an in-memory fake of the MiniMax API
(`httpx.MockTransport`) and a local throwaway HTTP server. It needs **no API
key and no network access**, and never calls MiniMax.

The code lives in `src/minimax_batch/`:

| Module | Role |
|---|---|
| `spec.py` | API constants and limits |
| `batch.py` | YAML/CSV loading, validation, payload building |
| `api.py` | Async HTTP client, retry policy, safe downloads, secret scrubbing |
| `manifest.py` | Atomic state file and run lock |
| `runner.py` | Concurrency, polling, resume and never-double-bill logic |
| `cli.py` | The `minimax-batch` entry point |

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE). Copyright 2026 the minimax-batch contributors.
"MiniMax" and "Hailuo" are trademarks of their respective owners.
