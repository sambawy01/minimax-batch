# Contributing to minimax-batch

Thanks for helping out! This is a small, focused tool; contributions that keep it
small, correct, and safe with people's money are the most welcome.

## Getting started

```bash
git clone https://github.com/sambawy01/minimax-batch && cd minimax-batch
uv sync
uv run pytest
```

The tests use a fake MiniMax server and never need an API key or network access.
**Never** put a real API key in code, tests, fixtures, issues, or logs.

## Guidelines

- **Money safety first.** Any change to submitting, retrying, or resuming must keep the
  invariant that a paid `POST /v2/video_generation` is never repeated automatically
  when its outcome is uncertain. Add a test that proves it.
- Keep dependencies minimal (currently `httpx`, `pyyaml`, `rich`).
- Add or update tests for every behaviour change; `uv run pytest` must pass.
- Keep the README accurate: document new flags and behaviour in the same PR.
- Keep PRs focused; describe *why* as well as *what*.

## Reporting bugs

Open an issue with the command you ran, the output (redact anything secret), your
OS and Python version, and `minimax-batch --version`. For API-behaviour surprises,
include the HTTP status and error type, but never your key.

## Security

If you find a security problem (for example, a way the API key could leak), please
open a GitHub security advisory rather than a public issue.

By contributing you agree that your contributions are licensed under the
Apache License 2.0.
