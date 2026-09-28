from __future__ import annotations

import functools
import json
from pathlib import Path

import httpx
import pytest

from minimax_batch import cli
from minimax_batch.api import MiniMaxClient
from minimax_batch.runner import Runner

from .conftest import TEST_KEY, nosleep

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
FAKE = ["--base-url", "https://api.test", "--allow-custom-host"]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(httpx.AsyncClient, "send", boom)


@pytest.fixture
def fake_api(monkeypatch, server):
    """Route the CLI's client through the fake server with instant sleeps."""
    monkeypatch.setattr(
        cli, "MiniMaxClient",
        functools.partial(MiniMaxClient, transport=httpx.MockTransport(server.handler), sleep=nosleep),
    )
    monkeypatch.setattr(cli, "Runner", functools.partial(Runner, sleep=nosleep))
    monkeypatch.setenv("MINIMAX_API_KEY", TEST_KEY)
    return server


@pytest.mark.parametrize("name", ["batch.yaml", "batch.csv"])
def test_dry_run_examples(name, capsys, no_network, tmp_path):
    code = cli.main(["run", str(EXAMPLES / name), "--dry-run", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert code == 0
    assert "POST https://api.minimax.io/v2/video_generation" in out
    assert "base64 chars" in out  # image data truncated
    assert '"adaptive"' in out
    assert "3 would be submitted" in out
    assert not (tmp_path / "o").exists()  # dry run writes nothing


def test_validation_failure_exits_2_without_network(tmp_path, capsys, no_network):
    bad = tmp_path / "bad.yaml"
    bad.write_text("jobs:\n  - name: a\n    prompt: p\n    ratio: adaptive\n  - name: b\n    prompt: p\n    duration: 99\n")
    code = cli.main(["run", str(bad)])
    out = capsys.readouterr().out
    assert code == 2
    assert "no jobs were submitted" in out and "adaptive" in out and "out of range" in out


def test_missing_api_key(tmp_path, capsys, no_network):
    code = cli.main(["run", str(EXAMPLES / "batch.yaml"), "--out", str(tmp_path / "o")])
    assert code == 2
    assert "MINIMAX_API_KEY" in capsys.readouterr().out


def test_dotenv_loading(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('# comment\nexport MINIMAX_API_KEY="from-dotenv"\n')
    assert cli.load_api_key() == "from-dotenv"
    monkeypatch.setenv("MINIMAX_API_KEY", "from-env")
    assert cli.load_api_key() == "from-env"


def test_full_run_status_and_rerun(tmp_path, capsys, fake_api):
    out = tmp_path / "o"
    code = cli.main(["run", str(EXAMPLES / "batch.yaml"), "--out", str(out), *FAKE])
    text = capsys.readouterr().out
    assert code == 0, text
    assert TEST_KEY not in text
    assert sorted(p.name for p in out.glob("*.mp4")) == [
        "coffee-pour-vertical.mp4", "neon-city-flyover.mp4", "sunrise-timelapse.mp4"]
    assert fake_api.creates == 3

    assert cli.main(["status", "--out", str(out)]) == 0
    status = capsys.readouterr().out
    assert "downloaded: 3" in status and TEST_KEY not in status

    assert cli.main(["status", "--out", str(out), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) == {"coffee-pour-vertical", "neon-city-flyover", "sunrise-timelapse"}
    assert "data:image" not in json.dumps(data)  # no base64 persisted

    # re-run: everything skipped, nothing resubmitted
    assert cli.main(["run", str(EXAMPLES / "batch.yaml"), "--out", str(out), *FAKE]) == 0
    assert fake_api.creates == 3


def test_run_exit_code_1_on_job_failure(tmp_path, capsys, fake_api):
    fake_api.next_statuses = [["failed"]]
    batch = tmp_path / "b.csv"
    batch.write_text("name,prompt\none,hello\n")
    code = cli.main(["run", str(batch), "--out", str(tmp_path / "o"), *FAKE])
    assert code == 1
    assert "failed" in capsys.readouterr().out


def test_status_without_manifest(tmp_path, capsys):
    assert cli.main(["status", "--out", str(tmp_path / "none")]) == 1


def test_invalid_base_url(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", TEST_KEY)
    code = cli.main(["run", str(EXAMPLES / "batch.yaml"), "--base-url", "ftp://x"])
    assert code == 2
