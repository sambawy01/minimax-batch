from __future__ import annotations

import base64
from pathlib import Path

import pytest

from minimax_batch.batch import (
    BatchFileError,
    ValidationError,
    load_batch_file,
    redact_payload,
    resolve_jobs,
    slugify,
)

from .conftest import PNG_BYTES, make_jobs

ROOT = Path(__file__).resolve().parents[1]


def errors_of(jobs, tmp_path, **kw) -> list[str]:
    with pytest.raises(ValidationError) as ei:
        make_jobs(jobs, tmp_path, **kw)
    return ei.value.errors


# ---------------------------------------------------------------- payload building


def test_text_job_payload(tmp_path):
    [job] = make_jobs([{"name": "My Clip!", "prompt": "a cat", "prompt_expansion": "Quality"}], tmp_path)
    assert job.name == "my-clip"
    assert job.filename == "my-clip.mp4"
    assert job.payload == {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "a cat"}],
        "resolution": "768P",
        "duration": 6,
        "ratio": "16:9",
        "extra": {"prompt_expansion_mode": "quality"},
    }


def test_no_extra_when_prompt_expansion_unset(tmp_path):
    [job] = make_jobs([{"name": "a", "prompt": "x"}], tmp_path)
    assert "extra" not in job.payload


def test_image_job_local_file_becomes_data_uri_and_adaptive(tmp_path, png_file):
    [job] = make_jobs(
        [{"name": "img", "prompt": "move", "first_frame": "frame.png", "last_frame": "https://x.test/l.jpg"}],
        tmp_path,
        defaults={"ratio": "9:16"},
    )
    assert job.payload["ratio"] == "adaptive"
    assert job.warnings == ()  # ratio came from defaults, not the job: silent
    text, first, last = job.payload["content"]
    assert text == {"type": "text", "text": "move"}
    assert first["role"] == "first_frame"
    expected = "data:image/png;base64," + base64.b64encode(png_file.read_bytes()).decode()
    assert first["image_url"]["url"] == expected
    assert last == {"type": "image_url", "image_url": {"url": "https://x.test/l.jpg"}, "role": "last_frame"}


def test_image_job_explicit_ratio_overridden_with_warning(tmp_path, png_file):
    [job] = make_jobs([{"name": "img", "prompt": "p", "first_frame": str(png_file), "ratio": "16:9"}], tmp_path)
    assert job.payload["ratio"] == "adaptive"
    assert any("adaptive" in w for w in job.warnings)


def test_image_only_job_prompt_optional(tmp_path, png_file):
    [job] = make_jobs([{"name": "img", "first_frame": "frame.png"}], tmp_path)
    assert [c["type"] for c in job.payload["content"]] == ["image_url"]


def test_reference_images(tmp_path, png_file):
    [job] = make_jobs([{"name": "r", "prompt": "p", "reference_images": ["frame.png", "https://x.test/a.webp"]}],
                      tmp_path)
    roles = [c.get("role") for c in job.payload["content"]]
    assert roles == [None, "reference_image", "reference_image"]
    assert job.payload["ratio"] == "16:9"


def test_precedence_builtin_defaults_cli_job(tmp_path):
    jobs = make_jobs(
        [{"name": "a", "prompt": "p"}, {"name": "b", "prompt": "p", "duration": 9}],
        tmp_path,
        defaults={"duration": 7, "resolution": "2K"},
        cli={"duration": 8, "model": None},
    )
    assert jobs[0].payload["duration"] == 8  # CLI beats YAML defaults
    assert jobs[0].payload["resolution"] == "2K"  # YAML default beats builtin
    assert jobs[1].payload["duration"] == 9  # job beats CLI


def test_params_hash_changes_with_payload(tmp_path):
    a, b = make_jobs([{"name": "a", "prompt": "p"}, {"name": "b", "prompt": "q"}], tmp_path)
    [a2] = make_jobs([{"name": "a", "prompt": "p"}], tmp_path)
    assert a.params_hash == a2.params_hash != b.params_hash


def test_redact_payload_truncates_base64(tmp_path, png_file):
    [job] = make_jobs([{"name": "img", "prompt": "p", "first_frame": "frame.png"}], tmp_path)
    red = redact_payload(job.payload)
    url = red["content"][1]["image_url"]["url"]
    full = job.payload["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,") and "base64 chars" in url
    assert len(url) < len(full)
    assert job.payload["content"][1]["image_url"]["url"] == full  # original untouched


# ---------------------------------------------------------------- validation rules


def test_text_only_adaptive_rejected(tmp_path):
    errs = errors_of([{"name": "a", "prompt": "p", "ratio": "adaptive"}], tmp_path)
    assert any("adaptive" in e for e in errs)


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"model": "MiniMax-H2"}, "model"),
        ({"model": "MiniMax-H3", "resolution": "480P"}, "resolution"),
        ({"model": "MiniMax-H3-Max", "resolution": "2K"}, "resolution"),
        ({"model": "MiniMax-H3", "duration": 3}, "out of range"),
        ({"model": "MiniMax-H3", "duration": 16}, "out of range"),
        ({"model": "MiniMax-H3-Max", "resolution": "768P", "duration": 4}, "out of range"),
        ({"duration": 6.5}, "integer"),
        ({"duration": True}, "integer"),
        ({"ratio": "2:1"}, "ratio"),
        ({"prompt_expansion": "max"}, "prompt_expansion"),
        ({"prompt": "x" * 7001}, "max 7000"),
        ({"prompt": ""}, "'prompt' is required"),
        ({"colour": "red"}, "unknown field"),
        ({"first_frame": "missing.png"}, "not found"),
        ({"first_frame": "notes.txt"}, "unsupported image type"),
        ({"first_frame": "data:image/gif;base64,AAAA"}, "data URI"),
        ({"reference_images": ["https://x.test/a.png"] * 10}, "at most 9"),
    ],
)
def test_invalid_jobs(tmp_path, overrides, fragment):
    job = {"name": "a", "prompt": "p", **overrides}
    errs = errors_of([job], tmp_path)
    assert any(fragment in e for e in errs), errs


def test_boundary_durations_ok(tmp_path):
    make_jobs([{"name": "a", "prompt": "p", "duration": 4}, {"name": "b", "prompt": "p", "duration": 15}], tmp_path)
    make_jobs([{"name": "c", "prompt": "p", "model": "MiniMax-H3-Max", "resolution": "480p", "duration": "5"}],
              tmp_path)


def test_oversized_image_rejected(tmp_path):
    big = tmp_path / "big.jpg"
    with big.open("wb") as fh:
        fh.truncate(30 * 1024 * 1024 + 1)
    errs = errors_of([{"name": "a", "prompt": "p", "first_frame": "big.jpg"}], tmp_path)
    assert any("30MB" in e for e in errs)


def test_all_errors_reported_together(tmp_path):
    errs = errors_of(
        [
            {"name": "ok", "prompt": "fine"},
            {"name": "bad1", "prompt": "p", "duration": 99},
            {"prompt": "no name"},
            {"name": "bad3", "prompt": "p", "ratio": "adaptive"},
        ],
        tmp_path,
    )
    assert len(errs) == 3
    assert any("#2" in e for e in errs) and any("#3" in e for e in errs) and any("#4" in e for e in errs)


def test_duplicate_names_after_slugify(tmp_path):
    errs = errors_of([{"name": "My Clip", "prompt": "p"}, {"name": "my-clip", "prompt": "q"}], tmp_path)
    assert any("collides" in e for e in errs)


def test_unknown_defaults_key(tmp_path):
    errs = errors_of([{"name": "a", "prompt": "p"}], tmp_path, defaults={"modle": "x"})
    assert any("defaults" in e for e in errs)


def test_slugify():
    assert slugify("  Héllo, World!! ") == "hello-world"
    assert slugify("///") == ""


# ---------------------------------------------------------------- file loading


def test_example_files_validate():
    for name in ("batch.yaml", "batch.csv"):
        raw = load_batch_file(ROOT / "examples" / name)
        jobs = resolve_jobs(raw)
        assert len(jobs) == 3
        assert any(j.payload["ratio"] == "adaptive" for j in jobs)


def test_csv_blank_cells_fall_back_to_defaults(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("name,prompt,duration,ratio,reference_images\nA,hello,,,\nB,world,9,1:1,x.png;https://h.test/y.jpg\n")
    (tmp_path / "x.png").write_bytes(PNG_BYTES)
    raw = load_batch_file(p)
    assert raw.jobs[0] == {"name": "A", "prompt": "hello"}
    a, b = resolve_jobs(raw, {"duration": 5})
    assert a.payload["duration"] == 5 and a.payload["ratio"] == "16:9"
    assert b.payload["duration"] == 9 and b.payload["ratio"] == "1:1"
    assert len(b.payload["content"]) == 3


def test_yaml_list_form_and_bad_structure(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text("- name: a\n  prompt: p\n")
    assert len(resolve_jobs(load_batch_file(p))) == 1
    p.write_text("jobs: []\n")
    with pytest.raises(BatchFileError):
        load_batch_file(p)
    p.write_text("jobz: []\n")
    with pytest.raises(BatchFileError):
        load_batch_file(p)
    with pytest.raises(BatchFileError):
        load_batch_file(tmp_path / "nope.yaml")
    txt = tmp_path / "b.txt"
    txt.write_text("x")
    with pytest.raises(BatchFileError):
        load_batch_file(txt)
