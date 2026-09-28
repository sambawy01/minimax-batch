"""Constants describing the MiniMax V2 video generation API contract.

Kept in one place so validation, payload building and docs stay in sync.
"""

from __future__ import annotations

DEFAULT_BASE_URL = "https://api.minimax.io"
CREATE_PATH = "/v2/video_generation"
QUERY_PATH = "/v2/query/video_generation/{task_id}"

# model -> allowed resolutions and inclusive duration range (seconds)
MODELS: dict[str, dict] = {
    "MiniMax-H3": {"resolutions": ("768P", "2K"), "duration": (4, 15)},
    "MiniMax-H3-Max": {"resolutions": ("480P", "768P"), "duration": (5, 15)},
}

RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "adaptive")
PROMPT_EXPANSION_MODES = ("disabled", "balanced", "quality")

MAX_PROMPT_CHARS = 7000
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_REFERENCE_IMAGES = 9

# file extension -> MIME subtype used in the data URI
IMAGE_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif",
}

# Task statuses reported by the query endpoint
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TERMINAL_TASK_STATUSES = (STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED)

# Built-in defaults (lowest precedence)
BUILTIN_DEFAULTS = {
    "model": "MiniMax-H3",
    "resolution": "768P",
    "duration": 6,
    "ratio": "16:9",
    "prompt_expansion": None,
}

# Download safety limits
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
DOWNLOAD_ATTEMPT_TIMEOUT = 15 * 60.0  # seconds, per download attempt

# Hosts the API key may be sent to without --allow-custom-host
TRUSTED_API_HOSTS = ("api.minimax.io", "api.minimaxi.com")
TRUSTED_API_HOST_SUFFIXES = (".minimax.io",)
