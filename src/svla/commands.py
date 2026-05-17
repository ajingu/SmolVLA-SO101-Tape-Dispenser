from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_dotenv(path: Path = ENV_PATH) -> dict[str, str]:
    if not path.exists():
        return {}

    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    if "HF_TOKEN" not in values and "HUGGINGFACE_TOKEN" in values:
        values["HF_TOKEN"] = values["HUGGINGFACE_TOKEN"]

    return values


def run(command: list[str]) -> int:
    print("+ " + " ".join(command))
    env = os.environ.copy()
    env.update(_load_dotenv())
    try:
        return subprocess.run(command, env=env).returncode
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
