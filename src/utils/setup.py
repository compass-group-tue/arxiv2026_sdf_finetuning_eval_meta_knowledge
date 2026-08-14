import json
import logging
import os
from pathlib import Path
from typing import Any

import dotenv
import jsonlines

LOGGER = logging.getLogger(__name__)

LOGGING_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def setup_logging(logging_level: str):
    level = LOGGING_LEVELS.get(logging_level.lower(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("git").setLevel(logging.CRITICAL)
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)
    logging.getLogger("anthropic._base_client").setLevel(logging.CRITICAL)
    logging.getLogger("fork_posix").setLevel(logging.CRITICAL)


def setup_environment(logging_level: str = "info", override_env: bool = True):
    setup_logging(logging_level)
    load_success = dotenv.load_dotenv(override=override_env)
    if not load_success:
        LOGGER.warning("No .env file found; set environment variables manually if needed.")
    if "HF_TOKEN" not in os.environ:
        LOGGER.warning("HF_TOKEN not found in environment.")


def load_jsonl(file_path: Path | str, **kwargs) -> list:
    try:
        with jsonlines.open(file_path, "r") as f:
            return [o for o in f.iter(**kwargs)]
    except jsonlines.jsonlines.InvalidLineError:
        data = []
        with open(file_path, "r") as file:
            for line in file:
                data.append(json.loads(line.strip()))
        return data


def _convert_paths_to_strings(dict_list: list) -> list:
    result = []
    for d in dict_list:
        if isinstance(d, dict):
            result.append({k: str(v) if isinstance(v, Path) else v for k, v in d.items()})
        else:
            result.append(d)
    return result


def save_jsonl(file_path: Path | str, data: list, mode: str = "w"):
    data = _convert_paths_to_strings(data)
    if isinstance(file_path, str):
        file_path = Path(file_path)
    assert file_path.suffix == ".jsonl", "file_path must end with .jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(file_path, mode=mode) as f:
        assert isinstance(f, jsonlines.Writer)
        f.write_all(data)
