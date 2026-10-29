"""Configurable transcript loaders."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import TURN_COLUMNS


def _guess_type(raw_role: str, speaker_name: str, text: str) -> str:
    value = f"{raw_role} {speaker_name}".lower()
    if any(word in value for word in ("operator", "moderator", "host")):
        return "operator"
    if any(word in value for word in ("analyst", "research", "capital", "fund")):
        return "analyst"
    if "?" in text:
        return "analyst"
    return "executive"


def _canonical(rows: list[dict[str, Any]], defaults: dict[str, Any]) -> pd.DataFrame:
    output = []
    for index, row in enumerate(rows):
        item = {
            column: row.get(column) if row.get(column) is not None else defaults.get(column, "")
            for column in TURN_COLUMNS
        }
        item["turn_idx"] = int(index if item["turn_idx"] in (None, "") else item["turn_idx"])
        item["section"] = item["section"] or "unknown"
        item["speaker_type"] = _guess_type(
            str(item["speaker_raw_role"]), str(item["speaker_name"]), str(item["text"])
        )
        item["text_raw"] = item["text"]
        item["was_interrupted"] = False
        output.append(item)
    return pd.DataFrame(
        output, columns=TURN_COLUMNS + ["speaker_type", "text_raw", "was_interrupted"]
    )


class TranscriptLoader(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and canonicalise turns."""


class CsvLoader(TranscriptLoader):
    def load(self) -> pd.DataFrame:
        dataset = self.config["dataset"]
        path = Path(dataset.get("path", dataset.get("root", "data/raw")))
        rows = pd.read_csv(path).to_dict("records")
        mapping = dataset.get("columns", {})
        mapped = [
            {canonical: row.get(source) for canonical, source in mapping.items()} for row in rows
        ]
        return _canonical(mapped, {"call_id": path.stem})


class JsonlLoader(TranscriptLoader):
    def load(self) -> pd.DataFrame:
        dataset = self.config["dataset"]
        path = Path(dataset.get("path", dataset.get("root", "data/raw")))
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        mapping = dataset.get("columns", {})
        mapped = [
            {canonical: row.get(source) for canonical, source in mapping.items()} for row in rows
        ]
        return _canonical(mapped, {"call_id": path.stem})


class TextFolderLoader(TranscriptLoader):
    def load(self) -> pd.DataFrame:
        dataset = self.config["dataset"]
        root = Path(dataset.get("root", "data/raw"))
        pattern = dataset.get("file_glob", "*.txt")
        header_pattern = dataset.get(
            "speaker_header_regex", r"^(?P<speaker>[^:]+):\s*(?P<text>.*)$"
        )
        header = re.compile(header_pattern, re.MULTILINE)
        prefix = header_pattern.split("(?P<text>", maxsplit=1)[0]
        lookahead_prefix = (
            prefix.removeprefix("^").replace("(?P<speaker>", "(?:").replace("(?P<role>", "(?:")
        )
        lookahead_prefix = lookahead_prefix.replace(r"[^:\n-]", r"[^:\n.!?-]")
        lookahead_prefix = r"\s+" + lookahead_prefix.removeprefix(r"\s*")
        blob_header = re.compile(
            prefix.removeprefix("^") + r"(?P<text>.*?)" + r"(?=" + lookahead_prefix + r"|$)",
            re.MULTILINE | re.DOTALL,
        )
        rows: list[dict[str, Any]] = []
        for path in sorted(root.glob(pattern)):
            turn_idx = 0
            text_blob = path.read_text(encoding="utf-8")
            matches = list(blob_header.finditer(text_blob))
            if not matches:
                matches = [header.match(line) for line in text_blob.splitlines()]
            for match in matches:
                if match is None:
                    continue
                groups = match.groupdict()
                speaker = groups.get("speaker", "").strip()
                role = (groups.get("role") or "").strip()
                rows.append(
                    {
                        "call_id": path.stem,
                        "company": "",
                        "ticker": "",
                        "call_date": "",
                        "section": "unknown",
                        "turn_idx": turn_idx,
                        "speaker_name": speaker,
                        "speaker_raw_role": role,
                        "speaker_affiliation": "",
                        "text": groups.get("text", "").strip(),
                    }
                )
                turn_idx += 1
        result = _canonical(rows, {})
        return result.sort_values(["call_id", "turn_idx"]).reset_index(drop=True)


def loader_for(config: dict[str, Any]) -> TranscriptLoader:
    format_name = config["dataset"].get("format", "text_folder").lower()
    loaders = {
        "csv": CsvLoader,
        "jsonl": JsonlLoader,
        "text_folder": TextFolderLoader,
        "txt": TextFolderLoader,
    }
    try:
        return loaders[format_name](config)
    except KeyError as error:
        raise ValueError(f"Unsupported dataset format: {format_name}") from error
