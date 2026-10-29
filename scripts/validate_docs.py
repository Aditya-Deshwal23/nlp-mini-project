"""Validate SVG XML files and relative Markdown links."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    root = Path(".")
    for path in (root / "docs").glob("*.svg"):
        ET.parse(path)
    markdown = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)#]+)")
    missing = []
    for source in markdown:
        for target in link_pattern.findall(source.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            if not (source.parent / target).exists():
                missing.append(f"{source}: {target}")
    if missing:
        raise SystemExit("Broken relative links:\n" + "\n".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
