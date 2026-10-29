"""Check project SVG diagrams for XML, marker, and readable-text contracts."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NS = "{http://www.w3.org/2000/svg}"


def main() -> int:
    errors = []
    for path in sorted(Path("docs").rglob("*.svg")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as error:
            errors.append(f"{path}: invalid XML: {error}")
            continue
        marker_ids = {
            element.attrib["id"]
            for element in root.iter(f"{SVG_NS}marker")
            if "id" in element.attrib
        }
        for element in root.iter():
            marker = element.attrib.get("marker-end", "")
            match = re.search(r"url\(#([^)]+)\)", marker)
            if match and match.group(1) not in marker_ids:
                errors.append(f"{path}: missing marker {match.group(1)}")
        for text in root.iter(f"{SVG_NS}text"):
            size = re.search(r"font-size:\s*([0-9.]+)px", text.attrib.get("style", ""))
            if size and float(size.group(1)) < 12:
                errors.append(f"{path}: text smaller than 12px")
            if "font-size" in text.attrib and float(text.attrib["font-size"]) < 12:
                errors.append(f"{path}: text smaller than 12px")
    if errors:
        raise SystemExit("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
