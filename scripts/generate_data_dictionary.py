"""Generate the human-readable data dictionary from feature_spec.yaml."""

from pathlib import Path

import yaml


def generate(
    spec_path: str | Path = "configs/feature_spec.yaml",
    output: str | Path = "docs/DATA_DICTIONARY.md",
) -> None:
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    lines = [
        "# Data Dictionary",
        "",
        "Generated from `configs/feature_spec.yaml`; the feature table is one row per exchange.",
        "",
        "| Name | Description | Type | Family |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        (
            f"| `{column['name']}` | {column['description']} | "
            f"`{column['type']}` | `{column['family']}` |"
        )
        for column in spec["columns"]
    )
    lines.extend(
        [
            "",
            "## Missing-data policy",
            "",
            f"- Nullable columns: `{', '.join(spec['missing_data_policy']['nullable_columns'])}`.",
            f"- {spec['missing_data_policy']['rule']}",
            "- There are no timestamps in the source turns, so latency and talk time "
            "are not measured.",
            "- Word counts are only a proxy for airtime.",
        ]
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate()
