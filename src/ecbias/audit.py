"""Trust audit for generated outputs and documentation claims."""

from __future__ import annotations

import ast
import importlib.metadata
import re
import tomllib
from pathlib import Path

import pandas as pd


def _todo_lines(root: Path) -> list[str]:
    directories = ("src", "docs", "scripts", "configs")
    paths = [
        path
        for directory in directories
        for path in (root / directory).rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".yaml", ".yml"}
    ]
    paths.extend([root / "README.md", root / "CHANGELOG.md"])
    return [
        f"{path}:{line_no}: {line.strip()}"
        for path in paths
        if path.exists()
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        )
        if "TODO" in line
    ]


def _speaker_names(root: Path) -> set[str]:
    path = root / "data/interim/turns.parquet"
    if not path.exists():
        return set()
    return set(pd.read_parquet(path)["speaker_name"].dropna().astype(str))


def _output_files(root: Path) -> list[Path]:
    return [
        path
        for directory in ("results", "models")
        if (root / directory).exists()
        for path in (root / directory).rglob("*")
        if path.is_file()
    ]


def _public_functions(root: Path) -> list[tuple[Path, str, bool]]:
    functions = []
    for path in (root / "src/ecbias").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_function and not node.name.startswith("_"):
                functions.append((path, node.name, ast.get_docstring(node) is not None))
    return functions


def _unused_dependencies(root: Path) -> list[str]:
    """List declared runtime dependencies with no import in project Python files."""
    manifest = root / "pyproject.toml"
    if not manifest.exists():
        return []
    dependencies = [
        re.split(r"[<>=!~]", item, maxsplit=1)[0].strip()
        for item in tomllib.loads(manifest.read_text(encoding="utf-8"))
        .get("project", {})
        .get("dependencies", [])
    ]
    import_roots = set()
    for directory in ("src", "app"):
        for path in (root / directory).rglob("*.py") if (root / directory).exists() else []:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    import_roots.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    import_roots.add(node.module.split(".")[0])
    aliases = {
        "gender-guesser": "gender_guesser",
        "scikit-learn": "sklearn",
        "vaderSentiment": "vaderSentiment",
        "PyYAML": "yaml",
    }
    imported = {name.lower() for name in import_roots}
    return [
        dependency
        for dependency in dependencies
        if aliases.get(dependency, dependency).lower() not in imported
    ]


def run_audit(root: str | Path = ".") -> dict[str, object]:
    """Run non-destructive trust checks and return structured findings."""
    root = Path(root)
    failures: list[str] = []
    quality_path = root / "results/tables/data_quality.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path).iloc[0].to_dict()
        claims = {
            "calls": int(quality["calls_loaded"]),
            "qa calls": int(quality["calls_with_qa_found"]),
            "exchanges": int(quality["exchanges"]),
        }
        documents = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
        text = "\n".join(path.read_text(encoding="utf-8") for path in documents if path.exists())
        for label, number in claims.items():
            patterns = [rf"\b{number}\s+{re.escape(label)}\b"]
            if label == "calls":
                patterns.append(rf"\b{number}\s+calls?\b")
            if label == "qa calls":
                patterns.append(rf"\b{number}\s+calls?\s+with\b")
            if label == "exchanges":
                patterns.append(rf"\b{number}\s+exchanges?\b")
            if not any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                failures.append(f"quoted {label} count {number} not found in documentation")
    names = _speaker_names(root)
    leaked = [
        path for path in _output_files(root)
        if any(name and name in path.read_text(encoding="utf-8", errors="ignore") for name in names)
    ]
    if leaked:
        failures.append(
            "speaker names found in output files: " + ", ".join(str(path) for path in leaked)
        )
    public_functions = _public_functions(root)
    missing_docs = [
        f"{path}:{name}" for path, name, documented in public_functions if not documented
    ]
    test_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "tests").rglob("*.py")
        if (root / "tests").exists()
    )
    untested = [f"{path}:{name}" for path, name, _ in public_functions if name not in test_text]
    return {
        "failures": failures,
        "todos": _todo_lines(root),
        "missing_docstrings": missing_docs,
        "untested_public_functions": untested,
        "unused_dependencies": _unused_dependencies(root),
    }


def print_audit(root: str | Path = ".") -> int:
    """Print the audit report and return a process status."""
    result = run_audit(root)
    print("ECBias audit")
    print("\nTODOs:")
    for todo in result["todos"]:
        print(f"- {todo}")
    print("\nInformational checks:")
    print(
        "- Public functions without docstrings: "
        f"{len(result['missing_docstrings'])}"
    )
    print(
        "- Public functions without a test reference: "
        f"{len(result['untested_public_functions'])}"
    )
    unused = result["unused_dependencies"]
    print(f"- Declared dependencies without detected imports: {', '.join(unused) or 'none'}")
    try:
        importlib.metadata.version("ecbias")
        print("- Installed package: PASS")
    except importlib.metadata.PackageNotFoundError:
        print("- Installed package: TODO (run pip install -e .)")
    failures = result["failures"]
    if failures:
        print("\nFAIL:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nPASS: no hard audit failures")
    return 0
