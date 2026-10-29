"""ECBias command-line skeleton."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .argmining.annotator import TerminalAnnotator, agreement_files, annotator_path
from .argmining.annotator import agreement as compute_agreement
from .argmining.classifier import hand_features, hybrid_label, train_and_evaluate
from .argmining.rules import default_lexicons, label_sentence
from .argmining.sentences import sentence_table
from .audit import print_audit
from .config import load_config
from .features.pipeline import build_feature_table
from .features.text_features import build_text_features
from .io.loaders import loader_for
from .io.schema import validate_turns
from .parsing.clean import clean_turns
from .parsing.exchanges import build_exchanges
from .parsing.segmenter import segment_qa
from .people.enrich import enrich_exchanges, write_gender_coverage
from .people.gender import infer_gender, load_overrides, speaker_id
from .people.roles import classify_turns, load_exec_titles
from .scoring.aqs import (
    alternative_scorers,
    score_aqs,
    scorer_agreement,
)
from .viz.figures import generate_all

app = typer.Typer(help="ECBias earnings-call analysis workflow.")
_config_path = Path("configs/default.yaml")


def _not_implemented() -> None:
    typer.echo("not implemented yet")


@app.callback()
def main(
    config: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Configuration YAML path."
    ),
) -> None:
    """Run an ECBias workflow command."""
    global _config_path
    _config_path = config


@app.command()
def ingest(
    config: Path = typer.Option(Path("configs/dataset.yaml"), "--config"),
    sample: int | None = typer.Option(None, "--sample", min=1),
) -> None:
    """Load transcripts, segment Q&A, and write turns/exchanges tables."""
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    turns = loader_for(settings).load()
    if sample:
        calls = turns["call_id"].drop_duplicates().head(sample)
        turns = turns[turns["call_id"].isin(calls)].copy()
    problems = validate_turns(turns)
    if problems:
        raise typer.BadParameter("; ".join(problems))
    parsing = settings["parsing"]
    turns = clean_turns(turns, parsing.get("interrupted_marker", "[Interrupted]"))
    turns, qa_status = segment_qa(turns, parsing["qa_markers"])
    qa_turns = turns[turns["qa_found"]].copy()
    exchanges = build_exchanges(qa_turns)
    interim = Path(settings["paths"]["interim"])
    results = Path(settings["paths"]["results"]) / "tables"
    interim.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    turns.to_parquet(interim / "turns.parquet", index=False)
    exchanges.to_parquet(interim / "exchanges.parquet", index=False)
    question_words = exchanges["question_text"].map(lambda text: len(str(text).split()))
    answer_words = exchanges["answer_text"].map(lambda text: len(str(text).split()))
    quality = {
        "calls_loaded": len(qa_status),
        "calls_with_qa_found": sum(qa_status.values()),
        "exchanges": len(exchanges),
        "median_question_words": question_words.median() if len(question_words) else 0,
        "median_answer_words": answer_words.median() if len(answer_words) else 0,
        "share_with_empty_answers": (
            float(exchanges["answer_text"].eq("").mean()) if len(exchanges) else 0
        ),
        "share_interrupted": (float(exchanges["was_interrupted"].mean()) if len(exchanges) else 0),
    }
    pd.DataFrame([quality]).to_csv(results / "data_quality.csv", index=False)
    typer.echo(f"Wrote {len(turns)} turns and {len(exchanges)} exchanges")


@app.command()
def enrich(
    config: Path = typer.Option(Path("configs/dataset.yaml"), "--config"),
    include_probable: bool | None = typer.Option(None, "--include-probable/--exclude-probable"),
) -> None:
    """Classify roles, infer perceived gender, and write privacy-safe enrichment."""
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    paths = settings["paths"]
    turns_path = Path(paths["interim"]) / "turns.parquet"
    exchanges_path = Path(paths["interim"]) / "exchanges.parquet"
    if not turns_path.exists() or not exchanges_path.exists():
        typer.echo("interim tables missing; run ecbias ingest first")
        raise typer.Exit(code=1)
    turns = pd.read_parquet(turns_path)
    exchanges = pd.read_parquet(exchanges_path)
    titles = load_exec_titles(settings["roles"]["executive_titles_path"])
    turns = classify_turns(turns, titles)
    # Rebuild exchanges so role decisions from this phase drive responder metadata.
    qa_turns = turns[turns["qa_found"]].copy()
    exchanges = build_exchanges(qa_turns)
    gender_config = settings["gender"]
    enriched = enrich_exchanges(
        exchanges,
        turns,
        gender_config["salt"],
        gender_config["overrides_path"],
    )
    output = Path(paths["interim"]) / "exchanges_enriched.parquet"
    enriched.to_parquet(output, index=False)
    results = Path(paths["results"]) / "tables"
    results.mkdir(parents=True, exist_ok=True)
    probable = (
        gender_config.get("include_probable", True)
        if include_probable is None
        else include_probable
    )
    eligible = enriched[enriched["analyst_gender"].ne("Unknown")]
    if not probable:
        eligible = eligible[eligible["analyst_gender_conf"].ne("probable")]
    write_gender_coverage(eligible, results / "gender_coverage.csv")
    typer.echo(f"Wrote {len(enriched)} enriched exchanges; analysis rows: {len(eligible)}")


@app.command()
def score() -> None:
    """Compute transparent question and answer quality scores."""
    settings = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
    exchanges = pd.read_parquet(Path(settings["paths"]["interim"]) / "exchanges_enriched.parquet")
    sentences_path = Path(settings["argmining"]["sentence_cache"])
    sentences = pd.read_parquet(sentences_path)
    if "label" not in sentences and "label_rule" in sentences:
        sentences["label"] = sentences["label_rule"]
    feature_path = Path(settings["paths"]["interim"]) / "text_features.parquet"
    features = build_text_features(sentences, exchanges, feature_path)
    scores = score_aqs(features)
    quality = scores.merge(exchanges[["exchange_id", "analyst_id"]], on="exchange_id", how="left")
    Path(settings["paths"]["processed"]).mkdir(parents=True, exist_ok=True)
    quality.to_parquet(Path(settings["paths"]["processed"]) / "quality.parquet", index=False)
    results = Path(settings["paths"]["results"]) / "tables"
    results.mkdir(parents=True, exist_ok=True)
    question_components = features[features["side"].eq("question")].set_index("exchange_id")
    scorers = alternative_scorers(question_components, "question")
    scorers["AQS_q"] = quality.set_index("exchange_id")["AQS_q"].reindex(question_components.index)
    scorer_agreement(scorers).to_csv(results / "aqs_scorer_agreement.csv", index=False)
    example_columns = [
        "exchange_id",
        "AQS_q",
        "structure_q",
        "specificity_q",
        "clarity_q",
        "calibration_q",
        "focus",
    ]
    examples = (
        exchanges[["exchange_id", "question_text"]]
        .merge(quality[example_columns], on="exchange_id")
        .sort_values("AQS_q")
    )
    examples["question_text"] = examples["question_text"].map(lambda text: str(text)[:500])
    example_rows = pd.concat(
        [examples.head(5).assign(rank="lowest"), examples.tail(5).assign(rank="highest")]
    ).drop(columns="exchange_id")
    example_rows.to_csv(results / "aqs_examples.csv", index=False)
    typer.echo(f"Wrote {len(quality)} rows to data/processed/quality.parquet")


@app.command()
def features() -> None:
    """Build the exchange-level response and interaction feature table."""
    settings = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
    interim = Path(settings["paths"]["interim"])
    processed = Path(settings["paths"]["processed"])
    exchanges_path = interim / "exchanges_enriched.parquet"
    quality_path = processed / "quality.parquet"
    if not exchanges_path.exists():
        raise typer.BadParameter("Missing exchanges_enriched.parquet; run ecbias enrich first")
    if not quality_path.exists():
        raise typer.BadParameter("Missing quality.parquet; run ecbias score first")
    exchanges = pd.read_parquet(exchanges_path)
    quality = pd.read_parquet(quality_path)
    output = processed / "features.parquet"
    feature_table = build_feature_table(exchanges, quality, output=output)
    summary = pd.DataFrame(
        {
            "metric": ["rows", "columns", "missing_cells"],
            "value": [
                len(feature_table),
                len(feature_table.columns),
                int(feature_table.isna().sum().sum()),
            ],
        }
    )
    typer.echo(summary.to_string(index=False))
    typer.echo(f"Wrote {output}")


@app.command()
def analyze() -> None:
    """Write a deterministic, descriptive analysis bundle."""
    features_path = Path("data/processed/features.parquet")
    results = Path("results")
    tables = results / "tables"
    if not features_path.exists():
        raise typer.BadParameter("Missing features.parquet; run ecbias features first")
    features = pd.read_parquet(features_path)
    rows = []
    for outcome in (
        "AQS_q",
        "n_requests",
        "relevance_tfidf",
        "answer_to_question_ratio",
        "hedge_rate",
        "deflection_rate",
        "courtesy_opener",
        "AQS_a",
    ):
        if outcome not in features:
            continue
        values = pd.to_numeric(features[outcome], errors="coerce")
        for gender in ("F", "M"):
            group = values[
                features.get("analyst_gender", pd.Series(index=features.index)).eq(gender)
            ]
            rows.append(
                {
                    "outcome": outcome,
                    "group": gender,
                    "n": int(group.notna().sum()),
                    "mean": group.mean(),
                }
            )
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "descriptives.csv", index=False)
    pd.DataFrame(columns=["outcome", "effect_size", "ci_low", "ci_high", "p_raw", "p_adj"]).to_csv(
        tables / "tests_primary.csv", index=False
    )
    pd.DataFrame(columns=["outcome", "term", "estimate", "p_value"]).to_csv(
        tables / "tests_exploratory.csv", index=False
    )
    pd.DataFrame(columns=["outcome", "term", "estimate", "std_error", "p_value"]).to_csv(
        tables / "models_ols.csv", index=False
    )
    pd.DataFrame(columns=["outcome", "term", "estimate", "std_error", "p_value"]).to_csv(
        tables / "models_mixed.csv", index=False
    )
    pd.DataFrame(columns=["outcome", "cell", "n", "estimate", "p_value"]).to_csv(
        tables / "interaction.csv", index=False
    )
    pd.DataFrame(columns=["outcome", "eligible_calls", "mean_difference", "p_value"]).to_csv(
        tables / "paired_within_call.csv", index=False
    )
    pd.DataFrame(columns=["outcome", "n_female", "n_male", "minimum_detectable_d"]).to_csv(
        tables / "power.csv", index=False
    )
    pd.DataFrame(columns=["outcome", "specification", "conclusion_holds"]).to_csv(
        tables / "robustness_matrix.csv", index=False
    )
    pd.DataFrame(
        np.zeros((5, 5), dtype=int),
        index=["CLAIM", "EVIDENCE", "REBUTTAL", "REQUEST", "OTHER"],
        columns=["CLAIM", "EVIDENCE", "REBUTTAL", "REQUEST", "OTHER"],
    ).to_csv(tables / "confusion_matrix.csv")
    (results / "SUMMARY.md").write_text(
        "# Analysis summary\n\nNo statistically tested primary results were available in this run. "
        "These outputs are descriptive scaffolding only; findings from observational data are associations, never proof of bias.\n",
        encoding="utf-8",
    )
    typer.echo(f"Wrote analysis tables to {tables}")


@app.command()
def figures() -> None:
    """Generate the eight offline publication figures."""
    exchanges = pd.read_parquet("data/interim/exchanges_enriched.parquet")
    sentences = pd.read_parquet("data/processed/sentences_labeled.parquet")
    features_table = pd.read_parquet("data/processed/features.parquet")
    tests_path = Path("results/tables/tests_primary.csv")
    robustness_path = Path("results/tables/robustness_matrix.csv")
    confusion_path = Path("results/tables/confusion_matrix.csv")
    tests = pd.read_csv(tests_path) if tests_path.exists() else pd.DataFrame()
    robustness = pd.read_csv(robustness_path) if robustness_path.exists() else pd.DataFrame()
    confusion = pd.read_csv(confusion_path, index_col=0) if confusion_path.exists() else None
    generate_all(exchanges, sentences, features_table, tests, robustness, confusion)
    typer.echo("Wrote eight figures to results/figures/")


@app.command("run-all")
def run_all(
    sample: int | None = typer.Option(None, "--sample", min=1),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config"),
) -> None:
    """Run the reproducible pipeline and write a manifest."""
    started = time.perf_counter()
    stages = [
        ("ingest", lambda: ingest(config=Path("configs/dataset.yaml"), sample=sample)),
        ("enrich", lambda: enrich(config=Path("configs/dataset.yaml"))),
        ("mine", mine),
        ("score", score),
        ("features", features),
        ("analyze", analyze),
        ("figures", figures),
    ]
    timings = []
    try:
        for name, action in stages:
            stage_start = time.perf_counter()
            action()
            timings.append({"stage": name, "seconds": round(time.perf_counter() - stage_start, 3)})
    except Exception as error:
        typer.echo(f"run-all failed at stage '{name}': {error}", err=True)
        raise typer.BadParameter(f"run-all failed at '{name}': {error}") from error
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    results = Path(settings.get("paths", {}).get("results", "results"))
    manifest = {
        "python": platform.python_version(),
        "libraries": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "spacy", "statsmodels")
        },
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "seed": settings.get("seed", 42),
        "input_file_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path("tests/fixtures").glob("*")
            if path.is_file()
        },
        "row_counts": {
            name: len(pd.read_parquet(path))
            for name, path in {
                "turns": Path("data/interim/turns.parquet"),
                "exchanges": Path("data/interim/exchanges_enriched.parquet"),
                "sentences": Path("data/processed/sentences_labeled.parquet"),
                "quality": Path("data/processed/quality.parquet"),
                "features": Path("data/processed/features.parquet"),
            }.items()
            if path.exists()
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or None,
        "timings": timings,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(pd.DataFrame(timings).to_string(index=False))


@app.command()
def doctor() -> None:
    """Check the local environment and input data."""
    config = load_config(_config_path)
    checks: list[tuple[str, bool, str]] = []
    python_ok = (3, 11) <= __import__("sys").version_info[:2] < (3, 12)
    checks.append(("Python 3.11", python_ok, __import__("sys").version.split()[0]))
    try:
        import spacy

        spacy.load("en_core_web_sm")
        model_ok, model_detail = True, "en_core_web_sm installed"
    except (ImportError, OSError) as error:
        model_ok, model_detail = False, str(error)
    checks.append(("spaCy model", model_ok, model_detail))
    raw_path = Path(config.paths.raw)
    files = (
        [item for item in raw_path.iterdir() if item.is_file() and item.name != ".gitkeep"]
        if raw_path.exists()
        else []
    )
    checks.append(("data/raw has files", bool(files), f"{len(files)} file(s)"))
    typer.echo("ECBias doctor")
    for name, passed, detail in checks:
        typer.echo(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


@app.command()
def audit() -> None:
    """Audit documentation claims, privacy outputs, TODOs, and package readiness."""
    raise typer.Exit(print_audit())


@app.command("gender-audit")
def gender_audit(
    config: Path = typer.Option(Path("configs/dataset.yaml"), "--config"),
    evaluate: bool = typer.Option(False, "--evaluate"),
) -> None:
    """Create or evaluate a manually verified perceived-gender audit sample."""
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    annotations = Path("data/annotations")
    annotations.mkdir(parents=True, exist_ok=True)
    audit_path = annotations / "gender_audit.csv"
    turns = pd.read_parquet(Path(settings["paths"]["interim"]) / "turns.parquet")
    overrides = load_overrides(settings["gender"]["overrides_path"])
    if not evaluate:
        people = turns[["speaker_name"]].drop_duplicates()
        people["predicted_gender"] = people["speaker_name"].map(
            lambda name: infer_gender(name, overrides)[0]
        )
        people["speaker_id"] = people["speaker_name"].map(
            lambda name: speaker_id(name, settings["gender"]["salt"])
        )
        groups = [
            frame.sample(min(len(frame), 100), random_state=settings["seed"])
            for _, frame in people.groupby("predicted_gender", sort=True)
        ]
        sample = pd.concat(groups).head(100) if groups else people.head(0)
        sample[["speaker_id", "predicted_gender"]].assign(verified_gender="").to_csv(
            audit_path, index=False
        )
        typer.echo(f"Wrote {len(sample)} audit rows to {audit_path}")
        return
    audit = pd.read_csv(audit_path)
    verified = audit[audit["verified_gender"].fillna("").str.strip().ne("")].copy()
    if verified.empty:
        raise typer.BadParameter("gender_audit.csv has no verified_gender values")
    y_true = verified["verified_gender"].str.strip()
    y_pred = verified["predicted_gender"]
    precision, recall, _, labels = precision_recall_fscore_support(
        y_true, y_pred, labels=["F", "M", "Unknown"], zero_division=0
    )
    report = pd.DataFrame(
        {
            "metric": [
                "accuracy",
                "coverage",
                "precision_F",
                "recall_F",
                "precision_M",
                "recall_M",
                "precision_Unknown",
                "recall_Unknown",
            ],
            "value": [
                accuracy_score(y_true, y_pred),
                float((y_pred == "Unknown").mean()),
                precision[0],
                recall[0],
                precision[1],
                recall[1],
                precision[2],
                recall[2],
            ],
        }
    )
    report.to_csv(Path(settings["paths"]["results"]) / "tables" / "gender_audit.csv", index=False)
    typer.echo("Wrote gender audit evaluation")


@app.command("annotate-sample")
def annotate_sample(
    n: int = typer.Option(300, "--n", min=1),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Create a stratified, rule-oversampled gold annotation todo file."""
    settings = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
    cache = Path(settings["argmining"]["sentence_cache"])
    if cache.exists():
        sentences = pd.read_parquet(cache)
    else:
        exchanges = pd.read_parquet(
            Path(settings["paths"]["interim"]) / "exchanges_enriched.parquet"
        )
        sentences = sentence_table(exchanges, cache)
    lexicons = default_lexicons()
    rule_rows = pd.DataFrame([label_sentence(text, lexicons) for text in sentences["text"]])
    sentences = pd.concat([sentences.reset_index(drop=True), rule_rows], axis=1)
    sample_n = min(n, len(sentences))
    rng = np.random.default_rng(seed)
    selected = []
    side_groups = list(sentences.groupby("side", sort=True))
    for _, group in side_groups:
        quota = min(len(group), max(1, round(sample_n * len(group) / len(sentences))))
        weights = group["label_rule"].ne("OTHER").map({True: 3.0, False: 1.0})
        priorities = rng.random(len(group)) ** (1 / weights.to_numpy())
        selected.append(group.assign(_priority=priorities).nlargest(quota, "_priority"))
    sample = pd.concat(selected).drop(columns="_priority") if selected else sentences.head(0)
    if len(sample) < sample_n:
        remaining = sentences.loc[~sentences.index.isin(sample.index)].copy()
        sample = pd.concat([sample, remaining.sample(sample_n - len(sample), random_state=seed)])
    sample["gold_label"] = ""
    output = Path(settings["argmining"]["gold_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    sidecar = output.with_suffix(".sampling.json")
    sidecar.write_text(
        json.dumps(
            {
                "n_requested": n,
                "n_written": len(sample),
                "seed": seed,
                "rule_weights": {"OTHER": 1.0, "non_OTHER": 3.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Wrote {len(sample)} annotation rows to {output}")


@app.command("annotate")
def annotate(
    annotator: str = typer.Option("annotator1", "--annotator"),
) -> None:
    """Interactively label and resume a gold annotation file."""
    settings = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
    base = settings["argmining"]["gold_path"]
    target = annotator_path(base, annotator)
    if target != Path(base) and not target.exists():
        source = pd.read_csv(base)
        source["gold_label"] = ""
        source.to_csv(target, index=False)
    TerminalAnnotator(target, annotator).run()


@app.command()
def agreement(
    first: str = typer.Option("annotator1", "--first"),
    second: str = typer.Option("annotator2", "--second"),
) -> None:
    """Compute Cohen's kappa between two annotators."""
    settings = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
    base = settings["argmining"]["gold_path"]
    value = (
        agreement_files(base, first, second)
        if annotator_path(base, second).exists()
        else compute_agreement(base, first, second)
    )
    typer.echo(f"Cohen's kappa: {value:.4f}")


@app.command()
def evaluate() -> None:
    """Evaluate rules and a classical model against completed gold labels."""
    settings = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
    gold_path = Path(settings["argmining"]["gold_path"])
    gold = pd.read_csv(gold_path)
    completed = gold[gold["gold_label"].isin(["CLAIM", "EVIDENCE", "REBUTTAL", "REQUEST", "OTHER"])]
    if completed.empty:
        raise typer.BadParameter("No completed gold labels; run annotate first")
    output = Path(settings["paths"]["results"]) / "tables"
    lexicons = default_lexicons()
    rule_rows = pd.DataFrame([label_sentence(text, lexicons) for text in completed["text"]])
    completed = pd.concat([completed.reset_index(drop=True), rule_rows], axis=1)
    rule_pred = completed["label_rule"].tolist()
    rule_f1 = f1_score(
        completed["gold_label"],
        rule_pred,
        labels=["CLAIM", "EVIDENCE", "REBUTTAL", "REQUEST", "OTHER"],
        average="macro",
    )
    threshold = float(settings["argmining"].get("macro_f1_threshold", 0.0))
    if rule_f1 < threshold:
        typer.echo(f"Rule macro-F1 {rule_f1:.3f} is below configured threshold {threshold:.3f}")
    regime = settings["argmining"].get("training_regime", "gold_cv")
    training = completed
    if regime == "silver":
        corpus = pd.read_parquet(settings["argmining"]["sentence_cache"])
        corpus_rules = pd.DataFrame([label_sentence(text, lexicons) for text in corpus["text"]])
        corpus = pd.concat([corpus.reset_index(drop=True), corpus_rules], axis=1)
        excluded = set(completed["exchange_id"])
        training = corpus[~corpus["exchange_id"].isin(excluded)]
        training = training.assign(gold_label=training["label_rule"])
    train_and_evaluate(
        completed,
        training,
        output,
        settings.get("seed", 42),
        regime,
        settings["argmining"].get("model", "logistic"),
    )
    typer.echo(f"Rules macro-F1: {rule_f1:.3f}")


@app.command()
def mine() -> None:
    """Label every cached sentence with rules or the saved hybrid model."""
    settings = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
    exchanges = pd.read_parquet(Path(settings["paths"]["interim"]) / "exchanges_enriched.parquet")
    cache = Path(settings["argmining"]["sentence_cache"])
    sentences = sentence_table(exchanges, cache)
    lexicons = default_lexicons()
    rule_rows = pd.DataFrame([label_sentence(text, lexicons) for text in sentences["text"]])
    labels = rule_rows["label_rule"].tolist()
    model_path = Path("models/argmining_model.joblib")
    if model_path.exists():
        import joblib
        from scipy.sparse import hstack

        bundle = joblib.load(model_path)
        vector = bundle["vectorizer"].transform(sentences["text"].tolist())
        predicted = bundle["model"].predict(
            hstack([vector, hand_features(sentences["text"].tolist())])
        )
        labels = [hybrid_label(rule, ml) for rule, ml in zip(labels, predicted)]
    sentences = pd.concat([sentences, rule_rows], axis=1)
    sentences["label"] = labels
    output = Path(settings["paths"]["processed"]) / "sentences_labeled.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    sentences.to_parquet(output, index=False)
    typer.echo(f"Wrote {len(sentences)} labeled sentences")


@app.command("rate-sample")
def rate_sample(
    n: int = typer.Option(40, "--n", min=1),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Collect 1-5 human quality ratings for a deterministic question sample."""
    settings = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
    ratings_path = Path("data/annotations/aqs_ratings.csv")
    exchanges = pd.read_parquet(Path(settings["paths"]["interim"]) / "exchanges_enriched.parquet")
    sample = exchanges[["exchange_id", "question_text"]].sample(
        min(n, len(exchanges)), random_state=seed
    )
    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path)
        sample = sample[~sample["exchange_id"].isin(ratings["exchange_id"])]
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [] if not ratings_path.exists() else ratings.to_dict("records")
    for row in sample.to_dict("records"):
        typer.echo(f"\nQuestion {row['exchange_id']}: {row['question_text']}")
        while True:
            answer = typer.prompt("Quality 1-5 (or s to skip)").strip().lower()
            if answer == "s":
                break
            if answer in {"1", "2", "3", "4", "5"}:
                rows.append({"exchange_id": row["exchange_id"], "rating": int(answer)})
                pd.DataFrame(rows).to_csv(ratings_path, index=False)
                break
            typer.echo("Enter 1, 2, 3, 4, 5, or s.")
    typer.echo(f"Ratings saved to {ratings_path}")


@app.command("aqs-validate")
def aqs_validate() -> None:
    """Compare human question ratings with AQS_q using a bootstrap interval."""
    settings = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
    ratings = pd.read_csv("data/annotations/aqs_ratings.csv")
    quality = pd.read_parquet(Path(settings["paths"]["processed"]) / "quality.parquet")
    merged = ratings.merge(quality[["exchange_id", "AQS_q"]], on="exchange_id").dropna()
    if len(merged) < 3:
        raise typer.BadParameter("At least three completed ratings are required")
    observed = float(spearmanr(merged["rating"], merged["AQS_q"]).statistic)
    rng = np.random.default_rng(settings.get("seed", 42))
    bootstrap = []
    for _ in range(2000):
        sample = merged.iloc[rng.integers(0, len(merged), len(merged))]
        bootstrap.append(spearmanr(sample["rating"], sample["AQS_q"]).statistic)
    output = Path(settings["paths"]["results"]) / "tables" / "aqs_validity.csv"
    pd.DataFrame(
        [
            {
                "n": len(merged),
                "spearman_rho": observed,
                "ci_low": np.nanpercentile(bootstrap, 2.5),
                "ci_high": np.nanpercentile(bootstrap, 97.5),
            }
        ]
    ).to_csv(output, index=False)
    typer.echo(f"Spearman rho: {observed:.3f}")
