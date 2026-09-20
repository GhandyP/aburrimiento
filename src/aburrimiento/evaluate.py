"""Deterministic, measured evaluation of the synthetic boredom classifier."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import Schema, load_schema
from .synthetic import DEFAULT_SEED, DatasetBundle, SyntheticDataGenerator

NOTES = (
    "The labels are drawn from the same class-conditional distributions the model then learns, "
    "so accuracy measured here is a measure of self-consistency of the generator, not evidence "
    "about boredom in the world.",
    "There is no external validation anywhere in this repository.",
)


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ModelReport:
    name: str
    accuracy: float
    kappa: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]
    labels: list[str]
    cv_scores: list[float]
    cv_mean: float
    cv_std: float


@dataclass(frozen=True)
class ScalerComparison:
    scaled: ModelReport
    unscaled: ModelReport
    metrics_differ: bool


@dataclass(frozen=True)
class SeedSweepRow:
    seed: int
    forest_accuracy: float
    logistic_accuracy: float
    difference_samples: int
    difference_percentage_points: float


@dataclass(frozen=True)
class SeedSweepReport:
    base_seed: int
    n_seeds: int
    rows: list[SeedSweepRow]
    mean_difference_percentage_points: float
    standard_deviation_percentage_points: float
    logistic_wins: int
    forest_wins: int
    ties: int
    conclusion: str


@dataclass(frozen=True)
class EvaluationReport:
    seed: int
    n_samples: int
    split_sizes: dict[str, int]
    feature_importances: dict[str, float]
    tree: ScalerComparison
    logistic_regression: ModelReport
    dummy: ModelReport
    seed_sweep: SeedSweepReport
    notes: list[str] = field(default_factory=lambda: list(NOTES))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _model_report(
    name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[str],
    cv: StratifiedKFold,
) -> ModelReport:
    fitted = model.fit(x_train, y_train)
    predicted = fitted.predict(x_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=labels, zero_division=0
    )
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
    return ModelReport(
        name=name,
        accuracy=float(accuracy_score(y_test, predicted)),
        kappa=float(cohen_kappa_score(y_test, predicted, labels=labels)),
        macro_f1=float(
            f1_score(y_test, predicted, labels=labels, average="macro", zero_division=0)
        ),
        weighted_f1=float(
            f1_score(y_test, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        per_class={
            label: ClassMetrics(float(p), float(r), float(f), int(s))
            for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        },
        confusion_matrix=confusion_matrix(y_test, predicted, labels=labels).tolist(),
        labels=labels,
        cv_scores=[float(score) for score in scores],
        cv_mean=float(scores.mean()),
        cv_std=float(scores.std()),
    )


def _tree(scaled: bool, seed: int) -> Pipeline | RandomForestClassifier:
    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_split=5,
        random_state=seed,
        class_weight="balanced",
    )
    if scaled:
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    return classifier


def evaluate(
    schema: Schema | None = None,
    n_samples: int = 3000,
    seed: int = DEFAULT_SEED,
    cv_folds: int = 5,
) -> EvaluationReport:
    """Evaluate models without I/O, timestamps, or other observable side effects."""
    resolved = load_schema() if schema is None else schema
    feature_names = list(resolved.indicator_ids)
    labels = list(resolved.level_ids)
    dataset: DatasetBundle = SyntheticDataGenerator(feature_names, labels, seed).generate(n_samples)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        dataset.features,
        dataset.labels,
        test_size=0.2,
        random_state=seed,
        stratify=dataset.labels,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=seed,
        stratify=y_train_val,
    )
    # Validation is intentionally recorded as a held-out partition; model comparison uses test.
    _ = x_validation, y_validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    scaled = _model_report(
        "random_forest_scaled", _tree(True, seed), x_train, y_train, x_test, y_test, labels, cv
    )
    unscaled = _model_report(
        "random_forest_unscaled", _tree(False, seed), x_train, y_train, x_test, y_test, labels, cv
    )
    logistic = _model_report(
        "logistic_regression",
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        x_train,
        y_train,
        x_test,
        y_test,
        labels,
        cv,
    )
    dummy = _model_report(
        "dummy_most_frequent",
        DummyClassifier(strategy="most_frequent"),
        x_train,
        y_train,
        x_test,
        y_test,
        labels,
        cv,
    )
    feature_importance_model = _tree(False, seed).fit(x_train, y_train)
    metrics_differ = any(
        (
            scaled.accuracy != unscaled.accuracy,
            scaled.kappa != unscaled.kappa,
            scaled.macro_f1 != unscaled.macro_f1,
            scaled.weighted_f1 != unscaled.weighted_f1,
            scaled.per_class != unscaled.per_class,
            scaled.confusion_matrix != unscaled.confusion_matrix,
            scaled.cv_scores != unscaled.cv_scores,
            scaled.cv_mean != unscaled.cv_mean,
            scaled.cv_std != unscaled.cv_std,
        )
    )
    sweep = compare_models_across_seeds(resolved, n_samples=n_samples)
    return EvaluationReport(
        seed=seed,
        n_samples=n_samples,
        split_sizes={"train": len(x_train), "validation": len(x_validation), "test": len(x_test)},
        feature_importances=dict(
            sorted(
                (
                    (feature, float(importance))
                    for feature, importance in zip(
                        feature_names, feature_importance_model.feature_importances_, strict=True
                    )
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        tree=ScalerComparison(scaled, unscaled, metrics_differ),
        logistic_regression=logistic,
        dummy=dummy,
        seed_sweep=sweep,
    )


def compare_models_across_seeds(
    schema: Schema | None = None,
    n_samples: int = 3000,
    base_seed: int = 1,
    n_seeds: int = 12,
) -> SeedSweepReport:
    """Compare the forest and linear baseline over deterministic seeded splits."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    resolved = load_schema() if schema is None else schema
    feature_names = list(resolved.indicator_ids)
    labels = list(resolved.level_ids)
    rows: list[SeedSweepRow] = []
    for seed in range(base_seed, base_seed + n_seeds):
        dataset = SyntheticDataGenerator(feature_names, labels, seed).generate(n_samples)
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            dataset.features,
            dataset.labels,
            test_size=0.2,
            random_state=seed,
            stratify=dataset.labels,
        )
        x_train, _, y_train, _ = train_test_split(
            x_train_val,
            y_train_val,
            test_size=0.25,
            random_state=seed,
            stratify=y_train_val,
        )
        forest = _tree(False, seed).fit(x_train, y_train)
        logistic = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ).fit(x_train, y_train)
        forest_accuracy = float(accuracy_score(y_test, forest.predict(x_test)))
        logistic_accuracy = float(accuracy_score(y_test, logistic.predict(x_test)))
        difference_samples = round((logistic_accuracy - forest_accuracy) * len(y_test))
        rows.append(
            SeedSweepRow(
                seed=seed,
                forest_accuracy=forest_accuracy,
                logistic_accuracy=logistic_accuracy,
                difference_samples=difference_samples,
                difference_percentage_points=(logistic_accuracy - forest_accuracy) * 100,
            )
        )
    differences = [row.difference_percentage_points for row in rows]
    mean_difference = float(pd.Series(differences).mean())
    standard_deviation = float(pd.Series(differences).std(ddof=0))
    logistic_wins = sum(row.difference_samples > 0 for row in rows)
    forest_wins = sum(row.difference_samples < 0 for row in rows)
    ties = sum(row.difference_samples == 0 for row in rows)
    if abs(mean_difference) < standard_deviation:
        conclusion = (
            "The two models are statistically indistinguishable on this generator; "
            "the linear model matches the 120-tree forest."
        )
    elif mean_difference > 0:
        conclusion = "The logistic regression wins this seed sweep."
    else:
        conclusion = "The 120-tree forest wins this seed sweep."
    return SeedSweepReport(
        base_seed=base_seed,
        n_seeds=n_seeds,
        rows=rows,
        mean_difference_percentage_points=mean_difference,
        standard_deviation_percentage_points=standard_deviation,
        logistic_wins=logistic_wins,
        forest_wins=forest_wins,
        ties=ties,
        conclusion=conclusion,
    )


def _report_json(report: EvaluationReport) -> dict[str, object]:
    return report.to_dict()


def write_report(report: EvaluationReport, reports_dir: Path) -> tuple[Path, Path]:
    """Write stable JSON content and human-readable Markdown with a write timestamp."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "evaluation.json"
    markdown_path = reports_dir / "evaluation.md"
    payload = _report_json(report)
    payload["generated_at"] = datetime.now(UTC).isoformat()
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(format_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _markdown_model(report: ModelReport) -> list[str]:
    lines = [
        f"### {report.name}",
        f"- Test accuracy: `{report.accuracy:.6f}`",
        f"- Cohen's kappa: `{report.kappa:.6f}`",
        f"- Macro F1: `{report.macro_f1:.6f}`; weighted F1: `{report.weighted_f1:.6f}`",
        f"- CV accuracy: `{[round(x, 6) for x in report.cv_scores]}`; "
        f"mean `{report.cv_mean:.6f}`, standard deviation `{report.cv_std:.6f}`",
        f"- Confusion matrix labels: `{report.labels}`",
        "- Per-class metrics:",
        "  | Class | Precision | Recall | F1 | Support |",
        "  |---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"  | {label} | {metrics.precision:.6f} | {metrics.recall:.6f} | "
        f"{metrics.f1:.6f} | {metrics.support} |"
        for label, metrics in report.per_class.items()
    )
    lines.append(f"- Confusion matrix: `{report.confusion_matrix}`")
    return lines


def format_markdown(report: EvaluationReport) -> str:
    lines = [
        "# Evaluation numbers measure generator self-consistency, not boredom in the world",
        "",
        f"- Samples: `{report.n_samples}`; seed: `{report.seed}`",
        f"- Stratified split sizes: `{report.split_sizes}`",
        *[f"- {note}" for note in report.notes],
        "",
        "## Random forest scaler comparison",
    ]
    lines.extend(_markdown_model(report.tree.scaled))
    lines.extend(_markdown_model(report.tree.unscaled))
    lines.extend(
        [
            f"- Scaler metrics differ: **{report.tree.metrics_differ}**",
            "- The tree is shown in scaled and unscaled forms because tree splits are expected "
            "to be invariant to feature scale; the comparison makes that assumption explicit. "
            "The linear baseline and dummy floor each have one model report.",
            "",
            "## Model comparison",
        ]
    )
    lines.extend(_markdown_model(report.logistic_regression))
    lines.extend(_markdown_model(report.dummy))
    lines.extend(
        [
            "",
            "## Seed-sweep comparison",
            "",
            f"- Seeds: `{report.seed_sweep.base_seed}` through `"
            f"{report.seed_sweep.base_seed + report.seed_sweep.n_seeds - 1}` "
            f"({report.seed_sweep.n_seeds} total)",
            "- Each row uses the same stratified train/validation/test split and compares "
            "the unscaled 120-tree forest with logistic regression.",
            "",
            "| Seed | Forest accuracy | Logistic accuracy | Difference (samples) | "
            "Difference (points) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row.seed} | {row.forest_accuracy:.6f} | {row.logistic_accuracy:.6f} | "
        f"{row.difference_samples:+d} | {row.difference_percentage_points:+.3f} |"
        for row in report.seed_sweep.rows
    )
    lines.extend(
        [
            "",
            f"- Mean difference (LogReg - forest): "
            f"`{report.seed_sweep.mean_difference_percentage_points:+.3f}` percentage points; "
            f"standard deviation: `{report.seed_sweep.standard_deviation_percentage_points:.3f}`.",
            f"- Wins: logistic regression `{report.seed_sweep.logistic_wins}`, "
            f"forest `{report.seed_sweep.forest_wins}`, ties `{report.seed_sweep.ties}`.",
            f"- Conclusion: {report.seed_sweep.conclusion}",
            "",
            "A small gap between the tree model and the linear baseline is evidence that the "
            "classification task is largely linearly separable in the generator, which says "
            "something about the generator and nothing about boredom. The dummy floor is "
            "reported with its measured accuracy above, not characterized as trivial without "
            "that measurement.",
            "",
            "## Feature importances",
            "",
        ]
    )
    lines.extend(
        f"- `{feature}`: `{importance:.12f}`"
        for feature, importance in sorted(
            report.feature_importances.items(), key=lambda item: item[1], reverse=True
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    write_report(evaluate(), Path("reports"))


if __name__ == "__main__":
    main()
