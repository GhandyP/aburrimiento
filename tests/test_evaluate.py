from pathlib import Path

from aburrimiento.evaluate import (
    compare_models_across_seeds,
    evaluate,
    format_markdown,
    write_report,
)


def test_evaluate_is_deterministic_and_records_comparison() -> None:
    first = evaluate(n_samples=180, cv_folds=3)
    second = evaluate(n_samples=180, cv_folds=3)
    assert first.to_dict() == second.to_dict()
    assert first.split_sizes == {"train": 108, "validation": 36, "test": 36}
    assert len(first.tree.scaled.cv_scores) == 3
    assert len(first.tree.unscaled.cv_scores) == 3
    assert set(first.feature_importances)
    assert all(isinstance(note, str) and len(note.split()) > 5 for note in first.notes)
    assert "measured accuracy" in format_markdown(first)
    assert "| Class | Precision | Recall | F1 | Support |" in format_markdown(first)
    assert "Confusion matrix labels" in format_markdown(first)
    assert "standard deviation" in format_markdown(first)
    assert "tree splits are expected to be invariant" in format_markdown(first)


def test_seed_sweep_is_deterministic_and_has_one_valid_row_per_seed() -> None:
    first = compare_models_across_seeds(n_samples=180, base_seed=7, n_seeds=4)
    second = compare_models_across_seeds(n_samples=180, base_seed=7, n_seeds=4)

    assert first == second
    assert len(first.rows) == 4
    assert [row.seed for row in first.rows] == [7, 8, 9, 10]
    assert all(0 <= row.forest_accuracy <= 1 for row in first.rows)
    assert all(0 <= row.logistic_accuracy <= 1 for row in first.rows)
    assert all(
        row.difference_percentage_points == (row.logistic_accuracy - row.forest_accuracy) * 100
        for row in first.rows
    )
    assert first.logistic_wins + first.forest_wins + first.ties == 4


def test_seed_sweep_does_not_claim_winner_when_mean_is_smaller_than_std() -> None:
    sweep = compare_models_across_seeds(n_samples=600, base_seed=1, n_seeds=12)

    assert abs(sweep.mean_difference_percentage_points) < sweep.standard_deviation_percentage_points
    assert "indistinguishable" in sweep.conclusion
    assert "wins this seed sweep" not in sweep.conclusion


def test_write_report_adds_only_write_time_timestamp(tmp_path: Path) -> None:
    report = evaluate(n_samples=90, cv_folds=3)
    json_path, markdown_path = write_report(report, tmp_path)
    assert json_path.name == "evaluation.json"
    assert markdown_path.name == "evaluation.md"
    assert '"generated_at"' in json_path.read_text(encoding="utf-8")
    assert markdown_path.read_text(encoding="utf-8").startswith("# Evaluation numbers")
