from pathlib import Path

from aburrimiento.evaluate import evaluate, format_markdown, write_report


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


def test_write_report_adds_only_write_time_timestamp(tmp_path: Path) -> None:
    report = evaluate(n_samples=90, cv_folds=3)
    json_path, markdown_path = write_report(report, tmp_path)
    assert json_path.name == "evaluation.json"
    assert markdown_path.name == "evaluation.md"
    assert '"generated_at"' in json_path.read_text(encoding="utf-8")
    assert markdown_path.read_text(encoding="utf-8").startswith("# Evaluation numbers")
