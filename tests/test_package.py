"""Smoke test: the package is importable and installable in editable mode."""

import aburrimiento


def test_package_is_importable() -> None:
    assert aburrimiento.__name__ == "aburrimiento"


def test_package_declares_no_public_exports_yet() -> None:
    assert aburrimiento.__all__ == []
