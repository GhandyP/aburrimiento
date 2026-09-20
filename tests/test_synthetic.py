import hashlib

import pytest

from aburrimiento.synthetic import (
    DEFAULT_CLASS_PROFILES,
    DEFAULT_FEATURE_MODIFIERS,
    DatasetBundle,
    SyntheticDataGenerator,
)

FEATURES = ["racismo_sistemico", "alta_excitacion", "angustia_profunda", "neutral"]
LEVELS = ["bajo", "medio", "alto"]


def test_same_seed_is_identical_and_different_seed_differs() -> None:
    first = SyntheticDataGenerator(FEATURES, LEVELS, seed=42).generate(300)
    same = SyntheticDataGenerator(FEATURES, LEVELS, seed=42).generate(300)
    other = SyntheticDataGenerator(FEATURES, LEVELS, seed=43).generate(300)

    def digest(dataset: DatasetBundle) -> str:
        return hashlib.sha256(
            dataset.features.to_csv(index=False).encode()
            + dataset.labels.to_csv(index=False).encode()
        ).hexdigest()

    assert digest(first) == digest(same)
    assert digest(first) != digest(other)


def test_values_classes_and_balance() -> None:
    dataset = SyntheticDataGenerator(FEATURES, LEVELS).generate(3000)
    assert dataset.features.to_numpy().min() >= 0
    assert dataset.features.to_numpy().max() <= 1
    counts = dataset.labels.value_counts()
    assert set(counts.index) == set(LEVELS)
    assert all(abs(count / 3000 - 1 / 3) < 0.06 for count in counts)


def test_modifier_lowers_equivalent_feature_mean() -> None:
    dataset = SyntheticDataGenerator(["racismo_sistemico", "neutral"], ["medio"], seed=42).generate(
        1000
    )
    assert dataset.features["racismo_sistemico"].mean() < dataset.features["neutral"].mean()


def test_default_snapshot_and_missing_profile() -> None:
    assert [(p.level_id, p.base, p.spread) for p in DEFAULT_CLASS_PROFILES] == [
        ("bajo", 0.3, 0.2),
        ("medio", 0.5, 0.15),
        ("alto", 0.8, 0.15),
    ]
    assert DEFAULT_FEATURE_MODIFIERS == {
        "racismo_sistemico": 0.7,
        "alta_excitacion": 0.6,
        "angustia_profunda": 0.9,
    }
    with pytest.raises(ValueError, match="No class profile"):
        SyntheticDataGenerator(["neutral"], ["desconocido"]).generate(1)


def test_limitation_is_explicit_in_module_and_generator_docs() -> None:
    import aburrimiento.synthetic as synthetic

    text = (synthetic.__doc__ or "") + (synthetic.SyntheticDataGenerator.__doc__ or "")
    assert "self-consistency" in text
    assert "not evidence about boredom" in text
