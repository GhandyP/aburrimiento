"""Regression guard for behavior captured from the legacy pipeline."""

import hashlib

from aburrimiento.model import BoredomModel

# These values were captured from 0.2/python/analizador.py (recovered from git)
# before the migration. This test is the regression guard for that implementation.
EXPECTED_ALTO_VECTOR = [
    0.8,
    0.8,
    0.8,
    0.5599999999999999,
    0.8,
    0.8,
    0.8,
    0.8,
    0.8,
    0.48,
    0.8,
    0.8,
    0.8,
    0.7200000000000001,
]
EXPECTED_PREDICTIONS = {
    "bajo": "bajo",
    "medio": "medio",
    "alto": "alto",
}
EXPECTED_FEATURES_SHA256 = "d06912e75c800b5fd608444cf51217a5f668dc7cadc871e3e89ebafd212d4f34"
EXPECTED_LABELS_SHA256 = "4cbfca46f8cac39fc8416e0c5bd0d9241d83d6a785e1c5e9122adda4ab4e0bf1"
EXPECTED_FEATURE_IMPORTANCES = [
    0.128160351743,
    0.070136256718,
    0.055445185962,
    0.02012140416,
    0.117084924587,
    0.079836249187,
    0.076001958477,
    0.072079920603,
    0.106273026809,
    0.02151501148,
    0.075344021334,
    0.081659222607,
    0.054362086148,
    0.041980380185,
]
EXPECTED_PREDICT_PROBA = {
    "bajo": [0.0, 0.940593971631, 0.059406028369],
    "medio": [0.0, 0.000296352584, 0.999703647416],
    "alto": [1.0, 0.0, 0.0],
}


def test_migrated_feature_vector_and_predictions() -> None:
    analyzer = BoredomModel(data_dir=None)
    dataset = analyzer.generate(300)
    analyzer.train(dataset.features, dataset.labels)

    assert analyzer.example_for("alto").iloc[0].tolist() == EXPECTED_ALTO_VECTOR
    for level, expected in EXPECTED_PREDICTIONS.items():
        assert analyzer.predict(analyzer.example_for(level))[0] == expected


def test_migrated_dataset_and_model_outputs_match_baseline() -> None:
    analyzer = BoredomModel(data_dir=None)
    dataset = analyzer.generate(300)
    analyzer.train(dataset.features, dataset.labels)

    assert (
        hashlib.sha256(dataset.features.to_csv(index=False).encode()).hexdigest()
        == EXPECTED_FEATURES_SHA256
    )
    assert (
        hashlib.sha256(dataset.labels.to_csv(index=False).encode()).hexdigest()
        == EXPECTED_LABELS_SHA256
    )
    assert [round(float(value), 12) for value in analyzer.model.feature_importances_] == (
        EXPECTED_FEATURE_IMPORTANCES
    )
    for level, expected in EXPECTED_PREDICT_PROBA.items():
        actual = analyzer.model.predict_proba(
            analyzer.scaler.transform(analyzer.example_for(level))
        )[0]
        assert [round(float(value), 12) for value in actual] == expected
