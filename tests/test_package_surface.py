from pathlib import Path

import pytest

from aburrimiento.cli import main
from aburrimiento.model import BoredomModel

EXPECTED_FEATURES = [
    "reflejo_sistemas_culturales",
    "productividad_capitalista",
    "alienacion_neoliberal",
    "racismo_sistemico",
    "malestar_generalizado",
    "carencia_de_sentido",
    "restriccion_de_libertad",
    "frustracion_de_agencia",
    "desenganche",
    "alta_excitacion",
    "inatencion",
    "percepcion_tiempo_lenta",
    "estrategias_bloqueadas",
    "angustia_profunda",
]


def test_canonical_fields_and_levels_load_in_order() -> None:
    analyzer = BoredomModel()
    assert analyzer.feature_names == EXPECTED_FEATURES
    assert analyzer.levels == ["bajo", "medio", "alto"]


def test_cli_main_returns_an_integer_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert isinstance(main(["--muestras", "1"]), int)
    capsys.readouterr()


def test_model_rejects_missing_schema_file() -> None:
    with pytest.raises(FileNotFoundError, match="Canonical schema file does not exist"):
        BoredomModel(schema_path=Path("does-not-exist/schema.json"))
