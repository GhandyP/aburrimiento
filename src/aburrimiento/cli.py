from __future__ import annotations

import argparse
from pathlib import Path

from .model import BoredomModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analizador de aburrimiento (demo).")
    parser.add_argument("--muestras", type=int, default=300)
    parser.add_argument("--nivel", type=str, default="alto")
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=None,
        help="Path to the canonical schema. Defaults to assets/schema.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    analizador = BoredomModel(schema_path=args.schema_path)
    dataset = analizador.generate(args.muestras)
    analizador.train(dataset.features, dataset.labels)

    ejemplo = analizador.example_for(args.nivel)
    prediccion = analizador.predict(ejemplo)[0]

    print("Demo completa")
    print(f"Muestras: {args.muestras}")
    print(f"Nivel solicitado: {args.nivel}")
    print(f"Prediccion: {prediccion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
