import argparse
from pathlib import Path

from analizador import AnalizadorAburrimiento


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analizador de aburrimiento (demo).")
    parser.add_argument("--muestras", type=int, default=300)
    parser.add_argument("--nivel", type=str, default="alto")
    parser.add_argument("--data-dir", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    analizador = AnalizadorAburrimiento(data_dir=args.data_dir)
    dataset = analizador.generar_datos(args.muestras)
    analizador.entrenar(dataset.features, dataset.labels)

    ejemplo = analizador.construir_ejemplo(args.nivel)
    prediccion = analizador.predecir(ejemplo)[0]

    print("Demo completa")
    print(f"Muestras: {args.muestras}")
    print(f"Nivel solicitado: {args.nivel}")
    print(f"Prediccion: {prediccion}")


if __name__ == "__main__":
    main()
