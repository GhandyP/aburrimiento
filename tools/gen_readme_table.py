from __future__ import annotations

from pathlib import Path

from aburrimiento.schema import Schema, load_schema

README_PATH = Path(__file__).parents[1] / "README.md"
BEGIN_MARKER = "<!-- BEGIN GENERATED INDICATORS -->"
END_MARKER = "<!-- END GENERATED INDICATORS -->"


def render_indicator_table(schema: Schema) -> str:
    """Render the indicator table body in canonical schema order."""
    groups = {group.id: group.labels.es for group in schema.groups}
    lines = [
        "| Group | Indicator ID | Spanish label | Theoretical basis |",
        "|---|---|---|---|",
    ]
    for indicator in sorted(schema.indicators, key=lambda item: item.order):
        lines.append(
            f"| {groups[indicator.group_id]} | `{indicator.id}` | "
            f"{indicator.labels.es} | {indicator.theoretical_basis} |"
        )
    return "\n".join(lines)


def regenerate_readme(readme: str, schema: Schema) -> str:
    """Replace only the content between the generated-table markers."""
    begin = readme.find(BEGIN_MARKER)
    end = readme.find(END_MARKER)
    if begin == -1 or end == -1:
        raise ValueError(f"README must contain both {BEGIN_MARKER!r} and {END_MARKER!r}")
    if begin > end:
        raise ValueError("README generated-table markers are out of order")
    content_start = begin + len(BEGIN_MARKER)
    return readme[:content_start] + "\n" + render_indicator_table(schema) + "\n" + readme[end:]


def main() -> None:
    schema = load_schema()
    original = README_PATH.read_text(encoding="utf-8")
    updated = regenerate_readme(original, schema)
    README_PATH.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
