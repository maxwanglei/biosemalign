"""Export the core contracts as JSON Schema.

Run as ``python -m biosemalign.schemas.export``. The output is committed under
``docs/schemas/``; CI fails when it drifts, which turns every schema change
into a reviewable diff instead of something noticed after data has already been
written under it (Milestone 0, exit criterion 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from biosemalign.schemas import ROOT_CONTRACTS, SCHEMA_VERSION

__all__ = ["export_schemas", "main"]

DEFAULT_OUTPUT = Path("docs/schemas")


def export_schemas(output_dir: str | Path = DEFAULT_OUTPUT) -> list[Path]:
    """Write one JSON Schema file per root contract."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for model in ROOT_CONTRACTS:
        schema = model.model_json_schema(mode="serialization")
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["x-biosemalign-schema-version"] = SCHEMA_VERSION

        path = directory / f"{_snake(model.__name__)}.schema.json"
        path.write_text(
            json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def _snake(name: str) -> str:
    out: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index:
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed schemas differ from the models",
    )
    args = parser.parse_args(argv)

    directory = Path(args.output_dir)
    if args.check:
        stale: list[str] = []
        for model in ROOT_CONTRACTS:
            path = directory / f"{_snake(model.__name__)}.schema.json"
            schema = model.model_json_schema(mode="serialization")
            schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
            schema["x-biosemalign-schema-version"] = SCHEMA_VERSION
            expected = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                stale.append(str(path))
        if stale:
            print("Committed JSON Schema is out of date:", file=sys.stderr)
            for stale_path in stale:
                print(f"  {stale_path}", file=sys.stderr)
            print(
                "\nRegenerate with: python -m biosemalign.schemas.export",
                file=sys.stderr,
            )
            return 1
        print(f"JSON Schema is current ({len(ROOT_CONTRACTS)} contracts).")
        return 0

    written = export_schemas(directory)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
