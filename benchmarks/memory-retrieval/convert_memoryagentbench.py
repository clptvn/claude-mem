#!/usr/bin/env python3
"""Convert official MemoryAgentBench parquet rows into portable JSONL tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as parquet


def json_safe(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    table = parquet.read_table(args.input)
    rows = table.to_pylist()
    if args.limit is not None:
        rows = rows[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            metadata = json_safe(row.get("metadata") or {})
            source = str(metadata.get("source") or f"row-{row_index}")
            questions = json_safe(row.get("questions") or [])
            answers = json_safe(row.get("answers") or [])
            handle.write(
                json.dumps(
                    {
                        "task_id": f"{source}:{row_index}",
                        "competency": output.stem,
                        "source": source,
                        "context": row.get("context") or "",
                        "questions": questions,
                        "answers": answers,
                        "metadata": metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "input": str(Path(args.input).resolve()),
                "output": str(output.resolve()),
                "rows": len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
