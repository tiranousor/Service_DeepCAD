"""Merge multiple Text2CAD JSONL annotation sources with exact de-duplication."""

from __future__ import annotations

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    seen = set()
    written = 0
    source_counts = {}
    split_counts = {}

    with open(args.output, "w", encoding="utf-8") as out:
        for path in args.inputs:
            with open(path, "r", encoding="utf-8") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (row.get("id"), row.get("text"), row.get("split"))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    source = row.get("source", "unknown")
                    split = row.get("split", "unknown")
                    source_counts[source] = source_counts.get(source, 0) + 1
                    split_counts[split] = split_counts.get(split, 0) + 1

    print(json.dumps({
        "written": written,
        "output": args.output,
        "splits": split_counts,
        "sources": source_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
