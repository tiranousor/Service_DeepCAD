"""Convert official Text2CAD CSV annotations to this project's JSONL format.

The official Text2CAD dataset uses DeepCAD UIDs and exposes columns such as
abstract, beginner, intermediate, expert and description. The dataset is gated
by its license on Hugging Face, so this script intentionally does not download
it; point --csv to the file after accepting the dataset terms.
"""

from __future__ import annotations

import argparse
import csv
import json
import os


def load_split_map(path):
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    exact = {}
    by_basename = {}
    for split in ("train", "validation", "test"):
        for uid in data.get(split, []):
            exact[str(uid)] = (split, str(uid))
            by_basename[os.path.basename(str(uid))] = (split, str(uid))
    return exact, by_basename


def resolve_uid(uid, exact, by_basename):
    uid = str(uid).strip().replace("\\", "/")
    if uid in exact:
        return exact[uid]
    basename = os.path.basename(uid)
    return by_basename.get(basename)


def clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return ""
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="official text2cad_v1.0.csv or text2cad_v1.1.csv")
    parser.add_argument("--split-json", default="data/train_val_test_split.json")
    parser.add_argument("--output", default="data/text2cad_official.jsonl")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["beginner", "intermediate", "expert", "description"],
        choices=["abstract", "beginner", "intermediate", "expert", "description"],
    )
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    exact, by_basename = load_split_map(args.split_json)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    written = 0
    skipped_uid = 0
    processed = 0
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as src, open(
        args.output, "w", encoding="utf-8"
    ) as out:
        reader = csv.DictReader(src)
        if "uid" not in (reader.fieldnames or []):
            raise ValueError("CSV does not contain required 'uid' column")

        for row in reader:
            processed += 1
            if args.max_rows and processed > args.max_rows:
                break
            resolved = resolve_uid(row.get("uid", ""), exact, by_basename)
            if resolved is None:
                skipped_uid += 1
                continue
            split, deepcad_uid = resolved

            seen = set()
            for level in args.levels:
                text = clean_text(row.get(level))
                if not text or text in seen:
                    continue
                seen.add(text)
                out.write(
                    json.dumps(
                        {
                            "id": deepcad_uid,
                            "text": text,
                            "split": split,
                            "source": "official_text2cad_{}".format(level),
                            "language": "en",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1

    print(
        "processed_rows={}, written_captions={}, skipped_unknown_uid={}, output={}".format(
            processed, written, skipped_uid, args.output
        )
    )


if __name__ == "__main__":
    main()
