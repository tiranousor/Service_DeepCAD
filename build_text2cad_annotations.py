"""Create paired text/CAD annotations directly from the DeepCAD JSON dataset.

This provides a reproducible training corpus without changing the original CAD
representation.  Captions describe the construction history in engineering
language (sketch primitives, positions and extrude/boolean operations).

The resulting JSONL can be consumed by ``train_text2cad.py``.

Example:
    python build_text2cad_annotations.py \
        --data-root data \
        --output data/text2cad_annotations.jsonl \
        --language ru
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List


def fnum(value) -> str:
    return ("{:.4f}".format(float(value))).rstrip("0").rstrip(".")


def point2(stat: Dict[str, Any]) -> str:
    return "({}, {})".format(fnum(stat["x"]), fnum(stat["y"]))


def point3(stat: Dict[str, Any]) -> str:
    return "({}, {}, {})".format(fnum(stat["x"]), fnum(stat["y"]), fnum(stat["z"]))


def describe_curve(curve: Dict[str, Any], language: str) -> str:
    ctype = curve["type"]
    if language == "ru":
        if ctype == "Line3D":
            return "линия из {} в {}".format(point2(curve["start_point"]), point2(curve["end_point"]))
        if ctype == "Circle3D":
            return "окружность с центром {} и радиусом {}".format(
                point2(curve["center_point"]), fnum(curve["radius"])
            )
        if ctype == "Arc3D":
            return "дуга из {} в {}, центр {}, радиус {}".format(
                point2(curve["start_point"]), point2(curve["end_point"]),
                point2(curve["center_point"]), fnum(curve["radius"])
            )
    else:
        if ctype == "Line3D":
            return "line from {} to {}".format(point2(curve["start_point"]), point2(curve["end_point"]))
        if ctype == "Circle3D":
            return "circle centered at {} with radius {}".format(
                point2(curve["center_point"]), fnum(curve["radius"])
            )
        if ctype == "Arc3D":
            return "arc from {} to {}, center {}, radius {}".format(
                point2(curve["start_point"]), point2(curve["end_point"]),
                point2(curve["center_point"]), fnum(curve["radius"])
            )
    return ctype


def describe_model(data: Dict[str, Any], language: str = "ru") -> str:
    entities = data["entities"]
    chunks: List[str] = []

    op_ru = {
        "NewBodyFeatureOperation": "создать новое тело",
        "JoinFeatureOperation": "объединить с телом",
        "CutFeatureOperation": "вырезать из тела",
        "IntersectFeatureOperation": "оставить пересечение",
    }
    op_en = {
        "NewBodyFeatureOperation": "create a new body",
        "JoinFeatureOperation": "join with the body",
        "CutFeatureOperation": "cut from the body",
        "IntersectFeatureOperation": "keep the intersection",
    }

    step_no = 0
    for item in data.get("sequence", []):
        if item.get("type") != "ExtrudeFeature":
            continue
        step_no += 1
        ext = entities[item["entity"]]
        distance = ext["extent_one"]["distance"]["value"]
        operation = ext["operation"]

        profile_descriptions = []
        for profile_ref in ext.get("profiles", []):
            sketch = entities[profile_ref["sketch"]]
            profile = sketch["profiles"][profile_ref["profile"]]
            transform = sketch["transform"]
            loops = []
            for loop_no, loop in enumerate(profile.get("loops", []), 1):
                curves = ", ".join(describe_curve(c, language) for c in loop.get("profile_curves", []))
                if language == "ru":
                    loops.append("контур {}: {}".format(loop_no, curves))
                else:
                    loops.append("loop {}: {}".format(loop_no, curves))

            if language == "ru":
                profile_descriptions.append(
                    "эскиз в начале координат {} с нормалью {}; {}".format(
                        point3(transform["origin"]), point3(transform["z_axis"]), "; ".join(loops)
                    )
                )
            else:
                profile_descriptions.append(
                    "sketch at origin {} with normal {}; {}".format(
                        point3(transform["origin"]), point3(transform["z_axis"]), "; ".join(loops)
                    )
                )

        if language == "ru":
            chunks.append(
                "Шаг {}: {}. Выполнить выдавливание на {} и {}.".format(
                    step_no, " ".join(profile_descriptions), fnum(distance), op_ru.get(operation, operation)
                )
            )
        else:
            chunks.append(
                "Step {}: {}. Extrude by {} and {}.".format(
                    step_no, " ".join(profile_descriptions), fnum(distance), op_en.get(operation, operation)
                )
            )

    return " ".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/text2cad_annotations.jsonl")
    parser.add_argument("--language", choices=["ru", "en"], default="ru")
    args = parser.parse_args()

    split_path = os.path.join(args.data_root, "train_val_test_split.json")
    cad_root = os.path.join(args.data_root, "cad_json")
    with open(split_path, "r", encoding="utf-8") as fp:
        splits = json.load(fp)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = 0
    skipped = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for split in ("train", "validation", "test"):
            for data_id in splits.get(split, []):
                path = os.path.join(cad_root, data_id + ".json")
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    text = describe_model(data, args.language)
                    if not text:
                        skipped += 1
                        continue
                    out.write(json.dumps({
                        "id": data_id,
                        "text": text,
                        "split": split,
                        "source": "deepcad_history_template",
                    }, ensure_ascii=False) + "\n")
                    written += 1
                except Exception as exc:
                    skipped += 1
                    print("skip {}: {}".format(data_id, exc))

    print("written={}, skipped={}, output={}".format(written, skipped, args.output))


if __name__ == "__main__":
    main()
