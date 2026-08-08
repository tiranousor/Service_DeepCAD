"""Text-to-CAD mapping for Service_DeepCAD.

The module converts a compact natural-language engineering description into the
same JSON schema that ``CADSequence.from_dict`` already consumes.  It is meant
as a deterministic, geometry-safe baseline for the Text-to-CAD part of the
project.  The generated history is parametric (sketch + extrude operations),
not a mesh approximation.

Supported primitives (Russian and English wording):
- cylinder / disk;
- hollow cylinder / tube;
- rectangular block / plate;
- optional coaxial circular through-hole for cylinders;
- optional centered circular through-hole for blocks.

Examples:
    "Цилиндр диаметром 20 мм высотой 35 мм"
    "Труба внешний диаметр 30 мм, внутренний диаметр 20 мм, длина 50 мм"
    "Пластина 60 x 40 x 5 мм с отверстием диаметром 10 мм"
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


class TextCADParseError(ValueError):
    """Raised when the text does not contain enough geometric information."""


@dataclass
class TextCADSpec:
    primitive: str
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    inner_radius: Optional[float] = None
    hole_radius: Optional[float] = None


_NUM = r"(-?\d+(?:[\.,]\d+)?)"


def _to_float(value: str) -> float:
    return float(value.replace(",", "."))


def _first(text: str, patterns) -> Optional[float]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _to_float(match.group(1))
    return None


def _positive(name: str, value: Optional[float]) -> Optional[float]:
    if value is not None and value <= 0:
        raise TextCADParseError(f"{name} must be positive, got {value}")
    return value


def _extract_xyz(text: str) -> Optional[Tuple[float, float, float]]:
    # Common engineering notation: 60x40x5, 60 x 40 x 5 mm.
    match = re.search(
        rf"{_NUM}\s*(?:x|х|×)\s*{_NUM}\s*(?:x|х|×)\s*{_NUM}",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return tuple(_to_float(match.group(i)) for i in range(1, 4))


def parse_text_description(description: str) -> TextCADSpec:
    if not description or not description.strip():
        raise TextCADParseError("Description is empty")

    text = description.lower().replace("ё", "е")

    outer_d = _first(text, [
        rf"(?:внешн\w*\s+)?диаметр\w*\s*(?:=|:)?\s*{_NUM}",
        rf"(?:outer|outside)\s+diameter\s*(?:=|:)?\s*{_NUM}",
        rf"diameter\s*(?:=|:)?\s*{_NUM}",
        rf"\b[dд]\s*(?:=|:)?\s*{_NUM}",
    ])
    radius = _first(text, [
        rf"(?:внешн\w*\s+)?радиус\w*\s*(?:=|:)?\s*{_NUM}",
        rf"(?:outer\s+)?radius\s*(?:=|:)?\s*{_NUM}",
        rf"\b[rр]\s*(?:=|:)?\s*{_NUM}",
    ])
    if radius is None and outer_d is not None:
        radius = outer_d / 2.0

    inner_d = _first(text, [
        rf"внутренн\w*\s+диаметр\w*\s*(?:=|:)?\s*{_NUM}",
        rf"inner\s+diameter\s*(?:=|:)?\s*{_NUM}",
    ])
    inner_r = _first(text, [
        rf"внутренн\w*\s+радиус\w*\s*(?:=|:)?\s*{_NUM}",
        rf"inner\s+radius\s*(?:=|:)?\s*{_NUM}",
    ])
    if inner_r is None and inner_d is not None:
        inner_r = inner_d / 2.0

    hole_d = _first(text, [
        rf"отверсти\w*\s+(?:диаметр\w*|d)\s*(?:=|:)?\s*{_NUM}",
        rf"hole\s+(?:diameter|d)\s*(?:=|:)?\s*{_NUM}",
        rf"диаметр\w*\s+отверсти\w*\s*(?:=|:)?\s*{_NUM}",
    ])
    hole_r = _first(text, [
        rf"отверсти\w*\s+(?:радиус\w*|r)\s*(?:=|:)?\s*{_NUM}",
        rf"hole\s+(?:radius|r)\s*(?:=|:)?\s*{_NUM}",
    ])
    if hole_r is None and hole_d is not None:
        hole_r = hole_d / 2.0

    height = _first(text, [
        rf"(?:высот\w*|длин\w*|толщин\w*)\s*(?:=|:)?\s*{_NUM}",
        rf"(?:height|length|thickness)\s*(?:=|:)?\s*{_NUM}",
        rf"\b[hл]\s*(?:=|:)?\s*{_NUM}",
    ])

    dims = _extract_xyz(text)
    length = width = None
    if dims:
        length, width, dim_h = dims
        if height is None:
            height = dim_h

    # Named dimensions for blocks override x-notation where present.
    length = _first(text, [rf"длин\w*\s*(?:=|:)?\s*{_NUM}", rf"length\s*(?:=|:)?\s*{_NUM}"]) or length
    width = _first(text, [rf"ширин\w*\s*(?:=|:)?\s*{_NUM}", rf"width\s*(?:=|:)?\s*{_NUM}"]) or width
    block_h = _first(text, [rf"(?:высот\w*|толщин\w*)\s*(?:=|:)?\s*{_NUM}", rf"(?:height|thickness)\s*(?:=|:)?\s*{_NUM}"])
    if block_h is not None:
        height = block_h

    is_tube = any(k in text for k in ("труб", "полый цилиндр", "hollow cylinder", "tube", "pipe"))
    is_cylinder = any(k in text for k in ("цилиндр", "диск", "cylinder", "disk"))
    is_block = any(k in text for k in (
        "пластин", "брусок", "параллелепипед", "прямоугольн", "block", "plate", "box", "cuboid"
    ))

    if is_tube:
        primitive = "tube"
    elif is_cylinder:
        primitive = "cylinder"
    elif is_block or dims:
        primitive = "block"
    elif radius is not None:
        primitive = "cylinder"
    else:
        raise TextCADParseError(
            "Unsupported or ambiguous primitive. Mention cylinder/tube/block/plate and its dimensions."
        )

    radius = _positive("radius", radius)
    inner_r = _positive("inner radius", inner_r)
    hole_r = _positive("hole radius", hole_r)
    height = _positive("height", height)
    length = _positive("length", length)
    width = _positive("width", width)

    if primitive in ("cylinder", "tube"):
        if radius is None or height is None:
            raise TextCADParseError("Cylinder/tube requires diameter or radius and height/length")
        if primitive == "tube" and inner_r is None:
            raise TextCADParseError("Tube requires inner diameter or inner radius")
        if inner_r is not None and inner_r >= radius:
            raise TextCADParseError("Inner radius must be smaller than outer radius")
        if hole_r is not None and hole_r >= radius:
            raise TextCADParseError("Hole radius must be smaller than cylinder radius")
    else:
        if length is None or width is None or height is None:
            raise TextCADParseError("Block/plate requires length, width and height (for example 60x40x5 mm)")
        if hole_r is not None and hole_r * 2 >= min(length, width):
            raise TextCADParseError("Hole diameter must fit inside the block profile")

    return TextCADSpec(
        primitive=primitive,
        length=length,
        width=width,
        height=height,
        radius=radius,
        inner_radius=inner_r,
        hole_radius=hole_r,
    )


def _plane_xy() -> Dict[str, Any]:
    return {
        "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
        "z_axis": {"x": 0.0, "y": 0.0, "z": 1.0},
        "x_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
        "y_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
    }


def _circle(radius: float, is_outer: bool = True) -> Dict[str, Any]:
    return {
        "is_outer": is_outer,
        "profile_curves": [{
            "type": "Circle3D",
            "center_point": {"x": 0.0, "y": 0.0},
            "radius": float(radius),
            "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
        }],
    }


def _rectangle(length: float, width: float) -> Dict[str, Any]:
    hx, hy = length / 2.0, width / 2.0
    pts = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy)]
    curves = []
    for start, end in zip(pts[:-1], pts[1:]):
        curves.append({
            "type": "Line3D",
            "start_point": {"x": start[0], "y": start[1]},
            "end_point": {"x": end[0], "y": end[1]},
        })
    return {"is_outer": True, "profile_curves": curves}


def _extrude(entity_id: str, sketch_id: str, profile_id: str, distance: float, operation: str) -> Dict[str, Any]:
    return {
        "start_extent": {"type": "ProfilePlaneStartDefinition"},
        "profiles": [{"sketch": sketch_id, "profile": profile_id}],
        "operation": operation,
        "extent_type": "OneSideFeatureExtentType",
        "extent_one": {"distance": {"value": float(distance)}},
    }


def spec_to_deepcad_json(spec: TextCADSpec) -> Dict[str, Any]:
    """Compile a parsed spec into the JSON schema expected by CADSequence.from_dict."""
    entities: Dict[str, Any] = {}
    sequence = []

    if spec.primitive in ("cylinder", "tube"):
        assert spec.radius is not None and spec.height is not None
        loops = [_circle(spec.radius, True)]
        cut_radius = spec.inner_radius if spec.primitive == "tube" else spec.hole_radius
        if cut_radius is not None:
            loops.append(_circle(cut_radius, False))

        entities["sk_1"] = {
            "transform": _plane_xy(),
            "profiles": {"p_1": {"loops": loops}},
        }
        entities["extrude_1"] = _extrude(
            "extrude_1", "sk_1", "p_1", spec.height, "NewBodyFeatureOperation"
        )
        sequence.append({"type": "ExtrudeFeature", "entity": "extrude_1"})

        r = spec.radius
        bbox_min = {"x": -r, "y": -r, "z": 0.0}
        bbox_max = {"x": r, "y": r, "z": spec.height}

    elif spec.primitive == "block":
        assert spec.length is not None and spec.width is not None and spec.height is not None
        loops = [_rectangle(spec.length, spec.width)]
        if spec.hole_radius is not None:
            loops.append(_circle(spec.hole_radius, False))

        entities["sk_1"] = {
            "transform": _plane_xy(),
            "profiles": {"p_1": {"loops": loops}},
        }
        entities["extrude_1"] = _extrude(
            "extrude_1", "sk_1", "p_1", spec.height, "NewBodyFeatureOperation"
        )
        sequence.append({"type": "ExtrudeFeature", "entity": "extrude_1"})

        hx, hy = spec.length / 2.0, spec.width / 2.0
        bbox_min = {"x": -hx, "y": -hy, "z": 0.0}
        bbox_max = {"x": hx, "y": hy, "z": spec.height}
    else:
        raise TextCADParseError(f"Unsupported primitive: {spec.primitive}")

    return {
        "sequence": sequence,
        "entities": entities,
        "properties": {
            "bounding_box": {
                "max_point": bbox_max,
                "min_point": bbox_min,
            }
        },
    }


def text_to_deepcad_json(description: str) -> Dict[str, Any]:
    return spec_to_deepcad_json(parse_text_description(description))
