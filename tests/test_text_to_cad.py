import unittest

from text_to_cad import parse_text_description, text_to_deepcad_json, TextCADParseError


class TextToCADTests(unittest.TestCase):
    def test_cylinder_russian(self):
        spec = parse_text_description("Цилиндр диаметром 20 мм высотой 35 мм")
        self.assertEqual(spec.primitive, "cylinder")
        self.assertAlmostEqual(spec.radius, 10.0)
        self.assertAlmostEqual(spec.height, 35.0)

        data = text_to_deepcad_json("Цилиндр диаметром 20 мм высотой 35 мм")
        self.assertEqual(data["sequence"][0]["type"], "ExtrudeFeature")
        ext = data["entities"]["extrude_1"]
        self.assertEqual(ext["operation"], "NewBodyFeatureOperation")
        self.assertEqual(ext["extent_one"]["distance"]["value"], 35.0)

    def test_tube_russian(self):
        spec = parse_text_description(
            "Труба: внешний диаметр 30 мм, внутренний диаметр 20 мм, длина 50 мм"
        )
        self.assertEqual(spec.primitive, "tube")
        self.assertAlmostEqual(spec.radius, 15.0)
        self.assertAlmostEqual(spec.inner_radius, 10.0)
        self.assertAlmostEqual(spec.height, 50.0)

        data = text_to_deepcad_json(
            "Труба: внешний диаметр 30 мм, внутренний диаметр 20 мм, длина 50 мм"
        )
        loops = data["entities"]["sk_1"]["profiles"]["p_1"]["loops"]
        self.assertEqual(len(loops), 2)
        self.assertTrue(loops[0]["is_outer"])
        self.assertFalse(loops[1]["is_outer"])

    def test_plate_with_hole(self):
        spec = parse_text_description("Пластина 60x40x5 мм с отверстием диаметром 10 мм")
        self.assertEqual(spec.primitive, "block")
        self.assertEqual((spec.length, spec.width, spec.height), (60.0, 40.0, 5.0))
        self.assertAlmostEqual(spec.hole_radius, 5.0)

        data = text_to_deepcad_json("Пластина 60x40x5 мм с отверстием диаметром 10 мм")
        loops = data["entities"]["sk_1"]["profiles"]["p_1"]["loops"]
        self.assertEqual(len(loops), 2)
        self.assertEqual(len(loops[0]["profile_curves"]), 4)

    def test_english_cylinder(self):
        spec = parse_text_description("Cylinder diameter 24 mm height 12 mm")
        self.assertEqual(spec.primitive, "cylinder")
        self.assertEqual(spec.radius, 12.0)
        self.assertEqual(spec.height, 12.0)

    def test_invalid_tube(self):
        with self.assertRaises(TextCADParseError):
            parse_text_description("Труба внешний диаметр 20 внутренний диаметр 30 длина 50")

    def test_missing_dimensions(self):
        with self.assertRaises(TextCADParseError):
            parse_text_description("Сделай красивую деталь")


if __name__ == "__main__":
    unittest.main()
