import unittest

try:
    import torch
    from text2cad_ml import TextVocabulary, TextLatentEncoder
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch runtime is not installed")
class Text2CADMLTests(unittest.TestCase):
    def test_dimensions_are_compositional_tokens(self):
        tokens = TextVocabulary.tokenize("Пластина 60x40x5 мм, отверстие -12.5 мм")
        self.assertIn("<num>", tokens)
        self.assertIn("6", tokens)
        self.assertIn("0", tokens)
        self.assertIn("x", tokens)
        self.assertIn("-", tokens)
        self.assertIn(".", tokens)
        self.assertNotIn("60x40x5", tokens)

    def test_unseen_numeric_combination_reuses_known_digits(self):
        vocab = TextVocabulary.build(["размер 20 мм", "размер 15 мм"])
        encoded = vocab.encode("размер 25 мм", 16)
        # 25 was never present as a whole token; with digit tokenization it is
        # nevertheless representable without <unk>.
        non_pad = [x for x in encoded if x != vocab.stoi[vocab.PAD]]
        self.assertNotIn(vocab.stoi[vocab.UNK], non_pad)

    def test_encoder_output_matches_deepcad_latent_dimension(self):
        vocab = TextVocabulary.build(["цилиндр диаметром 20 мм высотой 35 мм"])
        model = TextLatentEncoder(
            vocab_size=len(vocab),
            dim_z=256,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            max_len=32,
        )
        token_ids = torch.tensor(
            [vocab.encode("цилиндр диаметром 20 мм высотой 35 мм", 32)],
            dtype=torch.long,
        )
        out = model(token_ids)
        self.assertEqual(tuple(out.shape), (1, 256))


if __name__ == "__main__":
    unittest.main()
