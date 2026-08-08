"""Trainable Text -> DeepCAD latent mapper.

The model does not replace the existing DeepCAD decoder. It learns only the
mapping from tokenized text to the latent code ``z`` of the pretrained DeepCAD
autoencoder. At inference the existing ``TrainerAE.decode`` converts this z to
the standard DeepCAD command vector (Line/Arc/Circle/SOL/Ext/EOS).

Numbers are tokenized compositionally (sign/digits/decimal point) instead of as
whole dimension strings. Thus a dimension not seen verbatim during training can
still be represented using known numeric tokens.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List

import torch
import torch.nn as nn


RAW_TOKEN_RE = re.compile(r"[+-]?\d+(?:[\.,]\d+)?|[a-zа-я]+|×", re.IGNORECASE)


class TextVocabulary:
    PAD = "<pad>"
    UNK = "<unk>"
    NUM = "<num>"

    def __init__(self, stoi: Dict[str, int]):
        self.stoi = dict(stoi)
        self.itos = [None] * len(self.stoi)
        for token, idx in self.stoi.items():
            self.itos[idx] = token

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1):
        counter = Counter()
        for text in texts:
            counter.update(cls.tokenize(text))
        stoi = {cls.PAD: 0, cls.UNK: 1}
        for token, count in sorted(counter.items()):
            if count >= min_freq and token not in stoi:
                stoi[token] = len(stoi)
        return cls(stoi)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower().replace("ё", "е").replace("×", "x").replace("х", "x")
        raw_tokens = RAW_TOKEN_RE.findall(text)
        result = []
        for token in raw_tokens:
            normalized = token.replace(",", ".")
            # Numeric values are represented compositionally. Example:
            # -12.5 -> <num>, -, 1, 2, ., 5
            if re.match(r"^[+-]?\d", normalized):
                result.append(TextVocabulary.NUM)
                result.extend(list(normalized))
            else:
                result.append(normalized)
        return result

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.stoi.get(t, self.stoi[self.UNK]) for t in self.tokenize(text)][:max_len]
        return ids + [self.stoi[self.PAD]] * (max_len - len(ids))

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.stoi, fp, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as fp:
            return cls(json.load(fp))

    def __len__(self):
        return len(self.stoi)


class TextLatentEncoder(nn.Module):
    """Transformer text encoder whose output has DeepCAD latent dimension."""

    def __init__(
        self,
        vocab_size: int,
        dim_z: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dim_z),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch, seq]
        batch, seq = token_ids.shape
        if seq > self.max_len:
            raise ValueError("Text sequence is longer than max_len")
        pos = torch.arange(seq, device=token_ids.device).unsqueeze(0).expand(batch, seq)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        padding_mask = token_ids.eq(self.pad_idx)

        # PyTorch 1.5 TransformerEncoder expects [seq, batch, dim].
        x = self.encoder(x.transpose(0, 1), src_key_padding_mask=padding_mask).transpose(0, 1)
        valid = (~padding_mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.to_latent(self.norm(pooled))


def latent_alignment_loss(pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    """MSE + cosine term used to align text embeddings with DeepCAD latent codes."""
    if target_z.dim() == 3 and target_z.size(1) == 1:
        target_z = target_z[:, 0]
    mse = torch.mean((pred_z - target_z.detach()) ** 2)
    cosine = 1.0 - torch.nn.functional.cosine_similarity(pred_z, target_z.detach(), dim=-1).mean()
    return mse + 0.1 * cosine
