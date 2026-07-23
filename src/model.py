"""CLIP wrapper: L2-normalized image and text embeddings."""
from __future__ import annotations
from functools import lru_cache
from typing import Iterable
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from . import config

class ClipEncoder:
    def __init__(self, model_name: str = config.MODEL_NAME, device: str | None = None):
        self.device = device or config.DEVICE
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @staticmethod
    def _to_embeds(out) -> torch.Tensor:
        # transformers >=5 returns an output object; <5 returns the tensor directly.
        return out if torch.is_tensor(out) else out["pooler_output"]

    @torch.no_grad()
    def encode_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        images = [im.convert("RGB") for im in images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self._to_embeds(self.model.get_image_features(**inputs))
        return torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().astype("float32")

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        feats = self._to_embeds(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().astype("float32")

@lru_cache(maxsize=1)
def get_encoder() -> ClipEncoder:
    return ClipEncoder()