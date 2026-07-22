"""Input recognition (CLIP zero-shot for images, keyword parse for text)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image
from . import config
from .model import get_encoder

@dataclass
class Analysis:
    role: str
    category: str
    color: str
    style: str
    gender: str | None
    source: str

    def describe(self) -> str:
        role_en = config.ROLE_TITLE.get(self.role, self.role)
        cat = self.category.replace("a ", "").strip()
        parts = [f"type: {cat} ({role_en})", f"color: {self.color}", f"style: {self.style}"]
        if self.gender:
            parts.append(f"gender: {self.gender}")
        return " | ".join(parts)

def _best(sims: np.ndarray, labels: list[str]) -> str:
    return labels[int(sims.argmax())]

def analyze_image(image: Image.Image, img_vec: np.ndarray | None = None) -> Analysis:
    enc = get_encoder()
    if img_vec is None:
        img_vec = enc.encode_images([image])[0]
    cat_labels = list(config.INPUT_CATEGORY_TO_ROLE.keys())
    cat_vecs = enc.encode_texts([f"a photo of {c}" for c in cat_labels])
    category = _best(cat_vecs @ img_vec, cat_labels)
    role = config.INPUT_CATEGORY_TO_ROLE[category]
    color_vecs = enc.encode_texts([f"a photo of {c} colored clothing" for c in config.COLOR_VOCAB])
    color = _best(color_vecs @ img_vec, config.COLOR_VOCAB)
    style_vecs = enc.encode_texts([f"a photo of {s} style clothing" for s in config.STYLE_VOCAB])
    style = _best(style_vecs @ img_vec, config.STYLE_VOCAB)
    gender_vecs = enc.encode_texts([f"a photo of {g}'s clothing" for g in config.GENDER_VOCAB])
    gender = "Men" if _best(gender_vecs @ img_vec, config.GENDER_VOCAB) == "men" else "Women"
    return Analysis(role, category, color, style, gender, "image")

def analyze_text(text: str) -> Analysis:
    t = text.lower()
    role, category = config.ROLE_TOP, "a top"
    for cat, r in config.INPUT_CATEGORY_TO_ROLE.items():
        if cat.replace("a ", "").strip() in t:
            role, category = r, cat
            break
    color = next((c for c in config.COLOR_VOCAB if c in t), "black")
    style = next((s for s in config.STYLE_VOCAB if s in t), "casual")
    return Analysis(role, category, color, style, None, "text")