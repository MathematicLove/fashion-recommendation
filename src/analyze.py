"""Input recognition (CLIP zero-shot for images, keyword parse for text)."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
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
    age: str
    source: str

    def describe(self) -> str:
        role_en = config.ROLE_TITLE.get(self.role, self.role)
        cat = self.category.replace("a ", "").strip()
        parts = [f"type: {cat} ({role_en})", f"color: {self.color}", f"style: {self.style}"]
        if self.gender:
            parts.append(f"gender: {self.gender}")
        parts.append(f"age: {self.age}")
        return " | ".join(parts)

def _best(sims: np.ndarray, labels: list[str]) -> str:
    return labels[int(sims.argmax())]

@lru_cache(maxsize=8)
def _label_vecs(template: str, labels: tuple[str, ...]) -> np.ndarray:
    """Embeddings of a fixed label vocabulary; identical on every request, so cached."""
    return get_encoder().encode_texts([template.format(x) for x in labels])

def is_clothing(img_vec: np.ndarray) -> bool:
    """True when the photo shows one of the garment types the app accepts.

    The positives are the input categories themselves, so this shares the cached
    embeddings with the recognition step; the negatives are everyday non-fashion
    subjects. Photos of pets, food, rooms, devices and faces land far below the
    threshold and are rejected before any web search is started.
    """
    cats = tuple(config.INPUT_CATEGORY_TO_ROLE)
    others = tuple(config.NONFASHION_DISTRACTORS + config.NONFASHION_OBJECTS)
    sims = np.concatenate([_label_vecs("a photo of {}", cats),
                           _label_vecs("a photo of {}", others)]) @ img_vec
    logits = 100.0 * sims
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return float(probs[:len(cats)].sum()) >= config.CLOTHING_MIN_PROB

def analyze_image(image: Image.Image, img_vec: np.ndarray | None = None) -> Analysis:
    if img_vec is None:
        img_vec = get_encoder().encode_images([image])[0]
    cat_labels = list(config.INPUT_CATEGORY_TO_ROLE.keys())
    cat_vecs = _label_vecs("a photo of {}", tuple(cat_labels))
    category = _best(cat_vecs @ img_vec, cat_labels)
    role = config.INPUT_CATEGORY_TO_ROLE[category]
    color_vecs = _label_vecs("a photo of {} colored clothing", tuple(config.COLOR_VOCAB))
    color = _best(color_vecs @ img_vec, config.COLOR_VOCAB)
    style_vecs = _label_vecs("a photo of {} style clothing", tuple(config.STYLE_VOCAB))
    style = _best(style_vecs @ img_vec, config.STYLE_VOCAB)
    gender_vecs = _label_vecs("a photo of {}'s clothing", tuple(config.GENDER_VOCAB))
    gender = "Men" if _best(gender_vecs @ img_vec, config.GENDER_VOCAB) == "men" else "Women"
    age_vecs = _label_vecs("a photo of {} clothing", tuple(config.AGE_VOCAB))
    age = _best(age_vecs @ img_vec, config.AGE_VOCAB)
    return Analysis(role, category, color, style, gender, age, "image")

def analyze_text(text: str) -> Analysis:
    t = f" {text.lower()} "
    category = None
    for kw, cat in config.TEXT_KEYWORDS:
        if kw in t:
            category = cat
            break
    if category is None:
        for cat in config.INPUT_CATEGORY_TO_ROLE:
            if cat.replace("a ", "").strip() in t:
                category = cat
                break
    category = category or "a top"
    role = config.INPUT_CATEGORY_TO_ROLE.get(category, config.ROLE_TOP)
    color = next((c for c in config.COLOR_VOCAB if c in t), "black")
    style = next((s for s in config.STYLE_VOCAB if s in t), "casual")
    gender = "Women" if any(w in t for w in ("women", "woman", "female", "girl", "lady", "ladies")) else (
        "Men" if any(w in t for w in ("men", "man", "male", "boy")) else None)
    if gender is None:
        gender = config.UNDERWEAR_GENDER.get(category) or config.CATEGORY_GENDER.get(category)
    age = "kids" if any(w in t for w in ("kids", "kid", "child", "boy", "girl")) else "adult"
    return Analysis(role, category, color, style, gender, age, "text")