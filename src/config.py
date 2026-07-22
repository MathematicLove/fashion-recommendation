"""Configuration and vocabularies for input recognition and outfit slots."""
from __future__ import annotations
import os
import torch

MODEL_NAME = os.environ.get("VFR_MODEL", "openai/clip-vit-base-patch32")

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

ROLE_TOP = "top"
ROLE_OUTER = "outerwear"
ROLE_BOTTOM = "bottom"
ROLE_DRESS = "dress"
ROLE_FOOT = "footwear"
ROLE_ACC = "accessory"

ROLE_TITLE: dict[str, str] = {
    ROLE_TOP: "Top", ROLE_OUTER: "Outerwear", ROLE_BOTTOM: "Bottom",
    ROLE_DRESS: "Dress", ROLE_FOOT: "Footwear", ROLE_ACC: "Accessory",
}

ROLE_QUERY_NOUN: dict[str, str] = {
    ROLE_TOP: "shirt", ROLE_OUTER: "jacket", ROLE_BOTTOM: "trousers",
    ROLE_DRESS: "dress", ROLE_FOOT: "shoes", ROLE_ACC: "bag",
}

OUTFIT_SLOTS: dict[str, list[str]] = {
    ROLE_TOP:    [ROLE_BOTTOM, ROLE_FOOT, ROLE_OUTER, ROLE_ACC],
    ROLE_BOTTOM: [ROLE_TOP, ROLE_FOOT, ROLE_OUTER, ROLE_ACC],
    ROLE_OUTER:  [ROLE_TOP, ROLE_BOTTOM, ROLE_FOOT, ROLE_ACC],
    ROLE_DRESS:  [ROLE_FOOT, ROLE_OUTER, ROLE_ACC],
    ROLE_FOOT:   [ROLE_TOP, ROLE_BOTTOM, ROLE_OUTER, ROLE_ACC],
    ROLE_ACC:    [ROLE_TOP, ROLE_BOTTOM, ROLE_FOOT, ROLE_OUTER],
}

INPUT_CATEGORY_TO_ROLE: dict[str, str] = {
    "a t-shirt": ROLE_TOP, "a shirt": ROLE_TOP, "a blouse": ROLE_TOP,
    "a top": ROLE_TOP, "a polo shirt": ROLE_TOP,
    "a sweater": ROLE_OUTER, "a hoodie": ROLE_OUTER, "a jacket": ROLE_OUTER,
    "a blazer": ROLE_OUTER, "a coat": ROLE_OUTER,
    "jeans": ROLE_BOTTOM, "trousers": ROLE_BOTTOM, "shorts": ROLE_BOTTOM,
    "a skirt": ROLE_BOTTOM, "leggings": ROLE_BOTTOM,
    "a dress": ROLE_DRESS, "a jumpsuit": ROLE_DRESS,
    "sneakers": ROLE_FOOT, "formal shoes": ROLE_FOOT, "heels": ROLE_FOOT,
    "sandals": ROLE_FOOT, "boots": ROLE_FOOT,
    "a watch": ROLE_ACC, "a handbag": ROLE_ACC, "sunglasses": ROLE_ACC,
    "a belt": ROLE_ACC, "a cap": ROLE_ACC, "a backpack": ROLE_ACC,
}

COLOR_VOCAB = [
    "black", "white", "grey", "navy blue", "blue", "brown", "beige", "red",
    "green", "pink", "purple", "yellow", "orange", "maroon", "olive", "teal",
    "khaki", "cream", "silver", "gold",
]

STYLE_VOCAB = ["casual", "formal", "sports", "ethnic"]

GENDER_VOCAB = ["men", "women"]