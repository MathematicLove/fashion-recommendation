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
ROLE_UNDER = "underwear"
ROLE_SOCKS = "socks"

ROLE_TITLE: dict[str, str] = {
    ROLE_TOP: "Top", ROLE_OUTER: "Outerwear", ROLE_BOTTOM: "Bottom",
    ROLE_DRESS: "Dress", ROLE_FOOT: "Footwear", ROLE_ACC: "Accessory",
    ROLE_UNDER: "Underwear", ROLE_SOCKS: "Socks",
}

INTIMATE_ROLES = {ROLE_UNDER, ROLE_SOCKS}

OUTFIT_SLOTS: dict[str, list[str]] = {
    ROLE_TOP:    [ROLE_BOTTOM, ROLE_FOOT, ROLE_OUTER, ROLE_ACC],
    ROLE_BOTTOM: [ROLE_TOP, ROLE_FOOT, ROLE_OUTER, ROLE_ACC],
    ROLE_OUTER:  [ROLE_TOP, ROLE_BOTTOM, ROLE_FOOT, ROLE_ACC],
    ROLE_DRESS:  [ROLE_FOOT, ROLE_OUTER, ROLE_ACC],
    ROLE_FOOT:   [ROLE_TOP, ROLE_BOTTOM, ROLE_OUTER, ROLE_ACC],
    ROLE_ACC:    [ROLE_TOP, ROLE_BOTTOM, ROLE_FOOT, ROLE_OUTER],
}

SLOT_NOUN: dict[str, dict[str, str]] = {
    ROLE_TOP:    {"casual": "t-shirt", "formal": "dress shirt", "sports": "sports t-shirt", "ethnic": "shirt"},
    ROLE_BOTTOM: {"casual": "jeans", "formal": "trousers", "sports": "joggers", "ethnic": "trousers"},
    ROLE_OUTER:  {"casual": "jacket", "formal": "blazer", "sports": "track jacket", "ethnic": "jacket"},
    ROLE_FOOT:   {"casual": "sneakers", "formal": "formal shoes", "sports": "running shoes", "ethnic": "loafers"},
    ROLE_ACC:    {"casual": "watch", "formal": "watch", "sports": "watch", "ethnic": "watch"},
    ROLE_DRESS:  {"casual": "dress", "formal": "evening dress", "sports": "dress", "ethnic": "ethnic dress"},
}

NEUTRAL_COLORS = {"black", "white", "grey", "navy blue", "beige", "brown", "cream", "silver", "khaki"}

SLOT_COLOR: dict[str, dict[str, str]] = {
    ROLE_BOTTOM: {"casual": "blue", "formal": "navy blue", "sports": "grey", "ethnic": "beige"},
    ROLE_FOOT:   {"casual": "white", "formal": "black", "sports": "white", "ethnic": "brown"},
    ROLE_ACC:    {"casual": "silver", "formal": "silver", "sports": "black", "ethnic": "gold"},
    ROLE_TOP:    {"casual": "white", "formal": "white", "sports": "white", "ethnic": "cream"},
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
    "lingerie": ROLE_UNDER, "a bra": ROLE_UNDER, "panties": ROLE_UNDER,
    "boxers": ROLE_UNDER, "briefs": ROLE_UNDER, "underwear": ROLE_UNDER,
    "socks": ROLE_SOCKS,
}

TEXT_KEYWORDS: list[tuple[str, str]] = [
    ("lingerie", "lingerie"), ("bra", "a bra"), ("panties", "panties"),
    ("knickers", "panties"), ("thong", "panties"),
    ("boxer briefs", "boxers"), ("boxers", "boxers"), ("boxer", "boxers"),
    ("briefs", "briefs"), ("underwear", "underwear"), ("underpants", "underwear"),
    ("socks", "socks"), ("sock", "socks"),
]

UNDERWEAR_GENDER: dict[str, str] = {
    "lingerie": "Women", "a bra": "Women", "panties": "Women",
    "boxers": "Men", "briefs": "Men",
}

COLOR_VOCAB = [
    "black", "white", "grey", "navy blue", "blue", "brown", "beige", "red",
    "green", "pink", "purple", "yellow", "orange", "maroon", "olive", "teal",
    "khaki", "cream", "silver", "gold",
]

STYLE_VOCAB = ["casual", "formal", "sports", "ethnic"]
GENDER_VOCAB = ["men", "women"]
AGE_VOCAB = ["adult", "kids"]