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

SLOT_NOUN_OPTIONS: dict[str, dict[str, list[str]]] = {
    ROLE_TOP: {
        "casual": ["t-shirt", "shirt", "sweatshirt", "polo shirt", "knit sweater"],
        "formal": ["dress shirt", "button-up shirt", "fine knit sweater", "turtleneck"],
        "sports": ["sports t-shirt", "training top", "performance tank top", "track top"],
        "ethnic": ["kurta shirt", "embroidered shirt", "linen shirt", "tunic"],
    },
    ROLE_BOTTOM: {
        "casual": ["jeans", "chino trousers", "cargo trousers", "corduroy trousers", "shorts"],
        "formal": ["dress trousers", "tailored trousers", "pleated trousers", "wool trousers"],
        "sports": ["joggers", "track pants", "training shorts", "sweatpants"],
        "ethnic": ["linen trousers", "wide leg trousers", "embroidered trousers", "palazzo trousers"],
    },
    ROLE_OUTER: {
        "casual": ["denim jacket", "bomber jacket", "leather jacket", "hoodie", "overshirt"],
        "formal": ["blazer", "wool coat", "trench coat", "tailored jacket"],
        "sports": ["track jacket", "windbreaker jacket", "puffer jacket", "zip hoodie"],
        "ethnic": ["embroidered jacket", "linen jacket", "waistcoat", "long vest"],
    },
    ROLE_FOOT: {
        "casual": ["sneakers", "leather boots", "loafers", "canvas shoes", "chelsea boots"],
        "formal": ["oxford shoes", "derby shoes", "leather loafers", "monk strap shoes"],
        "sports": ["running shoes", "training sneakers", "basketball shoes", "trail shoes"],
        "ethnic": ["leather sandals", "embroidered loafers", "woven sandals", "mojari shoes"],
    },
    ROLE_ACC: {
        "casual": ["wrist watch", "leather belt", "sunglasses", "backpack", "baseball cap"],
        "formal": ["wrist watch", "leather belt", "leather briefcase", "silk tie", "cufflinks"],
        "sports": ["sports watch", "gym backpack", "sports cap", "sweatband", "water bottle"],
        "ethnic": ["beaded necklace", "leather sandal bag", "embroidered scarf", "bangle bracelet"],
    },
    ROLE_DRESS: {
        "casual": ["day dress", "shirt dress", "sundress", "knit dress"],
        "formal": ["evening dress", "cocktail dress", "gown"],
        "sports": ["athletic dress", "tennis dress"],
        "ethnic": ["embroidered dress", "kaftan dress", "anarkali dress"],
    },
}

WOMENS_NOUN_OPTIONS: dict[str, dict[str, list[str]]] = {
    ROLE_TOP: {
        "casual": ["blouse", "t-shirt", "knit top", "cropped sweatshirt", "cardigan"],
        "formal": ["silk blouse", "dress blouse", "fine knit top", "turtleneck top"],
        "sports": ["sports top", "training tank top", "performance t-shirt"],
        "ethnic": ["embroidered blouse", "tunic top", "kurti"],
    },
    ROLE_BOTTOM: {
        "casual": ["jeans", "denim skirt", "chino trousers", "midi skirt", "shorts"],
        "formal": ["tailored trousers", "pencil skirt", "midi skirt", "wide leg trousers"],
        "sports": ["leggings", "joggers", "training shorts", "yoga pants"],
        "ethnic": ["palazzo trousers", "long skirt", "embroidered skirt", "wide leg trousers"],
    },
    ROLE_FOOT: {
        "casual": ["sneakers", "ankle boots", "ballet flats", "loafers", "sandals"],
        "formal": ["heeled pumps", "block heel shoes", "pointed flats", "heeled sandals"],
        "sports": ["running shoes", "training sneakers", "trail shoes"],
        "ethnic": ["embroidered flats", "leather sandals", "beaded sandals"],
    },
    ROLE_ACC: {
        "casual": ["handbag", "wrist watch", "sunglasses", "tote bag", "silk scarf"],
        "formal": ["clutch bag", "wrist watch", "pearl necklace", "leather handbag"],
        "sports": ["sports watch", "gym backpack", "sports cap", "crossbody bag"],
        "ethnic": ["beaded necklace", "embroidered clutch", "bangle bracelet", "embroidered scarf"],
    },
}

NEUTRAL_COLORS = {"black", "white", "grey", "navy blue", "beige", "brown", "cream", "silver", "khaki", "gold"}

WARM_COLORS = {"red", "orange", "yellow", "brown", "maroon", "olive", "khaki", "gold", "beige", "cream"}
COOL_COLORS = {"blue", "navy blue", "green", "purple", "teal", "pink"}

SLOT_COLOR_OPTIONS: dict[str, dict[str, list[str]]] = {
    ROLE_BOTTOM: {
        "casual": ["blue", "black", "grey", "beige", "khaki", "navy blue"],
        "formal": ["navy blue", "black", "grey", "brown"],
        "sports": ["grey", "black", "navy blue", "olive"],
        "ethnic": ["beige", "cream", "white", "gold"],
    },
    ROLE_FOOT: {
        "casual": ["white", "black", "grey", "brown"],
        "formal": ["black", "brown", "grey"],
        "sports": ["white", "black", "grey"],
        "ethnic": ["brown", "gold", "beige"],
    },
    ROLE_ACC: {
        "casual": ["silver", "black", "brown", "gold"],
        "formal": ["silver", "gold", "black"],
        "sports": ["black", "grey", "silver"],
        "ethnic": ["gold", "silver", "brown"],
    },
    ROLE_TOP: {
        "casual": ["white", "black", "grey", "beige", "navy blue"],
        "formal": ["white", "black", "grey", "cream"],
        "sports": ["white", "black", "grey"],
        "ethnic": ["cream", "white", "gold", "beige"],
    },
    ROLE_OUTER: {
        "casual": ["black", "navy blue", "grey", "brown", "khaki"],
        "formal": ["black", "navy blue", "grey"],
        "sports": ["black", "grey", "navy blue"],
        "ethnic": ["brown", "gold", "beige"],
    },
}

ACC_COLOR_BY_KEYWORD: list[tuple[str, list[str]]] = [
    ("watch", ["silver", "black", "gold", "brown"]),
    ("necklace", ["gold", "silver"]),
    ("bracelet", ["gold", "silver"]),
    ("bangle", ["gold", "silver"]),
    ("cufflinks", ["silver", "gold"]),
    ("sunglasses", ["black", "brown", "grey"]),
    ("belt", ["black", "brown", "beige"]),
    ("briefcase", ["brown", "black"]),
    ("backpack", ["black", "grey", "navy blue", "khaki"]),
    ("bag", ["black", "brown", "beige", "navy blue"]),
    ("clutch", ["black", "gold", "beige"]),
    ("tie", ["navy blue", "black", "maroon", "grey"]),
    ("scarf", ["cream", "beige", "navy blue", "maroon"]),
    ("cap", ["black", "navy blue", "white", "grey"]),
    ("sweatband", ["black", "white", "grey"]),
    ("bottle", ["black", "white", "silver"]),
]

ROLE_DESCRIPTION: dict[str, str] = {
    ROLE_TOP: "a shirt or t-shirt",
    ROLE_BOTTOM: "trousers, jeans or a skirt",
    ROLE_OUTER: "a jacket or a coat",
    ROLE_FOOT: "a pair of shoes",
    ROLE_ACC: "a fashion accessory",
    ROLE_DRESS: "a dress",
    ROLE_UNDER: "underwear",
    ROLE_SOCKS: "a pair of socks",
}

NONFASHION_DISTRACTORS = [
    "a cat", "a dog", "a flower", "a bouquet of flowers", "a houseplant",
    "a landscape photograph", "a plate of food", "a car", "a building",
    "a room interior", "a piece of furniture", "a close-up of a human face",
    "a text banner with words", "a company logo", "an abstract pattern",
    "a smartphone", "a chart or infographic", "a collage of many products",
    "a cosmetics bottle", "a toy",
]

VERIFY_MIN_PROB = 0.30
VERIFY_MIN_SIM = 0.20

NONFASHION_OBJECTS = [
    "an electronic device", "a computer keyboard", "a household appliance",
    "a piece of furniture", "a bottle or a cup", "a food dish", "an animal",
    "a plant", "a cosmetics product", "a book or a document", "a vehicle",
    "a building", "a bare human face",
]

CLOTHING_MIN_PROB = 0.50

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

CATEGORY_GENDER: dict[str, str] = {
    "a dress": "Women", "a skirt": "Women", "a blouse": "Women",
    "heels": "Women", "a jumpsuit": "Women", "leggings": "Women",
}

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