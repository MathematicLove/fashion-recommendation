"""Assemble outfits by searching the web (DuckDuckGo) for items matching the input.

Every downloaded picture is verified with CLIP against the slot it is meant to
fill, so search noise (pets, flowers, banners, collages, wrong garments) never
reaches the gallery. Each outfit is planned separately, so the four results are
different items rather than four photos of the same t-shirt and sneakers.
"""
from __future__ import annotations
import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import httpx
import numpy as np
from ddgs import DDGS
from PIL import Image
from . import config
from .analyze import Analysis
from .model import get_encoder

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
_HTTP = httpx.Client(
    headers={"User-Agent": _UA, "Referer": "https://duckduckgo.com/", "Accept": "image/*,*/*"},
    timeout=8.0, follow_redirects=True,
)

_CANDIDATES_PER_SLOT = 8
_DOWNLOAD_WORKERS = 12
_MAX_SIDE = (448, 448)
_ENCODE_BATCH = 8
_MAX_VARIANT_DEPTH = 4


@dataclass(frozen=True)
class Slot:
    role: str
    title: str
    noun: str
    color: str
    audience: str


def _audience(gender: str | None, age: str) -> str:
    if age == "kids":
        return "kids"
    if gender == "Men":
        return "mens"
    if gender == "Women":
        return "womens"
    return ""


def _input_noun(a: Analysis) -> str:
    return a.category.replace("a ", "").strip()


def _noun_options(role: str, style: str, gender: str | None) -> list[str]:
    if gender == "Women":
        opts = config.WOMENS_NOUN_OPTIONS.get(role, {}).get(style)
        if opts:
            return list(opts)
    opts = config.SLOT_NOUN_OPTIONS.get(role, {}).get(style)
    if opts:
        return list(opts)
    return [config.ROLE_TITLE.get(role, role).lower()]


def _colors_compatible(a: str, b: str) -> bool:
    if a == b or a in config.NEUTRAL_COLORS or b in config.NEUTRAL_COLORS:
        return True
    if a in config.WARM_COLORS and b in config.WARM_COLORS:
        return True
    if a in config.COOL_COLORS and b in config.COOL_COLORS:
        return True
    return False


def _color_options(role: str, style: str, input_color: str, noun: str = "") -> list[str]:
    options = None
    if role == config.ROLE_ACC:
        for kw, colors in config.ACC_COLOR_BY_KEYWORD:
            if kw in noun:
                options = colors
                break
    if options is None:
        options = config.SLOT_COLOR_OPTIONS.get(role, {}).get(style)
    if not options:
        return [input_color]
    ok = [c for c in options if _colors_compatible(input_color, c)]
    return ok or list(options)


def _conflicts(noun: str, banned: str) -> bool:
    return noun == banned or banned in noun or noun in banned


def _regular_plans(a: Analysis, n_outfits: int, start: int = 0) -> list[list[Slot]]:
    who = _audience(a.gender, a.age)
    banned = _input_noun(a)
    plans: list[list[Slot]] = []
    for i in range(start, start + n_outfits):
        row: list[Slot] = []
        for role in config.OUTFIT_SLOTS.get(a.role, []):
            nouns = [n for n in _noun_options(role, a.style, a.gender) if not _conflicts(n, banned)]
            if not nouns:
                continue
            noun = nouns[i % len(nouns)]
            colors = _color_options(role, a.style, a.color, noun)
            row.append(Slot(role, config.ROLE_TITLE[role], noun, colors[i % len(colors)], who))
        plans.append(row)
    return plans


def _neutral(color: str) -> str:
    return color if color in config.NEUTRAL_COLORS else "white"


def _lingerie_complement(category: str) -> str:
    if "bra" in category:
        return "panties"
    if "panties" in category:
        return "bra"
    return "bra and panties set"


def _intimate_slots(a: Analysis) -> list[Slot]:
    color = a.color
    banned = _input_noun(a)
    u, s, t = config.ROLE_UNDER, config.ROLE_SOCKS, config.ROLE_TOP

    if a.role == config.ROLE_SOCKS:
        if a.gender == "Men":
            slots = [
                Slot(u, "Underwear", "boxer briefs", color, "mens"),
                Slot(u, "Underwear", "briefs", color, "mens"),
                Slot(s, "Socks", "dress socks", _neutral(color), "mens"),
            ]
        elif a.gender == "Women":
            slots = [
                Slot(u, "Underwear", "panties", color, "womens"),
                Slot(u, "Underwear", "lingerie set", color, "womens"),
                Slot(s, "Socks", "ankle socks", _neutral(color), "womens"),
            ]
        else:
            slots = [
                Slot(u, "Underwear", "briefs", color, ""),
                Slot(u, "Underwear", "underwear set", color, ""),
            ]
    elif a.gender == "Men":
        slots = [
            Slot(s, "Socks", "socks", _neutral(color), "mens"),
            Slot(u, "Underwear", "boxer briefs", color, "mens"),
            Slot(t, "Top", "undershirt", _neutral(color), "mens"),
        ]
    else:
        slots = [
            Slot(u, "Underwear", _lingerie_complement(a.category), color, "womens"),
            Slot(u, "Underwear", "lingerie set", color, "womens"),
            Slot(s, "Socks", "socks", _neutral(color), "womens"),
            Slot(t, "Top", "camisole top", _neutral(color), "womens"),
        ]
    return [s_ for s_ in slots if not _conflicts(s_.noun, banned)]


def _search_urls(query: str, count: int) -> list[str]:
    try:
        res = DDGS().images(query, safesearch="on", max_results=count)
    except Exception:
        return []
    out: list[str] = []
    for r in res:
        u = r.get("image")
        if u:
            out.append(u)
    return out


def _slot_urls(slot: Slot, count: int, relaxed: bool = False) -> list[str]:
    if relaxed:
        queries = [
            " ".join(x for x in (slot.color, slot.noun, "clothing product") if x),
            " ".join(x for x in (slot.audience, slot.noun) if x),
            f"{slot.noun} product photo",
        ]
    else:
        queries = [
            " ".join(x for x in (slot.audience, slot.color, slot.noun, "product photo") if x),
            " ".join(x for x in (slot.audience, slot.color, slot.noun) if x),
            " ".join(x for x in (slot.audience, slot.noun, "product photo") if x),
        ]
    seen: set[str] = set()
    urls: list[str] = []
    for q in queries:
        for u in _search_urls(q, count):
            if u not in seen:
                seen.add(u)
                urls.append(u)
        if len(urls) >= count:
            break
    return urls[:count]


def _download(url: str) -> Image.Image | None:
    try:
        r = _HTTP.get(url)
        if r.status_code != 200 or not r.content:
            return None
        ct = r.headers.get("content-type", "").lower()
        if "image" not in ct and not url.lower().split("?")[0].endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".gif")):
            return None
        img = Image.open(io.BytesIO(r.content))
        if img.width < 100 or img.height < 100:
            return None
        img.draft("RGB", _MAX_SIDE)
        img.load()
        img = img.convert("RGB")
        img.thumbnail(_MAX_SIDE, Image.LANCZOS)
        return img
    except Exception:
        return None


@lru_cache(maxsize=256)
def _prompt_vectors(role: str, noun: str, color: str) -> tuple[np.ndarray, int]:
    positives = [
        f"a product photo of {color} {noun}",
        f"a photo of a {color} {noun}",
        f"a {noun}",
    ]
    negatives = [f"a photo of {d}" for d in config.NONFASHION_DISTRACTORS]
    negatives += [f"a product photo of {desc}"
                  for r, desc in config.ROLE_DESCRIPTION.items() if r != role]
    vecs = get_encoder().encode_texts(positives + negatives)
    return vecs, len(positives)


def _rank(images: list[Image.Image], img_vecs: np.ndarray, slot: Slot) -> list[int]:
    if not images:
        return []
    txt_vecs, n_pos = _prompt_vectors(slot.role, slot.noun, slot.color)
    sims = img_vecs @ txt_vecs.T
    logits = 100.0 * sims
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    p_pos = probs[:, :n_pos].sum(axis=1)
    s_pos = sims[:, :n_pos].max(axis=1)
    keep = [i for i in range(len(images))
            if p_pos[i] >= config.VERIFY_MIN_PROB and s_pos[i] >= config.VERIFY_MIN_SIM]
    keep.sort(key=lambda i: float(s_pos[i]), reverse=True)
    return keep


def _collect(slots: list[Slot], need: Counter, relaxed: bool = False) -> dict[Slot, list[Image.Image]]:
    with ThreadPoolExecutor(max_workers=min(len(slots), 12) or 1) as pool:
        url_lists = list(pool.map(
            lambda s: _slot_urls(s, max(_CANDIDATES_PER_SLOT, need[s] * 5), relaxed), slots))

    flat: list[tuple[int, str]] = []
    seen: set[str] = set()
    for idx, urls in enumerate(url_lists):
        for u in urls:
            if u not in seen:
                seen.add(u)
                flat.append((idx, u))
    if not flat:
        return {}

    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        downloaded = list(pool.map(lambda item: _download(item[1]), flat))

    per_slot: dict[int, list[Image.Image]] = {i: [] for i in range(len(slots))}
    for (idx, _), img in zip(flat, downloaded):
        if img is not None:
            per_slot[idx].append(img)
    downloaded.clear()
    flat.clear()

    enc = get_encoder()
    out: dict[Slot, list[Image.Image]] = {}
    for idx, slot in enumerate(slots):
        imgs = per_slot.pop(idx, [])
        if not imgs:
            continue
        vecs = np.concatenate([enc.encode_images(imgs[i:i + _ENCODE_BATCH])
                               for i in range(0, len(imgs), _ENCODE_BATCH)], axis=0)
        order = _rank(imgs, vecs, slot)
        if order:
            out[slot] = [imgs[i] for i in order[:max(need[slot], 1)]]
        imgs.clear()
    return out


def recommend_outfit(analysis: Analysis, index: int = 0) -> list:
    """Build a single outfit variant.

    Same planning and verification as ``recommend_outfits``, but only one plan is
    searched for, so a result appears after a quarter of the web requests and CLIP
    passes. ``index`` selects the variant: it walks the noun/color options for
    regular items, and picks a deeper-ranked image for intimates (whose plan does
    not vary).
    """
    if analysis.role in config.INTIMATE_ROLES:
        plan = _intimate_slots(analysis)
        depth = min(index + 1, _MAX_VARIANT_DEPTH)
    else:
        plans = _regular_plans(analysis, 1, start=index)
        plan = plans[0] if plans else []
        depth = 1
    if not plan:
        raise RuntimeError("Nothing to pair with this item.")

    need: Counter = Counter({slot: depth for slot in plan})
    slots = list(need)
    pool = _collect(slots, need)
    missing = [s for s in slots if s not in pool]
    if missing:
        pool.update(_collect(missing, need, relaxed=True))
    if not pool:
        raise RuntimeError("Could not retrieve matching images from the web. "
                           "Check the network connection and try again.")

    out_row = [(None, "your item")]
    for slot in plan:
        imgs = pool.get(slot)
        if not imgs:
            continue
        out_row.append((imgs[index % len(imgs)], f"{slot.title}: {slot.color} {slot.noun}"))
    if len(out_row) == 1:
        raise RuntimeError("Could not assemble an outfit from the web results.")
    return out_row


def recommend_outfits(analysis: Analysis, n_outfits: int = 4) -> list[list]:
    if analysis.role in config.INTIMATE_ROLES:
        base = _intimate_slots(analysis)
        plans = [list(base) for _ in range(n_outfits)]
    else:
        plans = _regular_plans(analysis, n_outfits)
    plans = [p for p in plans if p]
    if not plans:
        raise RuntimeError("Nothing to pair with this item.")

    need: Counter = Counter()
    for row in plans:
        for slot in row:
            need[slot] += 1
    slots = list(need)

    pool = _collect(slots, need)
    missing = [s for s in slots if s not in pool]
    if missing:
        pool.update(_collect(missing, need, relaxed=True))
    if not pool:
        raise RuntimeError("Could not retrieve matching images from the web. "
                           "Check the network connection and try again.")

    used: Counter = Counter()
    outfits: list[list] = []
    for row in plans:
        out_row = [(None, "your item")]
        for slot in row:
            imgs = pool.get(slot)
            if not imgs:
                continue
            img = imgs[used[slot] % len(imgs)]
            used[slot] += 1
            out_row.append((img, f"{slot.title}: {slot.color} {slot.noun}"))
        if len(out_row) > 1:
            outfits.append(out_row)
    if not outfits:
        raise RuntimeError("Could not assemble an outfit from the web results.")
    return outfits
