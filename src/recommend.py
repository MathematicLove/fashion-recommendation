"""Assemble outfits by searching the web (DuckDuckGo) for items matching the input.

Images are downloaded and validated server-side so the UI never shows a broken
image placeholder: only pictures that actually decode are handed to the gallery.
"""
from __future__ import annotations
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import httpx
from ddgs import DDGS
from PIL import Image
from . import config
from .analyze import Analysis

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
_HTTP = httpx.Client(
    headers={"User-Agent": _UA, "Referer": "https://duckduckgo.com/", "Accept": "image/*,*/*"},
    timeout=8.0, follow_redirects=True,
)

@dataclass
class Slot:
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

def _slot_noun(role: str, style: str) -> str:
    return config.SLOT_NOUN.get(role, {}).get(style) or config.ROLE_TITLE.get(role, role).lower()

def _slot_color(role: str, style: str, input_color: str) -> str:
    table = config.SLOT_COLOR.get(role)
    if table:
        return table.get(style, next(iter(table.values())))
    if role == config.ROLE_OUTER:
        return input_color if input_color in config.NEUTRAL_COLORS else "black"
    return input_color

def _neutral(color: str) -> str:
    """A coordinating neutral: keep the input colour if it already goes with anything."""
    return color if color in config.NEUTRAL_COLORS else "white"

def _regular_slots(a: Analysis) -> list[Slot]:
    who = _audience(a.gender, a.age)
    banned = _input_noun(a)
    slots: list[Slot] = []
    for role in config.OUTFIT_SLOTS.get(a.role, []):
        noun = _slot_noun(role, a.style)
        if noun == banned:
            continue
        slots.append(Slot(config.ROLE_TITLE[role], noun, _slot_color(role, a.style, a.color), who))
    return slots

def _lingerie_complement(category: str) -> str:
    if "bra" in category:
        return "panties"
    if "panties" in category:
        return "bra"
    return "bra and panties set"


def _intimate_slots(a: Analysis) -> list[Slot]:
    """Underwear pairs only with underwear (a matching set / complementary piece)
    and, at most, socks or a light top - never a full outerwear outfit."""
    color = a.color
    banned = _input_noun(a)

    if a.role == config.ROLE_SOCKS:
        if a.gender == "Men":
            slots = [
                Slot("Underwear", "boxer briefs", color, "mens"),
                Slot("Underwear", "briefs", color, "mens"),
                Slot("Socks", "dress socks", _neutral(color), "mens"),
            ]
        elif a.gender == "Women":
            slots = [
                Slot("Underwear", "panties", color, "womens"),
                Slot("Underwear", "lingerie set", color, "womens"),
                Slot("Socks", "ankle socks", _neutral(color), "womens"),
            ]
        else:
            slots = [
                Slot("Underwear", "briefs", color, ""),
                Slot("Underwear", "underwear set", color, ""),
            ]
    elif a.gender == "Men":  # men's underwear -> matching socks, then a matching set
        slots = [
            Slot("Socks", "socks", _neutral(color), "mens"),
            Slot("Underwear", "boxer briefs", color, "mens"),
            Slot("Top", "undershirt", _neutral(color), "mens"),
        ]
    else:  # women's underwear -> matching set / complementary piece, socks, or a top
        slots = [
            Slot("Underwear", _lingerie_complement(a.category), color, "womens"),
            Slot("Underwear", "lingerie set", color, "womens"),
            Slot("Socks", "socks", _neutral(color), "womens"),
            Slot("Top", "camisole top", _neutral(color), "womens"),
        ]
    return [s for s in slots if s.noun != banned]

def _search_urls(query: str, count: int) -> list[str]:
    try:
        res = DDGS().images(query, safesearch="on", max_results=count)
    except Exception:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for r in res:
        u = r.get("image")
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _download(url: str) -> Image.Image | None:
    """Fetch a URL and return a decoded RGB image, or None if it is not a usable image."""
    try:
        r = _HTTP.get(url)
        if r.status_code != 200 or not r.content:
            return None
        ct = r.headers.get("content-type", "").lower()
        if "image" not in ct and not url.lower().split("?")[0].endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".gif")):
            return None
        img = Image.open(io.BytesIO(r.content))
        img.load()
        if img.width < 100 or img.height < 100:
            return None
        return img.convert("RGB")
    except Exception:
        return None

def _slot_images(slot: Slot, need: int) -> list[Image.Image]:
    """Return up to `need` distinct, verified images matching a slot."""
    base = " ".join(x for x in (slot.audience, slot.color, slot.noun) if x)
    urls = _search_urls(f"{base} product photo", need * 5) or _search_urls(base, need * 5)
    if len(urls) < need * 2:  # widen the search if the coloured query was thin
        urls += _search_urls(" ".join(x for x in (slot.audience, slot.noun) if x), need * 5)

    seen: set[str] = set()
    candidates: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            candidates.append(u)
    candidates = candidates[:need * 6]

    images: list[Image.Image] = []
    with ThreadPoolExecutor(max_workers=min(len(candidates), 10) or 1) as pool:
        for img in pool.map(_download, candidates):
            if img is not None:
                images.append(img)
                if len(images) >= need:
                    break
    return images

def recommend_outfits(analysis: Analysis, n_outfits: int = 4) -> list[list]:
    if analysis.role in config.INTIMATE_ROLES:
        slots = _intimate_slots(analysis)
    else:
        slots = _regular_slots(analysis)
    if not slots:
        raise RuntimeError("Nothing to pair with this item.")

    with ThreadPoolExecutor(max_workers=max(len(slots), 1)) as pool:
        fetched = list(pool.map(lambda s: (s, _slot_images(s, n_outfits)), slots))
    fetched = [(s, imgs) for s, imgs in fetched if imgs]
    if not fetched:
        raise RuntimeError("Could not retrieve images from the web. Check the network connection.")

    outfits: list[list] = []
    for i in range(n_outfits):
        row = [(None, "your item")]
        for slot, imgs in fetched:
            row.append((imgs[i % len(imgs)], f"{slot.title}: {slot.color} {slot.noun}"))
        outfits.append(row)
    return outfits