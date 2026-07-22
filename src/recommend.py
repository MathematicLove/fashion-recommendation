"""Assemble outfits from web images (Openverse) matching the recognized garment."""
from __future__ import annotations
import json
import urllib.parse
import urllib.request
from . import config
from .analyze import Analysis

OPENVERSE_ENDPOINT = "https://api.openverse.org/v1/images/"
_TIMEOUT = 12
_USER_AGENT = "visual-fashion-recommendation/1.0"

def _search_images(query: str, count: int) -> list[str]:
    params = urllib.parse.urlencode({"q": query, "page_size": max(count, 1), "mature": "false"})
    req = urllib.request.Request(
        f"{OPENVERSE_ENDPOINT}?{params}",
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    urls = [r.get("thumbnail") or r.get("url") for r in payload.get("results", [])]
    return [u for u in urls if u][:count]

def _fetch_slot(role: str, analysis: Analysis, count: int) -> list[str]:
    noun = config.ROLE_QUERY_NOUN.get(role, role)
    for query in (f"{analysis.color} {noun}".strip(), noun):
        try:
            urls = _search_images(query, count)
        except Exception:
            urls = []
        if urls:
            return urls
    return []

def recommend_outfits(analysis: Analysis, n_outfits: int = 4) -> list[list]:
    slots = config.OUTFIT_SLOTS.get(analysis.role, [])
    per_slot = {r: urls for r in slots if (urls := _fetch_slot(r, analysis, n_outfits))}
    if not per_slot:
        raise RuntimeError("Could not retrieve images from the web. Check the network connection.")
    outfits: list[list] = []
    for i in range(n_outfits):
        row = [(None, "your item")]
        for role in slots:
            urls = per_slot.get(role)
            if not urls:
                continue
            label = f"{config.ROLE_TITLE.get(role, role)}: {analysis.color} {analysis.style}"
            row.append((urls[i % len(urls)], label))
        if len(row) > 1:
            outfits.append(row)
    return outfits