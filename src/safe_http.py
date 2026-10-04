"""Safe downloading of remote images.

Image URLs come from web search results, i.e. from third parties. Fetching them
blindly lets a crafted result point the server at internal addresses (SSRF:
localhost services, the cloud metadata endpoint 169.254.169.254, a LAN router)
or at a huge file. This module only follows http(s) URLs that resolve to public
addresses, re-checks every redirect hop, and caps the size of the body.

Known limit: the hostname is resolved once for the check and again by the HTTP
client when it connects, so a DNS server that answers differently the second
time (DNS rebinding) is not covered. Pinning the connection to the checked IP
would close that gap.
"""
from __future__ import annotations
import ipaddress
import logging
import socket
from urllib.parse import urlsplit
import httpx

log = logging.getLogger(__name__)

MAX_BYTES = 8 * 1024 * 1024
MAX_REDIRECTS = 4


class UnsafeURL(ValueError):
    """The URL must not be fetched."""


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_global and not ip.is_multicast


def check_url(url: str, resolve=socket.getaddrinfo) -> None:
    """Raise UnsafeURL unless `url` is http(s) and every address it resolves to is public."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURL(f"unsupported scheme: {parts.scheme or 'none'}")
    host = parts.hostname
    if not host:
        raise UnsafeURL("no host in URL")
    if parts.username or parts.password:
        raise UnsafeURL("credentials in URL")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
        infos = resolve(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, ValueError, UnicodeError) as exc:
        raise UnsafeURL(f"cannot resolve {host}: {exc}") from exc
    if not infos:
        raise UnsafeURL(f"cannot resolve {host}")
    for info in infos:
        address = info[4][0]
        if not _is_public(address):
            raise UnsafeURL(f"{host} resolves to a non-public address ({address})")


def fetch(client: httpx.Client, url: str, max_bytes: int = MAX_BYTES,
          max_redirects: int = MAX_REDIRECTS, resolve=socket.getaddrinfo):
    """GET `url` safely. Returns (body, content_type), or None if it was refused or failed."""
    try:
        for _ in range(max_redirects + 1):
            check_url(url, resolve)
            with client.stream("GET", url, follow_redirects=False) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        return None
                    url = str(resp.url.join(location))
                    continue
                if resp.status_code != 200:
                    return None
                length = resp.headers.get("content-length", "")
                if length.isdigit() and int(length) > max_bytes:
                    return None
                body = bytearray()
                # iter_bytes() yields decoded data, so a compressed bomb is capped too.
                for chunk in resp.iter_bytes():
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        return None
                if not body:
                    return None
                return bytes(body), resp.headers.get("content-type", "")
        return None  # too many redirects
    except UnsafeURL as exc:
        log.warning("refused to fetch %s: %s", url, exc)
        return None
    except httpx.HTTPError:
        return None
