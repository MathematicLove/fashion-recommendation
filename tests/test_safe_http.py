"""SSRF protection for image downloads. No network: DNS and HTTP are faked."""
import socket

import httpx
import pytest

from src import safe_http
from src.safe_http import UnsafeURL, check_url, fetch

PUBLIC = "93.184.216.34"


def resolver(mapping):
    """Fake getaddrinfo: host -> list of IPs (default: one public address)."""
    def resolve(host, port, type=0):
        if host in mapping and mapping[host] is None:
            raise socket.gaierror("no such host")
        ips = mapping.get(host, [PUBLIC])
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET,
                 socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]
    return resolve


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestCheckUrl:
    def test_public_http_and_https_are_allowed(self):
        check_url("http://example.com/a.jpg", resolver({}))
        check_url("https://example.com:8443/a.jpg", resolver({}))

    @pytest.mark.parametrize("ip", [
        "127.0.0.1", "127.1.2.3", "10.0.0.5", "172.16.0.1", "192.168.1.1",
        "169.254.169.254",      # cloud metadata endpoint
        "100.64.0.1",           # carrier-grade NAT
        "0.0.0.0", "224.0.0.1", "::1", "fe80::1", "fc00::1",
        "::ffff:127.0.0.1",     # IPv4-mapped loopback
        "::ffff:10.0.0.1",
    ])
    def test_non_public_addresses_are_refused(self, ip):
        with pytest.raises(UnsafeURL):
            check_url("http://evil.example/x.jpg", resolver({"evil.example": [ip]}))

    def test_one_private_record_among_public_ones_is_refused(self):
        with pytest.raises(UnsafeURL):
            check_url("http://mixed.example/", resolver({"mixed.example": [PUBLIC, "10.0.0.1"]}))

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd", "ftp://example.com/a.jpg", "gopher://example.com/",
        "javascript:alert(1)", "//example.com/a.jpg", "example.com/a.jpg", "",
    ])
    def test_non_http_schemes_are_refused(self, url):
        with pytest.raises(UnsafeURL):
            check_url(url, resolver({}))

    def test_credentials_in_url_are_refused(self):
        with pytest.raises(UnsafeURL):
            check_url("http://user:pw@example.com/", resolver({}))

    def test_literal_private_ip_is_refused(self):
        with pytest.raises(UnsafeURL):
            check_url("http://127.0.0.1:8080/admin", resolver({"127.0.0.1": ["127.0.0.1"]}))

    def test_unresolvable_host_is_refused(self):
        with pytest.raises(UnsafeURL):
            check_url("http://nope.invalid/", resolver({"nope.invalid": None}))


class TestFetch:
    def test_returns_body_and_content_type(self):
        c = client(lambda r: httpx.Response(200, content=b"imgdata",
                                            headers={"content-type": "image/png"}))
        assert fetch(c, "http://example.com/a.png", resolve=resolver({})) == (b"imgdata", "image/png")

    def test_private_target_is_never_requested(self):
        calls = []

        def handler(request):
            calls.append(request.url)
            return httpx.Response(200, content=b"secret")

        res = fetch(client(handler), "http://internal.example/",
                    resolve=resolver({"internal.example": ["10.0.0.1"]}))
        assert res is None and calls == []

    def test_redirect_to_public_host_is_followed(self):
        def handler(request):
            if request.url.host == "a.example":
                return httpx.Response(302, headers={"location": "http://b.example/img.jpg"})
            return httpx.Response(200, content=b"ok", headers={"content-type": "image/jpeg"})

        res = fetch(client(handler), "http://a.example/", resolve=resolver({}))
        assert res == (b"ok", "image/jpeg")

    def test_redirect_to_private_host_is_blocked(self):
        seen = []

        def handler(request):
            seen.append(request.url.host)
            if request.url.host == "a.example":
                return httpx.Response(302, headers={"location": "http://metadata.internal/latest"})
            return httpx.Response(200, content=b"secret")

        res = fetch(client(handler), "http://a.example/",
                    resolve=resolver({"metadata.internal": ["169.254.169.254"]}))
        assert res is None
        assert seen == ["a.example"]

    def test_relative_redirect_is_followed(self):
        def handler(request):
            if request.url.path == "/old":
                return httpx.Response(301, headers={"location": "/new.jpg"})
            return httpx.Response(200, content=b"ok")

        assert fetch(client(handler), "http://a.example/old", resolve=resolver({}))[0] == b"ok"

    def test_redirect_loop_is_cut_off(self):
        c = client(lambda r: httpx.Response(302, headers={"location": str(r.url)}))
        assert fetch(c, "http://a.example/", max_redirects=3, resolve=resolver({})) is None

    def test_redirect_without_location_returns_none(self):
        c = client(lambda r: httpx.Response(302))
        assert fetch(c, "http://a.example/", resolve=resolver({})) is None

    def test_large_content_length_is_refused_early(self):
        c = client(lambda r: httpx.Response(200, content=b"x" * 10, headers={"content-length": "999999999"}))
        assert fetch(c, "http://a.example/", max_bytes=100, resolve=resolver({})) is None

    def test_body_over_the_cap_is_refused_without_content_length(self):
        c = client(lambda r: httpx.Response(200, content=b"x" * 5000))
        assert fetch(c, "http://a.example/", max_bytes=1000, resolve=resolver({})) is None

    def test_body_exactly_at_the_cap_is_accepted(self):
        c = client(lambda r: httpx.Response(200, content=b"x" * 1000))
        assert len(fetch(c, "http://a.example/", max_bytes=1000, resolve=resolver({}))[0]) == 1000

    @pytest.mark.parametrize("status", [404, 403, 500])
    def test_error_status_returns_none(self, status):
        c = client(lambda r: httpx.Response(status, content=b"x"))
        assert fetch(c, "http://a.example/", resolve=resolver({})) is None

    def test_empty_body_returns_none(self):
        c = client(lambda r: httpx.Response(200, content=b""))
        assert fetch(c, "http://a.example/", resolve=resolver({})) is None

    def test_transport_error_returns_none(self):
        def handler(request):
            raise httpx.ConnectError("boom")

        assert fetch(client(handler), "http://a.example/", resolve=resolver({})) is None

    def test_refusals_are_logged(self, caplog):
        with caplog.at_level("WARNING", logger=safe_http.__name__):
            fetch(client(lambda r: httpx.Response(200)), "http://x.example/",
                  resolve=resolver({"x.example": ["127.0.0.1"]}))
        assert "refused" in caplog.text
