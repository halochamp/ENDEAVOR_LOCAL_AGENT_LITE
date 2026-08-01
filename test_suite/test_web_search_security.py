# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""SSRF policy checks need no network or search engine."""
from __future__ import annotations

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import web_search


def test_fetch_url_rejects_private_and_non_http_hosts():
    assert not web_search._is_safe_fetch_url("http://127.0.0.1:8090/v1/models")
    assert not web_search._is_safe_fetch_url("http://169.254.169.254/latest/meta-data")
    assert not web_search._is_safe_fetch_url("file:///etc/passwd")


def test_fetch_url_accepts_public_resolved_host():
    with mock.patch.object(web_search.socket, "getaddrinfo", return_value=[(None, None, None, None, ("8.8.8.8", 443))]):
        assert web_search._is_safe_fetch_url("https://example.com/path")


def test_fetch_url_rejects_dns_answer_with_private_ip():
    with mock.patch.object(web_search.socket, "getaddrinfo", return_value=[
        (None, None, None, None, ("8.8.8.8", 443)),
        (None, None, None, None, ("10.0.0.1", 443)),
    ]):
        assert not web_search._is_safe_fetch_url("https://example.com/path")
