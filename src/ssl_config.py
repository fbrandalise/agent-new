"""Disable SSL verification globally for all HTTP clients.

Import this module BEFORE any other imports that create HTTP connections
(OpenAI SDK, DeepEval, LangChain, httpx, requests, urllib3).
"""

import os
import ssl

# --- stdlib ssl ---
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"

# --- urllib3 warnings ---
try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# --- httpx: monkey-patch Client and AsyncClient so every instance
#     (including those created internally by the OpenAI SDK and DeepEval)
#     defaults to verify=False ---
try:
    import httpx

    _orig_client_init = httpx.Client.__init__

    def _patched_client_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        _orig_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _patched_client_init  # type: ignore[assignment]

    _orig_async_init = httpx.AsyncClient.__init__

    def _patched_async_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        _orig_async_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore[assignment]
except ImportError:
    pass

# --- requests (used by some deepeval internals) ---
try:
    import requests
    from requests.adapters import HTTPAdapter

    class _NoVerifyAdapter(HTTPAdapter):
        def send(self, request, **kwargs):
            kwargs["verify"] = False
            return super().send(request, **kwargs)

    _orig_session_init = requests.Session.__init__

    def _patched_session_init(self, *args, **kwargs):
        _orig_session_init(self, *args, **kwargs)
        self.verify = False
        self.mount("https://", _NoVerifyAdapter())

    requests.Session.__init__ = _patched_session_init  # type: ignore[assignment]
except ImportError:
    pass
