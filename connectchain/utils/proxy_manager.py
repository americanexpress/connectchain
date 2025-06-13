# Copyright 2023 American Express Travel Related Services Company, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
"""Proxy Manager Mixin"""
from contextlib import contextmanager
from logging import Logger
from typing import Any, Dict, Generator, Optional

import requests
import requests.adapters
from pydantic import BaseModel

_logger_ = Logger(__name__)


class ProxyConfig(BaseModel):
    """Proxy configuration for HTTP/HTTPS requests."""

    host: str
    port: int


class ProxyManager:
    """Manages proxy configuration for LLM requests."""

    proxy_config: Optional[ProxyConfig]

    def __init__(self, proxy_config: Optional[ProxyConfig]) -> None:
        """Initialize proxy manager with optional proxy configuration."""
        self.proxy_config = proxy_config

    def _build_proxy_settings_(self) -> Dict[str, str]:
        """Build proxy settings dictionary for requests."""
        if not self.proxy_config:
            return {}
        return {
            "http": f"http://{self.proxy_config.host}:{self.proxy_config.port}",
            "https": f"https://{self.proxy_config.host}:{self.proxy_config.port}",
        }

    def _patch_session_proxies_(self) -> Generator[None, None, None]:
        """Patch requests.Session to use proxy configuration."""
        proxy_config = self._build_proxy_settings_()
        vanilla_session = requests.Session.__init__

        def monkeypatch_proxied_session(self: Any, *args: Any, **kwargs: Any) -> None:
            vanilla_session(self, *args, **kwargs)
            self.proxies.update(proxy_config)

        requests.Session.__init__ = monkeypatch_proxied_session  # type: ignore[method-assign]

        try:
            yield
        finally:
            requests.Session.__init__ = vanilla_session  # type: ignore[method-assign]

    @contextmanager
    def configure_proxy_sync(self) -> Generator[None, None, None]:
        """Configure the proxy for synchronous requests."""
        return self._patch_session_proxies_()

    @contextmanager
    def configure_proxy_async(self) -> Generator[None, None, None]:
        """Configure the proxy for asynchronous requests."""
        _logger_.warning("Async proxy support is not thread safe")
        return self._patch_session_proxies_()
