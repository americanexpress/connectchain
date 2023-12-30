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
import requests
import requests.adapters
from contextlib import contextmanager
from pydantic import BaseModel
from logging import Logger

_logger_ = Logger(__name__)

class ProxyConfig(BaseModel):
    """Proxy configuration"""
    host: str
    port: int

class ProxyManager():
    proxy_config: ProxyConfig

    def __init__(self, proxy_config: ProxyConfig | None) -> None:
        self.proxy_config = proxy_config

    def _build_proxy_settings_(self):
        return {
            'http': f'http://{self.proxy_config.host}:{self.proxy_config.port}',
            'https': f'https://{self.proxy_config.host}:{self.proxy_config.port}'
        }
    
    def _patch_session_proxies_(self):
        proxy_config = self._build_proxy_settings_()
        vanilla_session = requests.Session.__init__

        def monkeypatch_proxied_session(self, *args, **kwargs):
            vanilla_session(self, *args, **kwargs)
            self.proxies.update(proxy_config)

        requests.Session.__init__ = monkeypatch_proxied_session

        try:
            yield
        finally:
            requests.Session.__init__ = vanilla_session

    @contextmanager
    def configure_proxy_sync(self):
        """Configure the proxy for `aiohttp` [@see https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support]"""
        return self._patch_session_proxies_()

    @contextmanager
    def configure_proxy_async(self):
        """Configure the proxy for `aiohttp` [@see https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support]"""
        _logger_.warn('Async proxy support is not thread safe')
        return self._patch_session_proxies_()
