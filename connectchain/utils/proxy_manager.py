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
import os
from abc import ABC
from contextlib import contextmanager
from pydantic import BaseModel
from logging import Logger

_logger_ = Logger(__name__)

class ProxyConfig(BaseModel):
    """Proxy configuration"""
    host: str
    port: int

class ProxyManager(ABC):
    proxy_config: ProxyConfig

    def _build_proxy_settings_(self):
        return {
            'http': f'http://{self.proxy_config.host}:{self.proxy_config.port}',
            'https': f'https://{self.proxy_config.host}:{self.proxy_config.port}'
        }
    
    @contextmanager
    def configure_proxy_sync(self):
        original_proxy = requests.getproxies()
        requests.setproxies(self._build_proxy_settings_())
        try:
            yield
        finally:
            requests.setproxies(original_proxy)

    @contextmanager
    def configure_proxy_async(self):
        """Configure the proxy for `aiohttp` [@see https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support]"""
        _logger_.warn('Async proxy support is not thread safe')
        original_proxy = os.environ.get('HTTP_PROXY')
        original_proxy_https = os.environ.get('HTTPS_PROXY')
        proxy_config = self._build_proxy_settings()
        http_proxy = proxy_config['http']
        https_proxy = proxy_config['https']
        set_http_proxy = False
        set_https_proxy = False
        if http_proxy is not None:
            os.environ.setdefault('HTTP_PROXY', http_proxy)
            set_http_proxy = True
        if https_proxy is not None:
            os.environ.setdefault('HTTPS_PROXY', https_proxy)
            set_https_proxy = True
        os.environ['HTTP_PROXY'] = f'http://{self.proxy_config.host}:{self.proxy_config.port}'
        os.environ['HTTPS_PROXY'] = f'https://{self.proxy_config.host}:{self.proxy_config.port}'
        try:
            yield
        finally:
            if set_http_proxy:
                del os.environ['HTTP_PROXY']
            if set_https_proxy:
                del os.environ['HTTPS_PROXY']
            if original_proxy is not None:
                os.environ.setdefault('HTTP_PROXY', original_proxy)
            if original_proxy_https is not None:
                os.environ.setdefault('HTTPS_PROXY', original_proxy_https)
