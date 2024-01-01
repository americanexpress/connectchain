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
"""Unit testing for ProxyManager class"""
import unittest
from unittest.mock import Mock
from connectchain.utils.proxy_manager import ProxyManager, ProxyConfig

class TestProxyManager(unittest.TestCase):
    """Unit testing the ProxyManager class"""

    def test_built_proxy_settings(self):
        """Test building proxy settings from proxy config"""
        proxy_config = ProxyConfig(**{
            'host': 'test_host',
            'port': 1234,
        })
        proxy_manager = ProxyManager(proxy_config)
        proxy_settings = proxy_manager._build_proxy_settings_()
        self.assertDictEqual(proxy_settings, {
            'http': 'http://test_host:1234',
            'https': 'https://test_host:1234'
        })

    def test_configure_sync_proxy(self):
        """Test configuring a synchronous proxy"""
        mock_proxy_config = ProxyConfig(**{
            'host': 'test_host',
            'port': 1234,
        })
        mock_proxy_manager = ProxyManager(mock_proxy_config)
        mock_proxy_manager._patch_session_proxies_ = Mock(return_value={})
        mock_proxy_manager.configure_proxy_sync()
        mock_proxy_manager._patch_session_proxies_.assert_called_once()

    def test_configure_ssync_proxy(self):
        """Test configuring a asynchronous proxy"""
        mock_proxy_config = ProxyConfig(**{
            'host': 'test_host',
            'port': 1234,
        })
        mock_proxy_manager = ProxyManager(mock_proxy_config)
        mock_proxy_manager._patch_session_proxies_ = Mock(return_value={})
        mock_proxy_manager.configure_proxy_async()
        mock_proxy_manager._patch_session_proxies_.assert_called_once()

    def test_patch_session_proxies(self):
        """Test patching session proxies"""
        mock_proxy_config = ProxyConfig(**{
            'host': 'test_host',
            'port': 1234,
        })
        mock_proxy_manager = ProxyManager(mock_proxy_config)
        import requests
        pre_patch_session = requests.Session()
        pre_patch_proxies = pre_patch_session.proxies
        with mock_proxy_manager.configure_proxy_sync():
            test_session = requests.Session()
            self.assertDictEqual(test_session.proxies, {
                'http': 'http://test_host:1234',
                'https': 'https://test_host:1234'
            })
        post_patch_session = requests.Session()
        self.assertDictEqual(post_patch_session.proxies, pre_patch_proxies)
