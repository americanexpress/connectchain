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
"""Unit testing for llm proxy wrapping utilities"""
import unittest
import asyncio
from unittest.mock import Mock, patch
from connectchain.utils.proxy_manager import ProxyManager
from connectchain.utils.llm_proxy_wrapper import _sync_proxy_, _async_proxy_, _wrap_method_, wrap_llm_with_proxy,\
    _llm_async_methods_, _llm_sync_methods_
from .setup_utils import get_mock_ctx_manager

class MockLLM:
    """Mock LLM class"""
    def __init__(self):
        self._test = 'test'

    def test_method(self):
        """Test method"""
        return self._test
    
    async def atest_method(self):
        """Async test method"""
        return self._test

class TestProxyWrapping(unittest.TestCase):
    """Unit testing the proxy wrapping methods"""

    def test_sync_proxy(self):
        """Test wrapping synchronous proxied method"""
        mock_llm = MockLLM()
        mock_ctx = get_mock_ctx_manager()
        mock_proxy_manager = Mock(ProxyManager, configure_proxy_sync=lambda: mock_ctx)
        wrapped_proxied_method = _sync_proxy_(MockLLM.test_method, mock_proxy_manager)
        self.assertEqual(wrapped_proxied_method.__name__, MockLLM.test_method.__name__)
        self.assertEqual(wrapped_proxied_method(mock_llm), 'test')
        mock_ctx.__enter__.assert_called_once()
        mock_ctx.__exit__.assert_called_once()

    def test_async_proxy(self):
        """Test wrapping asynchronous proxied method"""
        mock_llm = MockLLM()
        mock_ctx = get_mock_ctx_manager()
        mock_proxy_manager = Mock(ProxyManager, configure_proxy_async=lambda: mock_ctx)
        wrapped_proxied_method = _async_proxy_(MockLLM.atest_method, mock_proxy_manager)
        self.assertEqual(wrapped_proxied_method.__name__, MockLLM.atest_method.__name__)
        result = asyncio.run(wrapped_proxied_method(mock_llm))
        self.assertEqual(result, 'test')
        mock_ctx.__enter__.assert_called_once()
        mock_ctx.__exit__.assert_called_once()

    def test_wrap_method(self):
        """Test wrapping a method"""
        mock_llm = MockLLM()
        mock_manager = Mock()

        def test_wrapping_an_existing_method():
            mock_decorator = Mock()
            # must reference before call, as override with wrapped method occurs
            expected_method = getattr(mock_llm, 'test_method')
            _wrap_method_(mock_manager, mock_llm, 'test_method', mock_decorator)
            mock_decorator.assert_called_once_with(expected_method, mock_manager)

        def test_wrapping_nonexistant_method():
            mock_decorator = Mock()
            _wrap_method_(mock_manager, mock_llm, 'nonexistant_method', mock_decorator)
            mock_decorator.assert_not_called()

        test_wrapping_an_existing_method()
        test_wrapping_nonexistant_method()

    @patch('connectchain.utils.llm_proxy_wrapper._wrap_method_')
    def test_wrapping_llm(self, wrap_mock:Mock):
        """Test wrapping an LLM with a proxy"""
        mock_llm = Mock()
        mock_config = { 'test': 'value' }
        wrap_llm_with_proxy(mock_llm, mock_config)
        for called_wrap in wrap_mock.call_args_list:
            proxy_manager, llm, method_name, decorator = called_wrap[0]
            self.assertIsInstance(proxy_manager, ProxyManager)
            self.assertIs(llm, mock_llm)
            if method_name in _llm_sync_methods_:
                self.assertIs(decorator, _sync_proxy_)
            elif method_name in _llm_async_methods_:
                self.assertIs(decorator, _async_proxy_)
            else:
                self.fail(f'Unexpected method name {method_name}')
