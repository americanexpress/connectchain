# Copyright 2025 American Express Travel Related Services Company, Inc.
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
"""Unit test module for the LCELRetry class"""
import asyncio
from unittest import TestCase
from unittest.mock import MagicMock, AsyncMock, patch
from connectchain.lcel.retry import LCELRetry


class TestLCELRetry(TestCase):
    """Unit test for the LCELRetry class"""

    def tearDown(self):
        """Tear down the test"""
        patch.stopall()

    def test_init(self):
        """Test the __init__ method"""
        runnable = MagicMock()
        max_retry = 3
        sleep_time = 1
        exceptions = Exception
        ebo = False
        log_func = print
        lcel_retry = LCELRetry(runnable, max_retry, sleep_time, exceptions, ebo, log_func)
        self.assertEqual(lcel_retry.runnable, runnable)
        self.assertEqual(lcel_retry.max_retry, max_retry)
        self.assertEqual(lcel_retry.sleep_time, sleep_time)
        self.assertEqual(lcel_retry.exceptions, exceptions)
        self.assertEqual(lcel_retry.ebo, ebo)
        self.assertEqual(lcel_retry.log_func, log_func)

    @patch('connectchain.lcel.retry.base_retry')
    def test_call(self, mock_base_retry):
        """Test the __call__ method"""
        runnable = MagicMock()
        lcel_retry = LCELRetry(runnable)
        lcel_retry.__call__()
        mock_base_retry.assert_called_once()

    @patch('connectchain.lcel.retry.abase_retry')
    def test_acall(self, mock_abase_retry):
        """Test the __acall__ method"""
        runnable = MagicMock()
        lcel_retry = LCELRetry(runnable)
        lcel_retry.runnable.__acall__ = AsyncMock()
        asyncio.run(lcel_retry.__acall__())
        mock_abase_retry.assert_called_once()

    @patch('connectchain.lcel.retry.base_retry')
    def test_invoke(self, mock_base_retry):
        """Test the invoke method"""
        runnable = MagicMock()
        lcel_retry = LCELRetry(runnable)
        lcel_retry.runnable.invoke = MagicMock()
        lcel_retry.invoke(1)
        mock_base_retry.assert_called_once()

    @patch('connectchain.lcel.retry.abase_retry')
    def test_ainvoke(self, mock_abase_retry):
        """Test the ainvoke method"""
        runnable = MagicMock()
        lcel_retry = LCELRetry(runnable)
        lcel_retry.runnable.ainvoke = AsyncMock()
        asyncio.run(lcel_retry.ainvoke(1))
        mock_abase_retry.assert_called_once()
