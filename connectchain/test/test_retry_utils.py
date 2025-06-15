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
"""Unit test module for the base_retry, abase_retry, retry_decorator, and aretry_decorator functions."""
import asyncio
from unittest import TestCase
from unittest.mock import patch, Mock, AsyncMock, call
from connectchain.utils.retry import base_retry, abase_retry, retry_decorator, aretry_decorator


def get_named_mock(*args, use_async=False, **kwargs) -> Mock:
    """Return a mock object."""
    mock = (AsyncMock if use_async else Mock)(*args, **kwargs)
    # Need to set the name of the mock function to avoid an AttributeError.
    mock.__name__ = 'mock_func'
    return mock

class TestRetryUtils(TestCase):
    """Unit test class for the base_retry, abase_retry, retry_decorator, and aretry_decorator functions."""
    @patch('connectchain.utils.retry.sleep')
    def test_base_retry(self, mock_sleep: Mock) -> None:
        """Unit test for the base_retry function."""
        # Test that the function returns the expected value.
        def test_func() -> int:
            return 42

        self.assertEqual(base_retry(test_func), 42)

        # Test that the function retries the expected number of times.
        mock_func = get_named_mock(side_effect=[Exception, 42])
        log_mock = Mock()
        self.assertEqual(base_retry(mock_func, log_func=log_mock), 42)
        self.assertEqual(mock_func.call_count, 2)
        mock_sleep.assert_called_once_with(1)
        log_mock.assert_called_once_with(
            'Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds.')

        # Test that the function retries the expected number of times with exponential backoff.
        mock_func = get_named_mock(side_effect=[Exception, Exception, 42])
        log_mock.reset_mock()
        self.assertEqual(base_retry(mock_func, max_retry=3, ebo=True, log_func=log_mock), 42)
        self.assertEqual(mock_func.call_count, 3)
        mock_sleep.call_args_list[0] == ((1,),)
        mock_sleep.call_args_list[1] == ((2,),)
        mock_sleep.call_args_list[2] == ((4,),)
        log_mock.assert_has_calls([
            call('Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds.'),
            call('Attempt #2 of function mock_func failed with exception . Trying again in 2 seconds.')
        ])

        # Test that the function raises exception and calls log_func after max retries.
        mock_func = get_named_mock(side_effect=[Exception, Exception, Exception])
        log_mock.reset_mock()
        with self.assertRaises(Exception):
            base_retry(mock_func, max_retry=3, log_func=log_mock)
        self.assertEqual(mock_func.call_count, 3)
        log_mock.assert_has_calls([
            call('Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds.'),
            call('Attempt #2 of function mock_func failed with exception . Trying again in 1 seconds.'),
            call('Function mock_func failed after 3 attempts.')
        ])

    @patch('connectchain.utils.retry.asyncio.sleep')
    def test_abase_retry(self, mock_sleep: Mock) -> None:
        """Unit test for the abase_retry function."""
        # Test that the function returns the expected value.
        test_func = get_named_mock(use_async=True, return_value=42)
        self.assertEqual(asyncio.run(abase_retry(test_func)), 42)

        # Test that the function retries the expected number of times.
        mock_func = get_named_mock(use_async=True, side_effect=[Exception, 42])
        log_mock = Mock()
        self.assertEqual(asyncio.run(abase_retry(mock_func, log_func=log_mock)), 42)
        self.assertEqual(mock_func.call_count, 2)
        mock_sleep.assert_called_once_with(1)
        log_mock.assert_called_once_with(
            'Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds.')

        # Test that the function retries the expected number of times with exponential backoff.
        mock_func = get_named_mock(use_async=True, side_effect=[Exception, Exception, 42])
        log_mock.reset_mock()
        self.assertEqual(asyncio.run(abase_retry(mock_func, max_retry=3, ebo=True, log_func=log_mock)), 42)
        self.assertEqual(mock_func.call_count, 3)
        mock_sleep.call_args_list[0] == ((1,),)
        mock_sleep.call_args_list[1] == ((2,),)
        mock_sleep.call_args_list[2] == ((4,),)
        log_mock.assert_has_calls([
            call('Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds.'),
            call('Attempt #2 of function mock_func failed with exception . Trying again in 2 seconds.')
        ])

        # Test that the function raises exception and calls log_func after max retries.
        mock_func = get_named_mock(use_async=True, side_effect=[Exception, Exception, Exception])
        log_mock.reset_mock()
        with self.assertRaises(Exception):
            asyncio.run(abase_retry(mock_func, max_retry=3, log_func=log_mock))
        self.assertEqual(mock_func.call_count, 3)
        log_mock.assert_has_calls([
            call('Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds.'),
            call('Attempt #2 of function mock_func failed with exception . Trying again in 1 seconds.'),
            call('Function mock_func failed after 3 attempts.')
        ])

    def test_retry_decorator(self) -> None:
        """Unit test for the retry_decorator function."""
        # Test that the decorator returns the expected value.
        @retry_decorator()
        def test_func() -> int:
            return 42
        self.assertEqual(test_func(), 42)

        # Test that the decorator retries the expected number of times.
        mock_func = get_named_mock(side_effect=[Exception, 42])
        @retry_decorator()
        def test_func() -> int:
            return mock_func()
        self.assertEqual(test_func(), 42)
        self.assertEqual(mock_func.call_count, 2)

        # Test that the decorator retries the expected number of times with exponential backoff.
        mock_func = get_named_mock(side_effect=[Exception, Exception, 42])
        @retry_decorator(max_retry=3, ebo=True)
        def test_func() -> int:
            return mock_func()
        self.assertEqual(test_func(), 42)
        self.assertEqual(mock_func.call_count, 3)

    def test_aretry_decorator(self) -> None:
        """Unit test for the aretry_decorator function."""
        # Test that the decorator returns the expected value.
        test_func = aretry_decorator()(get_named_mock(use_async=True, return_value=42))
        self.assertEqual(asyncio.run(test_func()), 42)

        # Test that the decorator retries the expected number of times.
        mock_func = get_named_mock(use_async=True, side_effect=[Exception, 42])
        test_func = aretry_decorator()(mock_func)
        self.assertEqual(asyncio.run(test_func()), 42)
        self.assertEqual(mock_func.call_count, 2)

        # Test that the decorator retries the expected number of times with exponential backoff.
        mock_func = get_named_mock(use_async=True, side_effect=[Exception, Exception, 42])
        test_func = aretry_decorator(max_retry=3, ebo=True)(mock_func)
        self.assertEqual(asyncio.run(test_func()), 42)
        self.assertEqual(mock_func.call_count, 3)
