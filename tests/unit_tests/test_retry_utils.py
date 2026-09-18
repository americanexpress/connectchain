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
from unittest.mock import AsyncMock, Mock, call, patch

from connectchain.utils.exceptions import NonRetryableError
from connectchain.utils.retry import abase_retry, aretry_decorator, base_retry, retry_decorator


class _PermanentError(Exception, NonRetryableError):
    """Stand-in for LCELModelException/ConfigException in these generic retry tests."""


class _UnrelatedPermanentError(Exception, NonRetryableError):
    """A second, unrelated NonRetryableError-marked family. Used to prove the
    explicit opt-in is per-family: opting THIS family into retries must not
    disable fail-fast for _PermanentError (and vice versa)."""


def get_named_mock(*args, use_async=False, **kwargs) -> Mock:
    """Return a mock object."""
    mock = (AsyncMock if use_async else Mock)(*args, **kwargs)
    # Need to set the name of the mock function to avoid an AttributeError.
    mock.__name__ = "mock_func"
    return mock


class TestRetryUtils(TestCase):
    """Unit test class for the base_retry, abase_retry, retry_decorator, and aretry_decorator functions."""

    @patch("connectchain.utils.retry.sleep")
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
            "Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds."
        )

        # Test that the function retries the expected number of times with exponential backoff.
        mock_func = get_named_mock(side_effect=[Exception, Exception, 42])
        log_mock.reset_mock()
        self.assertEqual(base_retry(mock_func, max_retry=3, ebo=True, log_func=log_mock), 42)
        self.assertEqual(mock_func.call_count, 3)
        mock_sleep.call_args_list[0] == ((1,),)
        mock_sleep.call_args_list[1] == ((2,),)
        mock_sleep.call_args_list[2] == ((4,),)
        log_mock.assert_has_calls(
            [
                call(
                    "Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds."
                ),
                call(
                    "Attempt #2 of function mock_func failed with exception . Trying again in 2 seconds."
                ),
            ]
        )

        # Test that the function raises exception and calls log_func after max retries.
        mock_func = get_named_mock(side_effect=[Exception, Exception, Exception])
        log_mock.reset_mock()
        with self.assertRaises(Exception):
            base_retry(mock_func, max_retry=3, log_func=log_mock)
        self.assertEqual(mock_func.call_count, 3)
        log_mock.assert_has_calls(
            [
                call(
                    "Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds."
                ),
                call(
                    "Attempt #2 of function mock_func failed with exception . Trying again in 1 seconds."
                ),
                call("Function mock_func failed after 3 attempts."),
            ]
        )

    @patch("connectchain.utils.retry.sleep")
    def test_base_retry_does_not_retry_nonretryable_error(self, mock_sleep: Mock) -> None:
        """Regression test: an exception matching the default `exceptions=Exception`
        filter but marked NonRetryableError (e.g. LCELModelException for a missing
        API key) must fail on the first attempt, not be retried max_retry times."""
        mock_func = get_named_mock(side_effect=_PermanentError("permanent config error"))
        with self.assertRaises(_PermanentError):
            base_retry(mock_func, max_retry=3, log_func=Mock())
        self.assertEqual(mock_func.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("connectchain.utils.retry.sleep")
    def test_base_retry_explicit_optin_overrides_nonretryable_marker(
        self, mock_sleep: Mock
    ) -> None:
        """CODE-REVIEW regression: a caller who EXPLICITLY lists a
        NonRetryableError-marked type in `exceptions` has asked to retry that
        family; the marker's fail-fast must not silently override that explicit
        request (it only applies under the generic default filter)."""
        mock_func = get_named_mock(side_effect=_PermanentError("flagged but opted in"))
        with self.assertRaises(_PermanentError):
            base_retry(mock_func, max_retry=3, exceptions=_PermanentError, log_func=Mock())
        self.assertEqual(mock_func.call_count, 3)

    @patch("connectchain.utils.retry.sleep")
    def test_base_retry_nonretryable_optin_is_per_family(self, mock_sleep: Mock) -> None:
        """CODE-REVIEW regression: the explicit opt-in must be PER-FAMILY.
        `exceptions=(_UnrelatedPermanentError, Exception)` opts only
        _UnrelatedPermanentError into retries; a _PermanentError caught via the
        broad `Exception` entry is NOT covered by that opt-in and must still
        fail fast on the first attempt."""
        mock_func = get_named_mock(side_effect=_PermanentError("not the opted-in family"))
        with self.assertRaises(_PermanentError):
            base_retry(
                mock_func,
                max_retry=3,
                exceptions=(_UnrelatedPermanentError, Exception),
                log_func=Mock(),
            )
        self.assertEqual(mock_func.call_count, 1)
        mock_sleep.assert_not_called()

        # The family that WAS explicitly listed is retried, even alongside the
        # broad Exception entry -- the opt-in applies to it and only it.
        mock_func = get_named_mock(side_effect=_UnrelatedPermanentError("opted-in family"))
        with self.assertRaises(_UnrelatedPermanentError):
            base_retry(
                mock_func,
                max_retry=3,
                exceptions=(_UnrelatedPermanentError, Exception),
                log_func=Mock(),
            )
        self.assertEqual(mock_func.call_count, 3)

        # And an UNMARKED exception caught via the broad entry keeps normal
        # retry behavior -- fail-fast only ever applies to marked families.
        mock_func = get_named_mock(side_effect=[ValueError("transient"), 42])
        self.assertEqual(
            base_retry(
                mock_func,
                max_retry=3,
                exceptions=(_UnrelatedPermanentError, Exception),
                log_func=Mock(),
            ),
            42,
        )
        self.assertEqual(mock_func.call_count, 2)

    @patch("connectchain.utils.retry.asyncio.sleep")
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
            "Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds."
        )

        # Test that the function retries the expected number of times with exponential backoff.
        mock_func = get_named_mock(use_async=True, side_effect=[Exception, Exception, 42])
        log_mock.reset_mock()
        self.assertEqual(
            asyncio.run(abase_retry(mock_func, max_retry=3, ebo=True, log_func=log_mock)), 42
        )
        self.assertEqual(mock_func.call_count, 3)
        mock_sleep.call_args_list[0] == ((1,),)
        mock_sleep.call_args_list[1] == ((2,),)
        mock_sleep.call_args_list[2] == ((4,),)
        log_mock.assert_has_calls(
            [
                call(
                    "Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds."
                ),
                call(
                    "Attempt #2 of function mock_func failed with exception . Trying again in 2 seconds."
                ),
            ]
        )

        # Test that the function raises exception and calls log_func after max retries.
        mock_func = get_named_mock(use_async=True, side_effect=[Exception, Exception, Exception])
        log_mock.reset_mock()
        with self.assertRaises(Exception):
            asyncio.run(abase_retry(mock_func, max_retry=3, log_func=log_mock))
        self.assertEqual(mock_func.call_count, 3)
        log_mock.assert_has_calls(
            [
                call(
                    "Attempt #1 of function mock_func failed with exception . Trying again in 1 seconds."
                ),
                call(
                    "Attempt #2 of function mock_func failed with exception . Trying again in 1 seconds."
                ),
                call("Function mock_func failed after 3 attempts."),
            ]
        )

    @patch("connectchain.utils.retry.asyncio.sleep")
    def test_abase_retry_does_not_retry_nonretryable_error(self, mock_sleep: Mock) -> None:
        """Async counterpart of test_base_retry_does_not_retry_nonretryable_error."""
        mock_func = get_named_mock(use_async=True, side_effect=_PermanentError("permanent"))
        with self.assertRaises(_PermanentError):
            asyncio.run(abase_retry(mock_func, max_retry=3, log_func=Mock()))
        self.assertEqual(mock_func.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("connectchain.utils.retry.asyncio.sleep")
    def test_abase_retry_explicit_optin_overrides_nonretryable_marker(
        self, mock_sleep: Mock
    ) -> None:
        """Async counterpart of test_base_retry_explicit_optin_overrides_nonretryable_marker."""
        mock_func = get_named_mock(use_async=True, side_effect=_PermanentError("opted in"))
        with self.assertRaises(_PermanentError):
            asyncio.run(
                abase_retry(mock_func, max_retry=3, exceptions=_PermanentError, log_func=Mock())
            )
        self.assertEqual(mock_func.call_count, 3)

    @patch("connectchain.utils.retry.asyncio.sleep")
    def test_abase_retry_nonretryable_optin_is_per_family(self, mock_sleep: Mock) -> None:
        """Async counterpart of test_base_retry_nonretryable_optin_is_per_family."""
        mock_func = get_named_mock(
            use_async=True, side_effect=_PermanentError("not the opted-in family")
        )
        with self.assertRaises(_PermanentError):
            asyncio.run(
                abase_retry(
                    mock_func,
                    max_retry=3,
                    exceptions=(_UnrelatedPermanentError, Exception),
                    log_func=Mock(),
                )
            )
        self.assertEqual(mock_func.call_count, 1)
        mock_sleep.assert_not_called()

    def test_util_exception_is_nonretryable(self) -> None:
        """CODE-REVIEW regression: UtilException raises for permanent config errors
        (missing models, unset env keys) and must carry the NonRetryableError marker
        like its siblings (LCELModelException, ConfigException) -- otherwise a
        retry-wrapped get_token_from_env() burns max_retry attempts on an error
        that can never succeed."""
        from connectchain.utils.token_util import UtilException  # pylint: disable=import-outside-toplevel

        self.assertTrue(issubclass(UtilException, NonRetryableError))
        self.assertTrue(issubclass(UtilException, Exception))
        mock_func = get_named_mock(side_effect=UtilException("No models defined in config"))
        with self.assertRaises(UtilException):
            base_retry(mock_func, max_retry=3, sleep_time=0, log_func=Mock())
        self.assertEqual(mock_func.call_count, 1)

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
