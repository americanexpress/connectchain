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
"""Retryability contract tests for token_util exceptions.

UtilException carries the NonRetryableError marker because it is raised for
permanent config errors (missing models, undefined index, unset env keys,
expired cert) that can never succeed on retry. But a non-200 response from the
EAS auth service may be transient (5xx, throttling), so it must surface as the
plain, retry-eligible EASAuthError instead -- otherwise a retry-wrapped token
fetch fails fast on a recoverable server error. These tests pin down that
split. The EAS HTTP layer is mocked throughout; no network calls are made."""
import asyncio
from unittest import TestCase
from unittest.mock import Mock, patch

from .setup_utils import get_mock_config
from connectchain.utils import TokenUtil, UtilException
from connectchain.utils.exceptions import NonRetryableError
from connectchain.utils.retry import base_retry
from connectchain.utils.token_util import EASAuthError


class TestTokenUtilRetryability(TestCase):
    """Unit test class proving EAS auth failures are retried while permanent
    config errors still fail fast."""

    def test_eas_auth_error_is_retry_eligible(self) -> None:
        """EASAuthError must be a plain Exception: neither NonRetryableError-marked
        nor a UtilException subclass (which would inherit the marker)."""
        self.assertTrue(issubclass(EASAuthError, Exception))
        self.assertFalse(issubclass(EASAuthError, NonRetryableError))
        self.assertFalse(issubclass(EASAuthError, UtilException))

    @patch("connectchain.utils.retry.sleep")
    @patch("os.path.exists", return_value=True)
    def test_non_200_response_is_retried_by_base_retry(self, _mock_exists, mock_sleep) -> None:
        """CODE-REVIEW regression: a transient 5xx from the EAS auth service must
        be retryable. A retry-wrapped token fetch that gets a 503 and then a 200
        should succeed on the second attempt instead of failing fast."""
        # Secret must be valid base64 -- __get_signature b64-decodes it for the HMAC key.
        token_util = TokenUtil("test_id", "dGVzdF9zZWNyZXQ=", get_mock_config())
        responses = [
            ({"description": "Service Unavailable"}, 503),
            ({"authorization_token": "test_token"}, 200),
        ]
        with patch.object(
            TokenUtil, "_TokenUtil__aio_http_post", side_effect=responses
        ) as mock_post:

            def fetch_token() -> str:
                return asyncio.run(token_util.get_token(token_util.config.models["1"]))

            token = base_retry(fetch_token, max_retry=3, log_func=Mock())
        self.assertEqual(token, "Bearer test_token")
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch("connectchain.utils.retry.sleep")
    @patch("os.path.exists", return_value=True)
    def test_non_200_response_raises_eas_auth_error(self, _mock_exists, _mock_sleep) -> None:
        """A persistent non-200 surfaces as EASAuthError (not UtilException) after
        the retries are exhausted, carrying the service's error description."""
        token_util = TokenUtil("test_id", "dGVzdF9zZWNyZXQ=", get_mock_config())
        with patch.object(
            TokenUtil,
            "_TokenUtil__aio_http_post",
            return_value=({"description": "Internal Server Error"}, 500),
        ) as mock_post:

            def fetch_token() -> str:
                return asyncio.run(token_util.get_token(token_util.config.models["1"]))

            with self.assertRaisesRegex(EASAuthError, "Internal Server Error"):
                base_retry(fetch_token, max_retry=3, log_func=Mock())
        # All attempts were used -- proof the error did NOT fail fast.
        self.assertEqual(mock_post.call_count, 3)

    @patch("connectchain.utils.retry.sleep")
    @patch(
        "connectchain.utils.token_util.Config.from_env",
        return_value=get_mock_config({}),
    )
    def test_missing_models_config_error_still_fails_fast(
        self, _mock_config, mock_sleep
    ) -> None:
        """Contrast case: the permanent no-models config error keeps the
        NonRetryableError marker, so a retry-wrapped get_token_from_env() fails
        on the first attempt instead of burning max_retry attempts."""
        # pylint: disable-next=import-outside-toplevel
        from connectchain.utils import get_token_from_env

        mock_func = Mock(wraps=get_token_from_env)
        mock_func.__name__ = "get_token_from_env"
        with self.assertRaisesRegex(UtilException, "No models defined in config"):
            base_retry(mock_func, args=("other",), max_retry=3, log_func=Mock())
        self.assertEqual(mock_func.call_count, 1)
        mock_sleep.assert_not_called()
