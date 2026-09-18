# Copyright 2024 American Express Travel Related Services Company, Inc.
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
"""ConnectChain exceptions."""


class NonRetryableError:
    """Marker mixin for exceptions that must never be retried.

    connectchain.utils.retry's base_retry()/abase_retry() re-raise instances of this
    immediately, even when they match the caller's `exceptions` filter (which defaults
    to `Exception`). Use for permanent/config errors -- e.g. a missing environment
    variable or unsupported provider will never succeed on retry, so retrying just
    delays the inevitable failure and burns `max_retry` attempts on nothing.
    """


class OperationNotPermittedException(Exception):
    """Operation Not Permitted Exception"""


class ConnectChainNoAccessException(BaseException):
    """ConnectChain does not allow access to this class or method.

    Deliberately not Exception: this enforces the APIChain security block in
    connectchain/__init__.py, and must not be catchable by an ordinary
    `except Exception` around application code wrapping a chain call."""
