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
"""Retry utilities and decorators for functions that may raise exceptions."""
import asyncio
from functools import wraps
from time import sleep
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Union

from .exceptions import NonRetryableError


def _fail_fast_(
    error: BaseException,
    exceptions: Union[Tuple[type[BaseException], ...], type[BaseException]],
) -> bool:
    """Whether a caught error should skip retries entirely.

    NonRetryableError-marked exceptions fail fast under the default
    `exceptions=Exception` filter -- a permanent/config error can never succeed
    on retry. But a caller who EXPLICITLY lists a NonRetryableError-marked type
    in `exceptions` has opted into retrying that family, and that explicit
    request wins over the marker.

    The opt-in is strictly PER-FAMILY: fail-fast is skipped only when the
    caught error is an instance of an explicitly-listed type that is itself
    NonRetryableError-marked. Listing one marked type (e.g.
    `exceptions=(ConfigException, Exception)`) must not silently re-enable
    retries for every other marked exception that merely matches a broader
    entry like `Exception` -- those unrelated permanent errors still fail fast.
    """
    if not isinstance(error, NonRetryableError):
        return False
    exc_types = exceptions if isinstance(exceptions, tuple) else (exceptions,)
    return not any(issubclass(t, NonRetryableError) and isinstance(error, t) for t in exc_types)


def base_retry(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    func: Callable[..., Any],
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    max_retry: int = 3,
    sleep_time: int = 1,
    exceptions: Union[Tuple[type[BaseException], ...], type[BaseException]] = Exception,
    ebo: bool = False,
    log_func: Callable[[str], None] = print,
) -> Any:
    """Retry a function that may raise exceptions.

    Args:
        func (callable): The function to retry.
        args (tuple): The positional arguments for the function.
        kwargs (dict): The keyword arguments for the function.
        max_retry (int): The maximum number of retries.
        sleep_time (int): The time to sleep between retries.
        exceptions (Union[List[Exception], Exception]): The exceptions to catch.
        ebo (bool): Whether to use exponential backoff.
        log_func (callable): The function to use for logging."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    f_name = getattr(func, "__qualname__", func.__name__)
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            if _fail_fast_(e, exceptions):
                # Permanent/config error: matches `exceptions` but retrying it can
                # never succeed, so fail fast instead of burning max_retry attempts.
                raise
            attempt += 1
            next_sleep = sleep_time * (2 ** (attempt - 1)) if ebo else sleep_time
            if attempt < max_retry:
                log_func(
                    f"Attempt #{attempt} of function {f_name} failed with exception {e}. "
                    f"Trying again in {next_sleep} seconds."
                )
                sleep(next_sleep)
                continue
            log_func(f"Function {f_name} failed after {attempt} attempts.")
            raise e


async def abase_retry(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    func: Callable[..., Awaitable[Any]],
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    max_retry: int = 3,
    sleep_time: int = 1,
    exceptions: Union[Tuple[type[BaseException], ...], type[BaseException]] = Exception,
    ebo: bool = False,
    log_func: Callable[[str], None] = print,
) -> Any:
    """Retry an async function that may raise exceptions.

    Args:
        func (callable): The function to retry.
        args (tuple): The positional arguments for the function.
        kwargs (dict): The keyword arguments for the function.
        max_retry (int): The maximum number of retries.
        sleep_time (int): The time to sleep between retries.
        exceptions (Union[List[Exception], Exception]): The exceptions to catch.
        ebo (bool): Whether to use exponential backoff.
        log_func (callable): The function to use for logging."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    f_name = getattr(func, "__qualname__", func.__name__)
    attempt = 0
    while True:
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            if _fail_fast_(e, exceptions):
                raise
            attempt += 1
            next_sleep = sleep_time * (2 ** (attempt - 1)) if ebo else sleep_time
            if attempt < max_retry:
                log_func(
                    f"Attempt #{attempt} of function {f_name} failed with exception {e}. "
                    f"Trying again in {next_sleep} seconds."
                )
                await asyncio.sleep(next_sleep)
                continue
            log_func(f"Function {f_name} failed after {attempt} attempts.")
            raise e


def retry_decorator(
    max_retry: int = 3,
    sleep_time: int = 1,
    exceptions: Union[Tuple[type[BaseException], ...], type[BaseException]] = Exception,
    ebo: bool = False,
    log_func: Callable[[str], None] = print,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for retrying functions that may raise exceptions.

    Args:
        max_retry (int): The maximum number of retries.
        sleep_time (int): The time to sleep between retries.
        exceptions (Union[List[Exception], Exception]): The exceptions to catch.
        ebo (bool): Whether to use exponential backoff.
        log_func (callable): The function to use for logging."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return base_retry(func, args, kwargs, max_retry, sleep_time, exceptions, ebo, log_func)

        return wrapper

    return decorator


def aretry_decorator(
    max_retry: int = 3,
    sleep_time: int = 1,
    exceptions: Union[Tuple[type[BaseException], ...], type[BaseException]] = Exception,
    ebo: bool = False,
    log_func: Callable[[str], None] = print,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator for retrying async functions that may raise exceptions.

    Args:
        max_retry (int): The maximum number of retries.
        sleep_time (int): The time to sleep between retries.
        exceptions (Union[List[Exception], Exception]): The exceptions to catch.
        ebo (bool): Whether to use exponential backoff.
        log_func (Callable): The function to use for logging."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await abase_retry(
                func, args, kwargs, max_retry, sleep_time, exceptions, ebo, log_func
            )

        return wrapper

    return decorator
