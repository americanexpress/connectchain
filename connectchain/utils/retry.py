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
""" Retry utilities and decorators for functions that may raise exceptions. """
import asyncio
from functools import wraps
from typing import Union, Tuple, Callable
from time import sleep


def base_retry( # pylint: disable=too-many-arguments, too-many-positional-arguments
    func,
    args = None,
    kwargs = None,
    max_retry=3,
    sleep_time=1,
    exceptions: Union[Tuple[Exception], Exception] = Exception,
    ebo: bool = False,
    log_func: Callable = print
):
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
    f_name = getattr(func, '__qualname__', func.__name__)
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            attempt += 1
            next_sleep = sleep_time * (2 ** (attempt - 1)) if ebo else sleep_time
            if attempt < max_retry:
                log_func(f'Attempt #{attempt} of function {f_name} failed with exception {e}. '
                    f'Trying again in {next_sleep} seconds.')
                sleep(next_sleep)
                continue
            log_func(f'Function {f_name} failed after {attempt} attempts.')
            raise e

async def abase_retry( # pylint: disable=too-many-arguments, too-many-positional-arguments
    func,
    args = None,
    kwargs = None,
    max_retry=3,
    sleep_time=1,
    exceptions: Union[Tuple[Exception], Exception] = Exception,
    ebo: bool = False,
    log_func: Callable = print
):
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
    f_name = getattr(func, '__qualname__', func.__name__)
    attempt = 0
    while True:
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            attempt += 1
            next_sleep = sleep_time * (2 ** (attempt - 1)) if ebo else sleep_time
            if attempt < max_retry:
                log_func(f'Attempt #{attempt} of function {f_name} failed with exception {e}. '
                    f'Trying again in {next_sleep} seconds.')
                await asyncio.sleep(next_sleep)
                continue
            log_func(f'Function {f_name} failed after {attempt} attempts.')
            raise e

def retry_decorator(
    max_retry=3,
    sleep_time=1,
    exceptions: Union[Tuple[Exception], Exception] = Exception,
    ebo: bool = False,
    log_func: Callable = print
):
    """Decorator for retrying functions that may raise exceptions.

    Args:
        max_retry (int): The maximum number of retries.
        sleep_time (int): The time to sleep between retries.
        exceptions (Union[List[Exception], Exception]): The exceptions to catch.
        ebo (bool): Whether to use exponential backoff.
        log_func (callable): The function to use for logging."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return base_retry(func, args, kwargs, max_retry, sleep_time, exceptions, ebo, log_func)
        return wrapper
    return decorator

def aretry_decorator(
    max_retry=3,
    sleep_time=1,
    exceptions: Union[Tuple[Exception], Exception] = Exception,
    ebo: bool = False,
    log_func: Callable = print
):
    """Decorator for retrying async functions that may raise exceptions.
    
    Args:
        max_retry (int): The maximum number of retries.
        sleep_time (int): The time to sleep between retries.
        exceptions (Union[List[Exception], Exception]): The exceptions to catch.
        ebo (bool): Whether to use exponential backoff.
        log_func (Callable): The function to use for logging."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await abase_retry(func, args, kwargs, max_retry, sleep_time, exceptions, ebo, log_func)
        return wrapper
    return decorator
