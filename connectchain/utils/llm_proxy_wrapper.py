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
"""Proxied LLM Utilities"""
import functools
from typing import Any, Callable, List, Optional

from langchain.llms import BaseLLM

from .proxy_manager import ProxyConfig, ProxyManager

_llm_sync_methods_: List[str] = [
    "invoke",
    "batch",
    "stream",
    "_generate",
    "_stream",
    "generate_prompt",
    "_generate_helper",
    "generate",
    "__call__",
    "predict",
    "predict_messages",
]
_llm_async_methods_: List[str] = [
    "ainvoke",
    "abatch",
    "astream",
    "_agenerate",
    "_astream",
    "agenerate_prompt",
    "_agenerate_helper",
    "agenerate",
    "_call_async",
    "apredict",
    "apredict_messages",
]


def _sync_proxy_(func: Callable[..., Any], proxy_manager: ProxyManager) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with proxy_manager.configure_proxy_sync():
            return func(self, *args, **kwargs)

    return wrapper


def _async_proxy_(func: Callable[..., Any], proxy_manager: ProxyManager) -> Callable[..., Any]:
    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with proxy_manager.configure_proxy_async():
            return await func(self, *args, **kwargs)

    return wrapper


def _wrap_method_(
    mixin: ProxyManager, llm: BaseLLM, method_name: str, decorator: Callable[..., Any]
) -> None:
    """Langchain misuses pydantic so we must be careful accessing attributes even when we know they exist.
    For the same reason, we must 'force-feed' the decorated function back to the class instance by setting
    directly on the instance __dict__ to avoid pydantic validation.

    Note: pydantic is a data validation framework and is not intended to be used as an architecture validation
    tool; despite the fact that it is used as such in Langchain."""
    if hasattr(llm, method_name):
        try:
            func = getattr(llm, method_name)
        except ValueError:
            return
        wrapped_func = decorator(func, mixin)
        llm.__dict__[method_name] = wrapped_func


def wrap_llm_with_proxy(llm: BaseLLM, proxy_config: Optional[ProxyConfig]) -> None:
    """Wrap an LLM instance with proxy functionality for network requests."""

    proxy_mixin = ProxyManager(proxy_config)

    decorator_pairs = [(_llm_sync_methods_, _sync_proxy_), (_llm_async_methods_, _async_proxy_)]

    for methods, decorator in decorator_pairs:
        for wrap_method_name in methods:
            _wrap_method_(proxy_mixin, llm, wrap_method_name, decorator)
