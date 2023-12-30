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
import functools
from .proxy_manager import ProxyManager, ProxyConfig
from langchain.llms import BaseLLM

_llm_sync_methods_ = [
    'invoke',
    'batch',
    'stream',
    '_generate',
    '_stream',
    'generate_prompt',
    '_generate_helper',
    'generate',
    '__call__',
    'predict',
    'predict_messages'
]
_llm_async_methods_ = [
    'ainvoke',
    'abatch',
    'astream',
    '_agenerate',
    '_astream',
    'agenerate_prompt',
    '_agenerate_helper',
    'agenerate',
    '_call_async',
    'apredict',
    'apredict_messages',
]

def _sync_proxy_(func, proxy_manager: ProxyManager):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with proxy_manager.configure_proxy_sync():
            return func(self, *args, **kwargs)
    return wrapper

def _async_proxy_(func, proxy_manager: ProxyManager):
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        async with proxy_manager.configure_proxy_async():
            return await func(self, *args, **kwargs)
    return wrapper

def _wrap_method_(mixin, llm, method_name, decorator):
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

def wrap_llm_with_proxy(llm: BaseLLM, proxy_config: ProxyConfig | None):

    proxy_mixin = ProxyManager(proxy_config)

    decorator_pairs = [
        (_llm_sync_methods_, _sync_proxy_),
        (_llm_async_methods_, _async_proxy_)
    ]

    for methods, decorator in decorator_pairs:
        for wrap_method_name in methods:
            _wrap_method_(proxy_mixin, llm, wrap_method_name, decorator)
