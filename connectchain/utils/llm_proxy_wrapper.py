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

def _sync_proxy_(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.configure_proxy_sync():
            return func(self, *args, **kwargs)
    return wrapper

def _async_proxy_(func):
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        async with self.configure_proxy_async():
            return await func(self, *args, **kwargs)
    return wrapper

class ProxiedLLM(ProxyManager):
    _sync_methods_ = [
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
    _async_methods_ = [
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

    def __init__(self, llm: BaseLLM, proxy_config: ProxyConfig | None) -> None:
        self.llm = llm
        self.proxy_config = proxy_config
        self._wrap_proxy_methods_(ProxiedLLM._sync_methods_, _sync_proxy_)
        self._wrap_proxy_methods_(ProxiedLLM._async_methods_, _async_proxy_)
    
    def _wrap_proxy_methods_(self, names, wrapper):
        for wrap_method_name in names:
            setattr(self, wrap_method_name, wrapper(getattr(self.llm, wrap_method_name)))
