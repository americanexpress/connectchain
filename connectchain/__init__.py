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
"""Init file for the connectchain package"""
from .utils.exceptions import ConnectChainNoAccessException
from langchain.chains.api.base import APIChain

# Disable langchain APIChain
def override(self, *args, **kwargs):
    raise ConnectChainNoAccessException("Operation not permitted")

APIChain.__init__ = override
APIChain.from_llm_and_api_docs = override
APIChain.run = override
APIChain.arun = override
APIChain.invoke = override
APIChain.ainvoke = override
APIChain.apply = override
APIChain.batch = override
APIChain.abatch = override
APIChain._call = override
APIChain._acall = override
