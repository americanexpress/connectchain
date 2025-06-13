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
from typing import Any

from langchain.chains.api.base import APIChain

from .utils.exceptions import ConnectChainNoAccessException


# Disable langchain APIChain
def override(self: Any, *args: Any, **kwargs: Any) -> None:
    raise ConnectChainNoAccessException("Operation not permitted")


# Use type ignore to suppress MyPy warnings for intentional method overrides
APIChain.__init__ = override  # type: ignore[method-assign]
APIChain.from_llm_and_api_docs = override  # type: ignore
APIChain.run = override  # type: ignore[method-assign]
APIChain.arun = override  # type: ignore[assignment]
APIChain.invoke = override  # type: ignore[assignment]
APIChain.ainvoke = override  # type: ignore[assignment]
APIChain.apply = override  # type: ignore[assignment]
APIChain.batch = override  # type: ignore[assignment]
APIChain.abatch = override  # type: ignore[assignment]
APIChain._call = override  # type: ignore[assignment]
APIChain._acall = override  # type: ignore[assignment]
