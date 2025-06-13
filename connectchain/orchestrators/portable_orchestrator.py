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
"""
This module contains the PortableOrchestrator class.
"""
from typing import Any, List, Optional

import connectchain.chains
import connectchain.prompts
import connectchain.utils
from connectchain.lcel import model


class PortableOrchestrator:
    """
    This class is a portable orchestrator that can be used to run a query against any chain.
    It is portable as it can wrap any third-party LLM framework.
    """

    def __init__(self, chain: connectchain.chains.ValidLLMChain, **kvargs: Any) -> None:
        """Constructor for the PortableOrchestrator class"""
        self._chain = chain
        is_lcel = kvargs.get("lcel")
        self._is_lcel = is_lcel if is_lcel is not None and is_lcel is True else False

    @staticmethod
    def from_prompt_template(
        prompt_template: str, input_variables: List[str], **kwargs: Any
    ) -> "PortableOrchestrator":
        """Method to build a PortableOrchestrator instance"""
        index = kwargs.get("index")
        llm = model(index or "1")

        prompt_sanitizer = (
            kwargs["prompt_sanitizer"] if kwargs.get("prompt_sanitizer") else lambda x: x
        )

        prompt = connectchain.prompts.ValidPromptTemplate(
            output_sanitizer=prompt_sanitizer,
            input_variables=input_variables,
            template=prompt_template,
        )

        chain = connectchain.chains.ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=None)
        return PortableOrchestrator(chain)

    def run_sync(self, query: str) -> Any:
        """Run the chain synchronously"""
        return self._chain.run(query)

    async def run(self, query: str) -> Any:
        """Run the chain asynchronously"""
        return await self._chain.arun(query)
