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
import logging
from typing import Any, List

import connectchain.chains
import connectchain.prompts
import connectchain.utils
from connectchain.lcel import model

logger = logging.getLogger(__name__)


class PortableOrchestrator:
    """
    This class is a portable orchestrator that can be used to run a query
    against any chain.  It is portable as it can wrap any third-party LLM
    framework.

    run_sync()/run() invoke the chain via the LCEL .invoke()/.ainvoke() API,
    passing query through unwrapped so Chain.prep_inputs() maps it onto the
    chain's own input key -- a bare value for single-input chains, or used
    as-is when the caller passes a pre-built dict for a multi-input chain.
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
        """Build a PortableOrchestrator from a prompt template.

        kwargs:
            index: model config index (default "1").
            prompt_sanitizer: applied to the rendered prompt before the LLM call.
            output_sanitizer: applied to the LLM's response before it's returned.
        """
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

        chain = connectchain.chains.ValidLLMChain(
            llm=llm, prompt=prompt, output_sanitizer=kwargs.get("output_sanitizer")
        )
        return PortableOrchestrator(chain)

    def run_sync(self, query: Any) -> Any:
        """Run the chain synchronously via LCEL .invoke()."""
        result = self._chain.invoke(query)
        return self._extract_output(result)

    async def run(self, query: Any) -> Any:
        """Run the chain asynchronously via LCEL .ainvoke()."""
        result = await self._chain.ainvoke(query)
        return self._extract_output(result)

    def _extract_output(self, result: Any) -> Any:
        """Pull the response out of a chain's output dict via its output_key
        (default "text"), falling back to str(result) if that key is absent.
        """
        if isinstance(result, dict):
            output_key = getattr(self._chain, "output_key", "text")
            missing = object()
            value = result.get(output_key, missing)
            if value is not missing:
                return value
            logger.warning(
                "output_key '%s' not found in result; falling back to str(result). Keys: %s",
                output_key,
                list(result.keys()),
            )
            return str(result)
        return result
