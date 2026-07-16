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
This module contains the ValidLLMChain class, which is a subclass of LLMChain.
In addition, it has a callback for sanitizing the output.
"""

import logging
from typing import Any, Callable, Dict, Optional

from langchain.chains.llm import LLMChain
from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun

logger = logging.getLogger(__name__)


class ValidLLMChain(LLMChain):
    # pylint: disable=too-few-public-methods
    """
    Extension to LLMChain that sanitizes the model response if an
    output_sanitizer callable is provided.

    The sanitizer runs inside _call()/_acall() -- the earliest point the raw LLM
    response becomes a dict, BEFORE Chain.invoke()/ainvoke()'s own machinery
    (prep_outputs) saves it to memory or fires the on_chain_end callback with it.
    Sanitizing later, e.g. by overriding invoke()/ainvoke() and sanitizing the dict
    those return, is too late: Chain.invoke() has already called
    self.memory.save_context(inputs, outputs) and run_manager.on_chain_end(outputs)
    with the *raw* outputs by that point (verified live: a memory-attached chain's
    save_context() and an on_chain_end callback both observed the unsanitized
    response even though the caller-visible return value was correctly sanitized).

    run(), arun(), invoke(), and ainvoke() all dispatch through _call()/_acall() via
    Chain's own __call__/invoke machinery, so this one override covers every entry
    point -- no per-dispatch-path override is needed. It is intentionally NOT applied
    to the user's input.
    """

    output_sanitizer: Optional[Callable[[str], str]]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        outputs = super()._call(inputs, run_manager=run_manager)
        return self._sanitize_dict(outputs)

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        outputs = await super()._acall(inputs, run_manager=run_manager)
        return self._sanitize_dict(outputs)

    def _sanitize_dict(self, result: Any) -> Any:
        """Apply output_sanitizer to result[self.output_key] in place, if configured.

        Logs a warning rather than silently no-op'ing when output_sanitizer
        is set but output_key isn't present in result (a downstream chain
        override or key mismatch), since an unlogged skip here would be an
        unsanitized-output bypass with no visible trace.
        """
        if not self.output_sanitizer:
            return result
        if isinstance(result, dict) and self.output_key in result:
            result[self.output_key] = self.output_sanitizer(result[self.output_key])
        else:
            logger.warning(
                "output_sanitizer is set but output_key '%s' not found in result: %s",
                self.output_key,
                list(result.keys()) if isinstance(result, dict) else type(result),
            )
        return result
