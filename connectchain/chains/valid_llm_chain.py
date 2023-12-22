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
In addition, it has a callback for sanitizing the output
"""
from typing import Callable, Optional, Any, Dict, List
from langchain.callbacks.base import Callbacks
from langchain.chains import LLMChain


class ValidLLMChain(LLMChain):
    #pylint: disable=too-few-public-methods
    """
    Extension to LLMChain that sanitizes the output if provided with a sanitizer function
    """
    output_sanitizer: Optional[Callable[[str], str]]

    def run(
            self,
            *args: Any,
            callbacks: Callbacks = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        # pylint: disable=unused-argument
        """ Run the chain with the given input and return the output """
        query = self.output_sanitizer(args[0]) if self.output_sanitizer else args[0]

        return super().run(query, callbacks=callbacks, tags=tags, metadata=metadata)
