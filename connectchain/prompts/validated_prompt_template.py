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
This module contains unit tests for the ValidPromptTemplate class.
It takes an output_sanitizer function as an argument to the constructor.
This function is called on the output of the format_prompt method for validation.
"""
#pylint: disable=too-many-ancestors
from typing import Any, Callable

from langchain.prompts import PromptTemplate
from langchain.schema import PromptValue


class ValidPromptTemplate(PromptTemplate):
    """Class to validate Prompt Template"""
    output_sanitizer: Callable[[str], str] | None

    # pylint: disable=E1102
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        if self.output_sanitizer:
            kwargs = {k: self.output_sanitizer(v) for k, v in kwargs.items()}
        return super().format_prompt(**kwargs)
