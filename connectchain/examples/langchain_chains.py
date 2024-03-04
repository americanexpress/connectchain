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
Example usage for the ValidLLMChain class.

IMPORTANT: This is a simplified example designed to showcase concepts and should not used
as a reference for production code. The features are experimental and may not be suitable for
use in sensitive environments or without additional safeguards and testing.

Any use of this code is at your own risk.
"""

from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from connectchain.chains import ValidLLMChain
from connectchain.lcel import model
from connectchain.utils.exceptions import OperationNotPermittedException


def my_sanitizer(query: str) -> str:
    """Sample sanitizer

    IMPORTANT: This is a simplified example designed to showcase concepts and should not used
    as a reference for production code. The features are experimental and may not be suitable for
    use in sensitive environments or without additional safeguards and testing.

    Any use of this code is at your own risk.
    """
    if query == "BADWORD":
        raise OperationNotPermittedException(f"Illegal execution detected: {query}")
    return query

#pylint: disable=duplicate-code
if __name__ == '__main__':
    load_dotenv(find_dotenv())

    PROMPT_TEMPLATE = "Tell me about {adjective} animals"
    prompt = PromptTemplate(
        input_variables=["adjective"], template=PROMPT_TEMPLATE
    )

    chain = ValidLLMChain(llm=model('1'), prompt=prompt, output_sanitizer=my_sanitizer)

    output = chain.run('cute and cuddly')
    print(output)

    try:
        output = chain.run('BADWORD')
    except OperationNotPermittedException as e:
        print(e)
