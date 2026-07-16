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

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate

from connectchain.chains import ValidLLMChain
from connectchain.lcel import model
from connectchain.utils.exceptions import OperationNotPermittedException


def my_sanitizer(response: str) -> str:
    """Sample output sanitizer.

    ValidLLMChain.output_sanitizer is applied to the LLM's *response*, not the
    user's input query -- to block or transform something in the prompt itself,
    use ValidPromptTemplate's own sanitizer instead (see connectchain.prompts).

    IMPORTANT: This is a simplified example designed to showcase concepts and should not used
    as a reference for production code. The features are experimental and may not be suitable for
    use in sensitive environments or without additional safeguards and testing.

    Any use of this code is at your own risk.
    """
    if "BADWORD" in response:
        raise OperationNotPermittedException(f"Illegal content detected in response: {response}")
    return response


# pylint: disable=duplicate-code
if __name__ == "__main__":
    load_dotenv(find_dotenv())

    PROMPT_TEMPLATE = "Tell me about {adjective} animals"
    prompt = PromptTemplate(input_variables=["adjective"], template=PROMPT_TEMPLATE)

    chain = ValidLLMChain(llm=model("1"), prompt=prompt, output_sanitizer=my_sanitizer)

    output = chain.run("cute and cuddly")
    print(output)

    try:
        # This only raises if the model's *response* happens to contain "BADWORD" --
        # unlike input validation, output sanitization can't reject a query up front.
        output = chain.run("cute and cuddly")
    except OperationNotPermittedException as e:
        print(e)
