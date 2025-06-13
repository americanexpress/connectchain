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
"""Example for using a custom sanitizer.

IMPORTANT: This is a simplified example designed to showcase concepts and should not used
as a reference for production code. The features are experimental and may not be suitable for
use in sensitive environments or without additional safeguards and testing.

Any use of this code is at your own risk.
"""
import re

from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain

from connectchain.lcel import model
from connectchain.prompts import ValidPromptTemplate
from connectchain.utils.exceptions import OperationNotPermittedException

if __name__ == "__main__":
    load_dotenv(find_dotenv())

    def example_sanitizer(query: str) -> str:
        """Sample sanitizer

        IMPORTANT: This is a simplified example designed to showcase concepts and should not used
        as a reference for production code. The features are experimental and may not be suitable for
        use in sensitive environments or without additional safeguards and testing.

        Any use of this code is at your own risk.
        """
        pattern = r"BADWORD"

        if re.search(pattern, query):
            print("BADWORD found!")
            raise OperationNotPermittedException(f"Illegal execution detected: {query}")

        return query

    prompt_template = "Tell me about {adjective} books"
    prompt = ValidPromptTemplate(
        output_sanitizer=example_sanitizer, input_variables=["adjective"], template=prompt_template
    )

    chain = LLMChain(llm=model("1"), prompt=prompt)

    output = chain.run("history and science")
    print(output)

    try:
        chain.run("BADWORD")
    except OperationNotPermittedException as e:
        print(f"Bad stuff: {e}")
