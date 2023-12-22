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
Example of using LCEL with multiple models.
"""
from dotenv import load_dotenv, find_dotenv
from langchain.schema import StrOutputParser
from connectchain.prompts import ValidPromptTemplate
from connectchain.lcel import model, Logger

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    class PrintLogger(Logger):
        """Prints the payload to the console"""
        def print(self, payload):
            print(payload)

    logger = PrintLogger()

    test_prompt = ValidPromptTemplate(
        input_variables=["description"],
        template=
        """
        Describe {description}.
        """
    )

    expand_prompt = ValidPromptTemplate(
        input_variables=["ballad_seed"],
        template=
        """
        Write an epic ballad given the following information: {ballad_seed}.
        """
    )

    chain = (
            { "ballad_seed": test_prompt
                | model()
                | logger.log()
                | StrOutputParser()
            }
            | expand_prompt
            | model('3')
            | logger.log()
            | StrOutputParser()
    )

    out = chain.invoke({ "description": "the universe" })
