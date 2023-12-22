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
"""

from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI as AzureOpenAI
from connectchain.chains import ValidLLMChain
from connectchain.utils import get_token_from_env, Config

class OperationNotPermittedException(BaseException):
    """Operation Not Permitted Exception"""


def my_sanitizer(query: str) -> str:
    """Sample sanitizer"""
    if query == "BADWORD":
        raise OperationNotPermittedException(f"Illegal execution detected: {query}")
    return query

#pylint: disable=duplicate-code
if __name__ == '__main__':
    load_dotenv(find_dotenv())
    auth_token = get_token_from_env()
    model_config = Config.from_env().models['1']

    PROMPT_TEMPLATE = "Tell me about {adjective} animals"
    prompt = PromptTemplate(
        input_variables=["adjective"], template=PROMPT_TEMPLATE
    )

    llm = AzureOpenAI(
        model_name=model_config.model_name,
        openai_api_key=auth_token,
        openai_api_base=model_config.api_base,
        model_kwargs={
            "engine": model_config.engine,
            "api_version": model_config.api_version,
            "api_type": "azure"
        })

    chain = ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=my_sanitizer)

    output = chain.run('cute and cuddly')
    print(output)

    try:
        output = chain.run('BADWORD')
    except OperationNotPermittedException as e:
        print(e)
