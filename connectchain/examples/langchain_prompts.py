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
"""Example for using a custom sanitizer"""
import re

from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI as AzureOpenAI
from connectchain.utils import get_token_from_env, Config
from connectchain.prompts import ValidPromptTemplate


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    auth_token = get_token_from_env()
    model_config = Config.from_env().models['1']

    class OperationNotPermittedException(Exception):
        """Operation Not Permitted Exception"""


    def my_sanitizer(query: str) -> str:
        """Sample sanitizer"""
        pattern = r'BADWORD'

        if re.search(pattern, query):
            print("BADWORD found!")
            raise OperationNotPermittedException(f'Illegal execution detected: {query}')

        return query



    prompt_template = "Tell me about {adjective} books"
    prompt = ValidPromptTemplate(
        output_sanitizer=my_sanitizer,
        input_variables=["adjective"],
        template=prompt_template
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

    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain.run('history and science')
    print(output)

    try:
        chain.run('BADWORD')
    except OperationNotPermittedException as e:
        print(f'Bad stuff: {e}')
