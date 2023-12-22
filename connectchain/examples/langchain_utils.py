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
"""Example of using the langchain package to create a language chain."""
# pylint: disable=no-name-in-module
from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI as AzureOpenAI
from langchain.tools import PythonREPLTool
from connectchain.utils import get_token_from_env, Config

# pylint: disable=duplicate-code
if __name__ == '__main__':
    load_dotenv(find_dotenv())
    auth_token = get_token_from_env()
    model_config = Config.from_env().models['1']

    PROMPT_TEMPLATE = "Tell me about {topic}"
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

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run('computer science')
    print(output)

    agent_executor = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt="just output the result, no other comments",
    )

    output = agent_executor.run("what is 10 to the power of 10?")
    print(output)
