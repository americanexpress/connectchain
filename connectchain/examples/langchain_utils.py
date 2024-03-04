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
"""Example of using the langchain package to create a language chain.

IMPORTANT: This example is a simplified example designed to showcase concepts and should not used
as a reference for production code. The features are experimental and may not be suitable for
use in sensitive environments or without additional safeguards and testing.

Any use of this code is at your own risk.
"""
# pylint: disable=no-name-in-module
from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import PythonREPLTool
from connectchain.lcel import model

# pylint: disable=duplicate-code
if __name__ == '__main__':
    load_dotenv(find_dotenv())

    PROMPT_TEMPLATE = "Tell me about {topic}"
    prompt = PromptTemplate(
        input_variables=["adjective"], template=PROMPT_TEMPLATE
    )
    llm = model('1')
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
