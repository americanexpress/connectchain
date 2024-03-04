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
"""Example for using a code generation tool.

IMPORTANT: This is a simplified example designed to showcase concepts and should not used
as a reference for production code. The features are experimental and may not be suitable for
use in sensitive environments or without additional safeguards and testing.

Any use of this code is at your own risk.
"""
#pylint: disable=no-name-in-module
import re
from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI as AzureOpenAI
from connectchain.tools import ValidPythonREPLTool
from connectchain.utils import get_token_from_env, Config
from connectchain.utils.exceptions import OperationNotPermittedException

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    auth_token = get_token_from_env()
    model_config = Config.from_env().models['1']

    llm = AzureOpenAI(
        model_name=model_config.model_name,
        openai_api_key=auth_token,
        openai_api_base=model_config.api_base,
        model_kwargs={
            "engine": model_config.engine,
            "api_version": model_config.api_version,
            "api_type": "azure"
        })

    def simple_sanitizer(query: str) -> str:
        """Sample sanitizer
        
        IMPORTANT: This is a simplified example designed to showcase concepts and should not used
        as a reference for production code. The features are experimental and may not be suitable for
        use in sensitive environments or without additional safeguards and testing.

        Any use of this code is at your own risk.
        """
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)

        pattern = r"with open\('[\w\-\.]+','wb'\) as \w+:"
        match = re.search(pattern, query)
        if match:
            raise OperationNotPermittedException(f'Illegal execution detected: {query}')

        print('my_sanitizer: ', query)
        return query


    agent_executor = create_python_agent(
        llm=llm,
        tool=ValidPythonREPLTool(simple_sanitizer),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt="Just output the result, no other text or comments.",
    )

    output = agent_executor.run("what is 10 to the power of 10?")
    print(output)

    # this will throw an exception as we are trying something illegal
    try:
        output = agent_executor.run("with open('test.txt','wb') as f: f.write(b'hello world')")
    except OperationNotPermittedException as e:
        print(f'Execution aborted. I/O operation detected: {e}')
