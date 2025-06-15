# Copyright 2025 American Express Travel Related Services Company, Inc.
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
"""This example demonstrates how to use the LCELRetry class to retry a failed chain."""
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from connectchain.lcel import LCELRetry, model


load_dotenv(find_dotenv())

EXAMPLE_INPUT = { 'species': 'birds' }

n_failure = 0
def simulated_failure(input):
    global n_failure
    n_failure += 1
    if n_failure < 3:
        raise Exception('Simulated failure')
    return model('2').invoke(input)

prompt = PromptTemplate(
    input_variables=['species'], template='Tell me about the a biggest {adjective} animal in the world.'
)

try:
    failed_chain = (
        prompt
        | RunnableLambda(lambda x: simulated_failure(x))
        | StrOutputParser()
    )
    res = failed_chain.invoke(EXAMPLE_INPUT)
except Exception as e:
    print(f'Failed to invoke the model: {e}')

chain = (
    prompt
    | LCELRetry(RunnableLambda(lambda x: simulated_failure(x)))
    | StrOutputParser()
)
res = chain.invoke(EXAMPLE_INPUT)
print(res)
