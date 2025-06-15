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
"""Example of using retry decorator."""
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from connectchain.lcel import model
from connectchain.utils import retry_decorator

load_dotenv(find_dotenv())

n_failure = 0


@retry_decorator()
def simulated_failure(input):
    global n_failure
    n_failure += 1
    if n_failure < 2:
        raise Exception("Simulated failure")
    prompt = PromptTemplate(
        input_variables=["species"], template="What is your favorite {species}?"
    )
    return (prompt | model("2") | StrOutputParser()).invoke(input)


res = None

while not res:
    res = simulated_failure({"species": "mammal"})

print(res)
