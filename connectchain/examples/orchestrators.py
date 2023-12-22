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
Example of using the PortableOrchestrator.
"""

from dotenv import load_dotenv, find_dotenv
from connectchain.orchestrators import PortableOrchestrator

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    orchestrator = PortableOrchestrator.from_prompt_template(
        prompt_template="Tell me about the {bird_type} family of birds.",
        input_variables=["bird_type"])

    for bird_type in ['Flycatcher', 'Flicker', 'Thrasher']:
        output = orchestrator.run_sync(bird_type)
        print(f'Response for {bird_type}:\n{output}\n')

    orchestrator = PortableOrchestrator.from_prompt_template(
        prompt_template="Write me a poem about birds, specifically the {bird_type}.",
        input_variables=["bird_type"]
    )

    output = orchestrator.run_sync('Cactus Wren')

    print(f'Response for bird poem:\n{output}\n')

    orchestrator = PortableOrchestrator.from_prompt_template(
        prompt_template="Translate this poem into {language}: {poem}",
        input_variables=["language", "poem"],
        index="2" # e.g. another model
    )

    output = orchestrator.run_sync({ "poem": output, "language": "Shakesperian English" })

    print(f'16th century bird poem:\n{output}')
