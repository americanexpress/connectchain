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
"""Example of using OpenAI API with Connectchain."""
import openai
from dotenv import find_dotenv, load_dotenv

from connectchain.utils import Config, get_token_from_env

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    auth_token = get_token_from_env()
    model_config = Config.from_env().models["1"]

    openai.api_type = "azure"
    openai.api_version = model_config.api_version
    openai.api_base = model_config.api_base
    openai.azure_deployment_name = model_config.engine
    openai.azure_model_name = model_config.model_name
    openai.api_key = auth_token
    out = openai.ChatCompletion.create(
        engine=model_config.engine,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What is the primary habitat of the Melozone aberti (Abert's Towhee)?",
            },
            {
                "role": "assistant",
                "content": "The Abert's Towhee lives in the Sonoran Desert, preferring dense brush along rivers and streams.",
            },
            {"role": "user", "content": "Do Abert's Towhee migrate?"},
        ],
    )

    print(out)
