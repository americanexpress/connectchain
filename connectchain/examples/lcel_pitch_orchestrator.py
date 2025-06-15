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
Example of using LCEL for a more complex chaining. The flow is as follows:
1. Ask for 10 cooking recipes about a type of food.
2. Analyze the recipes and pick the one that is the easiest to follow by a teenager.
3. Re-word the recipe by using relevant information from a food description.
"""
from dotenv import find_dotenv, load_dotenv
from langchain.schema import StrOutputParser

from connectchain.lcel import LCELLogger, model
from connectchain.prompts import ValidPromptTemplate

if __name__ == "__main__":
    load_dotenv(find_dotenv())

    class PrintLogger(LCELLogger):
        """Prints the payload to the console"""

        def log(self, payload):
            print(payload)

    logger = PrintLogger()

    recipe_prompt = ValidPromptTemplate(
        input_variables=["food_type"],
        template="""
        Give me 10 recipes using {food_type}.
        """,
    )

    analysis_prompt = ValidPromptTemplate(
        input_variables=["recipes"],
        template="Given the recipes below, analyze each, and finally pick the one that, "
        "in your view, is the easiest to follow by a teenager. "
        "Reiterate the complete recipe as well: {recipes}",
    )

    summary_prompt = ValidPromptTemplate(
        input_variables=["selected_recipe", "food_description"],
        template="Re-word this pitch: {selected_recipe} by using relevant information "
        "from the following text: {food_description}",
    )

    with open("example_files/food_description.txt", "r", encoding="utf-8") as f:
        food_description = f.read()

    chain = (
        {"recipes": recipe_prompt | model() | StrOutputParser()}
        | analysis_prompt
        | logger.log()
        | model()
        | logger.log()
        | StrOutputParser()
    )

    selected_recipe = chain.invoke({"food_type": "salmon"})

    # # sleep 16 minutes
    # import time
    # time.sleep(20 * 60)

    chain = summary_prompt | logger.log() | model("3") | logger.log() | StrOutputParser()

    out = chain.invoke({"selected_recipe": selected_recipe, "food_description": food_description})

    print(out)
