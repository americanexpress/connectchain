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
Example of using LCEL for a tree of thoughts. The flow is as follows:
1. Ask for three distinct ideas for a hand-made gift for a five-year-old boy, considering some key factors.
2. Evaluate the potential against the key factors for each idea, and select the best one only.
3. For the selected ideas, provide two distinct solutions for how to hand-made it.
4. Evaluate each solution considering resources, difficulties, cost, time, risks and mitigations. Pick the best one only
"""
from dotenv import load_dotenv, find_dotenv
from langchain.schema import StrOutputParser
from operator import itemgetter
from connectchain.prompts import ValidPromptTemplate
from connectchain.lcel import model, Logger

MODELS = {"GPT35": "1", "GPT4": "2", "DAVINCI": "3"}


class PrintLogger(Logger):
    """Prints the payload to the console"""

    def print(self, payload):
        print(f'\n{"="*100}\n', payload)


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    idea_prompt = ValidPromptTemplate(
        input_variables=["action", "factors"],
        template="Step 1 : "
                 "I need to {action}. Could you brainstorm three distinct ideas? "
                 "Please consider a variety of factors such as {factors}."
    )

    idea_eval_prompt = ValidPromptTemplate(
        input_variables=["ideas", "factors"],
        template="Step 2 : "
                 "Evaluate each of the three proposed ideas {ideas} against the factors {factors}. "
                 "Assign an overall score to each idea based on the evaluation. "
                 "A higher score means that the idea aligns better with the factors {factors}. "
                 "Select one idea that have the highest score. Only output the selected idea."
    )

    solution_prompt = ValidPromptTemplate(
        input_variables=["selected_idea"],
        template="Step 3 : "
                 "For the selected idea {selected_idea}, provide two distinct solutions for how to hand-made it. "
                 "Provide step-by-step instructions for each solution."
    )

    solution_eval_prompt = ValidPromptTemplate(
        input_variables=["solutions"],
        template="Step 4 : "
                 "Evaluate each of the solutions {solutions} considering "
                 "resources, difficulties, cost, time, risks and mitigations. "
                 "Assign an overall score to each solution based on the evaluation. "
                 "A higher score means easier to access the resources needed, less difficult to implement, "
                 "lower cost, less time-consuming, less risky, and less likely to encounter problems. "
                 "Select one solution that have the highest score. Output the selected solution."
    )

    model_name = "GPT35"
    model_parser = model(MODELS[model_name]) | StrOutputParser()
    model_parser_logger = model(MODELS[model_name]) | StrOutputParser() | PrintLogger().log()

    chain1 = idea_prompt | model_parser_logger
    chain2 = {"ideas": chain1, "factors": itemgetter("factors")} | idea_eval_prompt | model_parser_logger
    chain3 = {"selected_idea": chain2} | solution_prompt | model_parser_logger
    chain4 = {"solutions": chain3} | solution_eval_prompt | model_parser

    action = "hand-make a birthday gift for a five-year-old boy"
    factors = "playfulness, safety, durability"

    output = chain4.invoke({"action": action, "factors": factors})
    print(f"\n{'='*100}\n{model_name}:\n{output}\n{'='*100}")
