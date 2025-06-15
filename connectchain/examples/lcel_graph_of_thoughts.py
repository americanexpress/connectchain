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
An example of using LCEL for a graph of thoughts.
    a. generate titles for a children's story.
    b. the titles need to align with a few settings, such as
        story_elements
            theme                (e.g. love)
            characters           (e.g. animals)
            setting              (e.g. forest)
            plot                 (e.g. contest)
            conflict             (e.g. competition)
        tones                    (e.g. interesting, attractive)

The flow is as follows:
1. Ask for three summaries for a children's story, which should include the must_have.
2. Aggregate the three summaries to one summary.
3. Create four titles based on the aggregated summary, which should have the target tones.
4. Evaluate the four titles - whether they meet the requirements of the targeted tones.
5. Refine the titles to make them more tones, based on the evaluation in step 4.
"""
from operator import itemgetter

from dotenv import find_dotenv, load_dotenv
from langchain.schema import StrOutputParser

from connectchain.lcel import LCELLogger, model
from connectchain.prompts import ValidPromptTemplate

MODELS = {"GPT35": "1", "GPT4": "2", "DAVINCI": "3"}


class PrintLogger(LCELLogger):
    """Prints the payload to the console"""

    def log(self, payload):
        print(f'\n{"="*100}\n', payload)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    summary_prompt = ValidPromptTemplate(
        input_variables=["children_book", "story_elements"],
        template="Step 1 : "
        "Provide me three summaries for the children's book {children_book}. "
        "Each summary should include all the elements in the story_elements {story_elements}.",
    )

    summary_aggregation_prompt = ValidPromptTemplate(
        input_variables=["story_summaries"],
        template="Step 2 : "
        "Concatenate all the three story summaries {story_summaries} together to get one final summary. "
        "Remove any duplicated part.",
    )

    title_prompt = ValidPromptTemplate(
        input_variables=["summary_aggregation", "tones"],
        template="Step 3 : "
        "Create four distinct titles for each of the summary aggregations {summary_aggregation}. "
        "Make sure each title has the targeted tones {tones}.",
    )

    title_eval_prompt = ValidPromptTemplate(
        input_variables=["titles", "tones"],
        template="Step 4 : "
        "For each of the four titles {titles}, write out the title first, "
        "evaluate the title against the targeted tones {tones}, "
        "and assign an score out of 10 for the title. "
        "A higher score means that the title has more {tones} tones.",
    )

    refine_tone_prompt = ValidPromptTemplate(
        input_variables=["title_eval", "tones"],
        template="Step 5 : "
        "Based on the evaluation (title_eval) on the tones {tones} for the four titles, "
        "please refine each of the titles to make it more {tones}.",
    )

    model_name = "GPT35"
    model_parser = model(MODELS[model_name]) | StrOutputParser()
    model_parser_logger = model(MODELS[model_name]) | StrOutputParser() | PrintLogger().log()

    chain1 = summary_prompt | model_parser_logger
    chain2 = {"story_summaries": chain1} | summary_aggregation_prompt | model_parser_logger
    chain3 = (
        {"summary_aggregation": chain2, "tones": itemgetter("tones")}
        | title_prompt
        | model_parser_logger
    )
    chain4 = (
        {"titles": chain3, "tones": itemgetter("tones")} | title_eval_prompt | model_parser_logger
    )
    chain5 = (
        {"title_eval": chain4, "tones": itemgetter("tones")} | refine_tone_prompt | model_parser
    )

    with open("example_files/children_book.txt", "r", encoding="utf-8") as f:
        children_book = f.read()
    story_elements = ["theme", "characters", "setting", "plot", "conflict"]
    tones = ["interesting", "attractive"]

    output = chain4.invoke(
        {"children_book": children_book, "story_elements": story_elements, "tones": tones}
    )
    print(f"\n{'=' * 100}\n{model_name}:\n{output}\n{'=' * 100}")
