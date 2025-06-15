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
"""This is the unit test for the lcel_logger"""
from unittest import TestCase

from connectchain.lcel import LCELLogger, PrintLogger
from connectchain.prompts import ValidPromptTemplate


class SelfLogger(LCELLogger):
    """Sets the target log as a SelfLogger attribute for testing"""

    def __init__(self):
        self.output = ""

    def log(self, payload):
        self.output = payload.text


TEMPLATE = "Give me 10 different {animal_type} species."

prompt = ValidPromptTemplate(input_variables=["animal_type"], template=TEMPLATE)


class TestLogger(TestCase):
    """This is the unit test for the LCEL logger"""

    def test_lcellogger(self):
        logger = SelfLogger()
        chain = prompt | logger
        chain.invoke({"animal_type": "bird"})
        self.assertEqual(logger.output, "Give me 10 different bird species.")

    def test_printlogger(self):
        logger = PrintLogger()
        chain = prompt | logger
        out = chain.invoke({"animal_type": "bird"})
        self.assertEqual(out.text, "Give me 10 different bird species.")
