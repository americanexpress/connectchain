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
import re
from unittest import TestCase

from connectchain.prompts import ValidPromptTemplate
from connectchain.lcel import Logger

class PrintLogger(Logger):
    """Prints the payload to the console"""
    def __init__(self):
        self.output = ""

    def print(self, payload):
        self.output = payload.text

    def sanitize_output(self, str):
        string = re.sub(r"^(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?$", "", payload)
        self.output = string

TEMPLATE = "Give me 10 different {animal_type} species."

prompt = ValidPromptTemplate(
    input_variables=["animal_type"],
    template=TEMPLATE
)


class TestLogger(TestCase):
    """This is the unit test for the lcel_logger"""
    def test_log(self):
        """Test the log method"""
        logger = PrintLogger()
        chain = prompt | logger.log()
        chain.invoke({"animal_type": "bird"})
        self.assertEqual(logger.output,
                         'Give me 10 different bird species.')
