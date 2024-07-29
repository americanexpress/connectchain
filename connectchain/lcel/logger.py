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
"""This module defines the interface for the logger component"""
from abc import ABC, abstractmethod
from langchain.schema.runnable import RunnableLambda


class Logger(ABC):
    """This is the class that defines the interface for the logger component"""
    @abstractmethod
    def print(self, payload):
        """Send the payload to the desired component"""

    @abstractmethod
    def sanitize_output(self, payload):
        """Sanitize the payload to remove sensitive information"""

    def log(self):
        """Return a lambda function that logs the payload"""

        def _log_helper(payload: any):
            self.sanitize_output(payload)
            self.print(payload)
            return payload

        return RunnableLambda(_log_helper)
