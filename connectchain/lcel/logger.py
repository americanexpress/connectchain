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
import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain.schema.runnable import RunnableLambda


class LCELLogger(ABC):
    """This is the class that defines the interface for the logger component"""

    @abstractmethod
    def log(self, payload: Any) -> None:
        """Send the payload to the desired component"""

    def _log(self) -> RunnableLambda:
        """Return a lambda function that logs the payload"""

        def _log_helper(payload: Any) -> Any:
            self.log(payload)
            return payload

        return RunnableLambda(_log_helper)

    def __call__(self, payload: Any) -> RunnableLambda:
        return self._log()


class PrintLogger(LCELLogger):
    """This class displays the payload in the console"""

    def log(self, payload: Any) -> None:
        print(payload)


# Example class to extend/implement for support of enterprise logging frameworks.
'''
class EnterpriseLogger(LCELLogger):
    """Enterprise LCEL Logger"""

    def __init__(self):
        # Pass enterprise log handler
        pass

    def log(self, payload):
        self.info(payload)

    def info(self, payload):
        logger = self._get_logger(logging.INFO)
        logger.info(payload)

    def warning(self, payload):
        logger = self._get_logger(logging.WARNING)
        logger.warning(payload)

    def error(self, payload):
        logger = self._get_logger(logging.ERROR)
        logger.error(payload)

    def critical(self, payload):
        logger = self._get_logger(logging.CRITICAL)
        logger.critical(payload)

    def _get_logger(self, level=logging.INFO) -> logging.Logger:
        logger = logging.getLogger(__name__)

        logger.setLevel(level)

        # logger.addHandler(...)
        logger.propagate = True
        return logger
'''
