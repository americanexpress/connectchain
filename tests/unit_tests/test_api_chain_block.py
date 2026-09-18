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
"""Unit testing for the APIChain security block and its exception type"""
import unittest

from langchain.chains.api.base import APIChain

import connectchain  # noqa: F401  pylint: disable=unused-import
from connectchain.utils.exceptions import ConnectChainNoAccessException


class TestAPIChainBlock(unittest.TestCase):
    """ConnectChainNoAccessException must not be catchable by a generic except Exception,
    since it enforces a deliberate security block on APIChain."""

    def test_not_a_subclass_of_exception(self):
        """The security block must survive an ordinary except Exception around it"""
        self.assertFalse(issubclass(ConnectChainNoAccessException, Exception))
        self.assertTrue(issubclass(ConnectChainNoAccessException, BaseException))

    def test_api_chain_raises_and_is_not_swallowed_by_except_exception(self):
        """Simulates application code wrapping a chain call in except Exception"""
        with self.assertRaises(ConnectChainNoAccessException):
            try:
                APIChain.run(None, "some query")
            except Exception:  # pylint: disable=broad-except
                self.fail(
                    "ConnectChainNoAccessException must not be caught by except Exception"
                )
