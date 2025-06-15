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
"""Unit testing for PortableOrchestrator class"""
import os
import unittest
from unittest.mock import Mock, patch

from langchain.chat_models import ChatOpenAI

from connectchain.chains import ValidLLMChain
from connectchain.orchestrators import PortableOrchestrator
from connectchain.prompts import ValidPromptTemplate
from connectchain.test.setup_utils import get_mock_config


class TestPortableOrchestrator(unittest.TestCase):
    """Unit testing for PortableOrchestrator class"""

    def setUp(self) -> None:
        self.from_env_patcher = patch(
            "connectchain.utils.Config.from_env", return_value=get_mock_config()
        )
        self.mock_from_env = self.from_env_patcher.start()
        self.chat_openai_patcher = patch(
            "connectchain.lcel.model.ChatOpenAI", return_value=Mock(ChatOpenAI)
        )
        self.mock_chat_openai = self.chat_openai_patcher.start()

    @patch.dict(os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"})
    @patch("connectchain.prompts.ValidPromptTemplate", return_value=Mock(ValidPromptTemplate))
    @patch("connectchain.chains.ValidLLMChain", return_value=Mock(ValidLLMChain))
    @patch("connectchain.lcel.model.SessionMap.uuid_from_config", return_value="TEST_MODEL_ENV")
    @patch("connectchain.lcel.model.get_token_from_env", return_value="test_token")
    # pylint: disable=unused-argument
    def test_build_and_model_with_default_llm(self, mock_get_token, *args):
        """Test that a PortableOrchestrator instance can be built with the default LLM"""
        orchestrator = PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        # pylint: disable=protected-access
        self.assertEqual(orchestrator._is_lcel, False)
        self.mock_from_env.assert_called_once()
        mock_get_token.assert_called_once()
        self.mock_chat_openai.assert_called_once()
        PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        # second call should not invoke ChatOpenAI as it takes it
        # from cache. hence still only called once
        self.mock_chat_openai.assert_called_once()

    @patch.dict(os.environ, {"CONFIG_PATH": "test_path"})
    def test_build_with_missing_config(self, *args):  # pylint: disable=unused-argument
        """Test that an exception is raised when an unsupported LLM is passed in"""
        with self.assertRaisesRegex(
            BaseException, 'Model config at index "gpt5" is not defined'
        ) as _:
            PortableOrchestrator.from_prompt_template("test_template", ["var1"], index="gpt5")

    @patch.dict(os.environ, {"CONFIG_PATH": "any_path"})
    def test_build_with_unsupported_llm(self, *args):  # pylint: disable=unused-argument
        """Test that an exception is raised when an unsupported LLM is passed in"""
        with self.assertRaisesRegex(
            BaseException, 'Model config at index "gpt5" is not defined'
        ) as _:
            PortableOrchestrator.from_prompt_template("test_template", ["var1"], index="gpt5")

    @patch("connectchain.lcel.model.get_token_from_env", return_value="test_token")
    @patch("connectchain.prompts.ValidPromptTemplate", return_value=Mock(ValidPromptTemplate))
    @patch("connectchain.chains.ValidLLMChain", return_value=Mock(ValidLLMChain))
    @patch.dict(os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"})
    # pylint: disable=unused-argument
    def test_run_sync(self, *args):
        """Test that a PortableOrchestrator instance can be built with the default model config index"""
        orchestrator = PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        # pylint: disable=protected-access
        orchestrator._chain.run.return_value = "test_response"
        response = orchestrator.run_sync("test_query")
        self.assertEqual(response, "test_response")
