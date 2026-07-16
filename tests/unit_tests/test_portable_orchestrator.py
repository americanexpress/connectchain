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
"""Unit testing for PortableOrchestrator class — BUG-3 regression suite"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from connectchain.chains import ValidLLMChain
from connectchain.orchestrators import PortableOrchestrator
from connectchain.prompts import ValidPromptTemplate

from .setup_utils import get_mock_config


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
    def test_build_and_model_with_default_llm(
        self, mock_get_token, *args
    ):  # pylint: disable=unused-argument
        """PortableOrchestrator can be built with the default LLM."""
        orchestrator = PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        self.assertEqual(orchestrator._is_lcel, False)  # pylint: disable=protected-access
        self.mock_from_env.assert_called_once()
        mock_get_token.assert_called_once()
        self.mock_chat_openai.assert_called_once()
        PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        self.mock_chat_openai.assert_called_once()  # second call uses cache

    @patch.dict(os.environ, {"CONFIG_PATH": "test_path"})
    def test_build_with_missing_config(self, *args):  # pylint: disable=unused-argument
        """Missing model index raises a descriptive exception."""
        with self.assertRaisesRegex(BaseException, 'Model config at index "gpt5" is not defined'):
            PortableOrchestrator.from_prompt_template("test_template", ["var1"], index="gpt5")

    @patch.dict(os.environ, {"CONFIG_PATH": "any_path"})
    def test_build_with_unsupported_llm(self, *args):  # pylint: disable=unused-argument
        """Unsupported model index raises a descriptive exception."""
        with self.assertRaisesRegex(BaseException, 'Model config at index "gpt5" is not defined'):
            PortableOrchestrator.from_prompt_template("test_template", ["var1"], index="gpt5")

    @patch.dict(os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"})
    @patch("connectchain.prompts.ValidPromptTemplate", return_value=Mock(ValidPromptTemplate))
    @patch("connectchain.lcel.model.SessionMap.uuid_from_config", return_value="TEST_MODEL_ENV")
    @patch("connectchain.lcel.model.get_token_from_env", return_value="test_token")
    def test_from_prompt_template_output_sanitizer_wiring(
        self, *args
    ):  # pylint: disable=unused-argument
        """CODE-REVIEW FOLLOWUP regression: from_prompt_template() used to
        hardcode output_sanitizer=None on the ValidLLMChain it builds, so the
        documented factory method had no way to attach a response sanitizer at
        all -- only prompt_sanitizer (the input side) was wired up. An
        output_sanitizer kwarg must now reach the built chain; omitting it must
        still default to None (unchanged behavior)."""

        def my_sanitizer(text: str) -> str:
            return f"[SANITIZED:{text}]"

        cases = [({"output_sanitizer": my_sanitizer}, my_sanitizer), ({}, None)]
        for extra_kwargs, expected in cases:
            with self.subTest(extra_kwargs=extra_kwargs):
                with patch("connectchain.chains.ValidLLMChain") as mock_chain_cls:
                    PortableOrchestrator.from_prompt_template(
                        "test_template", ["var1"], **extra_kwargs
                    )
                    mock_chain_cls.assert_called_once()
                    self.assertIs(mock_chain_cls.call_args.kwargs["output_sanitizer"], expected)

    # ── BUG-3 FIX: run_sync must call .invoke(), not deprecated .run() ──────

    @patch("connectchain.lcel.model.get_token_from_env", return_value="test_token")
    @patch("connectchain.prompts.ValidPromptTemplate", return_value=Mock(ValidPromptTemplate))
    @patch("connectchain.chains.ValidLLMChain", return_value=Mock(ValidLLMChain))
    @patch.dict(os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"})
    def test_run_sync_passes_query_unwrapped(self, *args):  # pylint: disable=unused-argument
        """CODE-REVIEW FOLLOWUP regression: run_sync() must pass query straight
        through to .invoke(), NOT wrapped in {"input": query}. A dict is used
        as-is by Chain.prep_inputs(), so the literal key "input" would have to
        match the prompt's declared input_variables -- which it essentially
        never does for a real (non-mocked) chain. Passing the bare value lets
        Chain.prep_inputs() map it onto the chain's real input key, exactly
        like the old (deprecated) .run(query) did."""
        orchestrator = PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        orchestrator._chain.invoke = Mock(
            return_value={"text": "invoke_response"}
        )  # pylint: disable=protected-access
        response = orchestrator.run_sync("test_query")
        orchestrator._chain.invoke.assert_called_once_with(
            "test_query"
        )  # pylint: disable=protected-access
        self.assertEqual(response, "invoke_response")

    def test_run_sync_against_real_chain_with_named_variable(self):
        """CODE-REVIEW FOLLOWUP regression: reproduces the exact failure from the
        README's own usage example against the REAL (non-mocked) ValidLLMChain/
        PromptTemplate classes. Before the fix, .invoke({"input": query}) raised
        `ValueError: Missing some input keys: {'area_of_interest'}` for any
        prompt whose variable isn't literally named "input"."""
        prompt = ValidPromptTemplate(
            output_sanitizer=lambda x: x,
            input_variables=["area_of_interest"],
            template="Tell me about the climate in {area_of_interest}.",
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="test-key")
        chain = ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=None)
        orchestrator = PortableOrchestrator(chain)
        with patch.object(ValidLLMChain, "invoke", return_value={"text": "It's warm."}):
            response = orchestrator.run_sync("Peru")
        self.assertEqual(response, "It's warm.")

    def test_run_sync_output_extraction(self):
        """run_sync() must extract the chain's actual output_key from the result
        dict -- including when the value is a legitimate falsy empty string.
        PR-7-FOLLOWUP regression: the prior `result.get("text") or
        result.get("output") or str(result)` treated an empty-string completion
        as if the key were absent, returning the stringified dict instead of
        the real (empty) response. CODE-REVIEW FOLLOWUP regression: a hardcoded
        "text"/"output" guess (instead of reading the chain's real output_key)
        silently mishandled any chain with a custom output_key."""
        prompt = PromptTemplate(input_variables=["q"], template="{q}")
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="test-key")
        cases = [
            ("text", {"text": "invoke_response"}, "invoke_response"),
            ("output", {"output": "output_key_response"}, "output_key_response"),
            ("text", {"text": ""}, ""),
        ]
        for output_key, return_value, expected in cases:
            with self.subTest(output_key=output_key):
                chain = ValidLLMChain(
                    llm=llm, prompt=prompt, output_sanitizer=None, output_key=output_key
                )
                orchestrator = PortableOrchestrator(chain)
                with patch.object(ValidLLMChain, "invoke", return_value=return_value):
                    response = orchestrator.run_sync("test_query")
                self.assertEqual(response, expected)

    def test_extract_output_warns_when_output_key_absent(self):
        """When the chain's output_key is genuinely absent from the result dict
        (e.g. a caller wraps a non-ValidLLMChain runnable whose output uses a
        different key -- this class documents that it "can wrap any third-party
        LLM framework"), _extract_output() falls back to str(result). That
        fallback must NOT be silent: it emits a logger.warning so the skip
        leaves a visible trace, mirroring ValidLLMChain._sanitize_dict()'s
        handling of the analogous key-not-found case (an unlogged skip there is
        a bypass with no visible trace)."""
        prompt = PromptTemplate(input_variables=["q"], template="{q}")
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="test-key")
        chain = ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=None, output_key="text")
        orchestrator = PortableOrchestrator(chain)
        result = {"unexpected_key": "value"}
        with patch.object(ValidLLMChain, "invoke", return_value=result):
            with self.assertLogs(
                "connectchain.orchestrators.portable_orchestrator", level="WARNING"
            ) as captured:
                response = orchestrator.run_sync("test_query")
        self.assertEqual(response, str(result))
        self.assertEqual(len(captured.records), 1)
        self.assertIn("text", captured.output[0])

    @patch("connectchain.lcel.model.get_token_from_env", return_value="test_token")
    @patch("connectchain.prompts.ValidPromptTemplate", return_value=Mock(ValidPromptTemplate))
    @patch("connectchain.chains.ValidLLMChain", return_value=Mock(ValidLLMChain))
    @patch.dict(os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"})
    def test_run_async_uses_ainvoke(self, *args):  # pylint: disable=unused-argument
        """async run() must call .ainvoke() with query passed through unwrapped,
        not deprecated .arun(), and not wrapped in {"input": query}."""
        orchestrator = PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        orchestrator._chain.ainvoke = AsyncMock(
            return_value={"text": "async_response"}
        )  # pylint: disable=protected-access
        response = asyncio.run(orchestrator.run("async_query"))
        orchestrator._chain.ainvoke.assert_called_once_with(
            "async_query"
        )  # pylint: disable=protected-access
        self.assertEqual(response, "async_response")
