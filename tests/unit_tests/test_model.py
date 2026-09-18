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

from langchain_openai import AzureOpenAI, ChatOpenAI

from connectchain.lcel import LCELModelException, model
from connectchain.lcel.model import LCELModelConfigError, _get_direct_model_
from connectchain.utils import SessionMap
from connectchain.utils.exceptions import NonRetryableError

from .setup_utils import get_mock_config, wrap_model_config


class TestModel(unittest.TestCase):
    """Unit testing the model LCEL method"""

    def setUpWithConfig(self, mock_config):
        """Set up the test with a mock config"""
        patcher_env = patch.dict(
            os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"}
        )
        patcher_config = patch("connectchain.utils.Config.from_env", return_value=mock_config)
        patcher_token = patch(
            "connectchain.lcel.model.get_token_from_env", return_value="test_token"
        )
        patcher_uuid = patch(
            "connectchain.lcel.model.SessionMap.uuid_from_config", return_value="TEST_MODEL_ENV"
        )

        return {
            "env": patcher_env.start(),
            "config": patcher_config.start(),
            "token": patcher_token.start(),
            "uuid": patcher_uuid.start(),
        }

    def tearDown(self):
        """Tear down the test"""
        patch.stopall()

    @patch("connectchain.lcel.model.ChatOpenAI", return_value=Mock(ChatOpenAI))
    # pylint: disable=unused-argument
    def test_model_with_default_llm(self, *args):
        self.setUpWithConfig(get_mock_config())
        test_model = model()
        self.assertIsInstance(test_model, ChatOpenAI)
        test_token = os.getenv("TEST_MODEL_ENV")
        self.assertEqual(test_token, "test_token")

    @patch("connectchain.lcel.model.AzureOpenAI", return_value=Mock(AzureOpenAI))
    # pylint: disable=unused-argument
    def test_model_with_defined_llm(self, *args):
        self.setUpWithConfig(get_mock_config())
        test_model = model("2")
        self.assertIsInstance(test_model, AzureOpenAI)
        test_token = os.getenv("TEST_MODEL_ENV")
        self.assertEqual(test_token, "test_token")

    def test_model_without_eas_section_routes_direct(self):
        """A config with NO eas section at all must route to the direct
        (non-EAS) path instead of crashing: now that Config raises
        AttributeError for missing top-level keys, the routing check probes
        the section via getattr defaults, which genuinely work."""
        test_config = get_mock_config()
        del test_config.data["eas"]
        mocks = self.setUpWithConfig(test_config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            test_model = model()
        self.assertIsInstance(test_model, ChatOpenAI)
        mocks["token"].assert_not_called()

    def test_model_with_empty_eas_section_routes_direct(self):
        """A bare `eas:` line in YAML parses to None. Attribute access on the
        wrapped None section raises AttributeError (formerly TypeError, which
        blew through the routing check's except clause), so the getattr-based
        probe must degrade to direct routing."""
        test_config = get_mock_config()
        test_config.data["eas"] = None
        mocks = self.setUpWithConfig(test_config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            test_model = model()
        self.assertIsInstance(test_model, ChatOpenAI)
        mocks["token"].assert_not_called()

    def test_model_with_eas_section_missing_id_key_routes_direct(self):
        """An eas section without an id_key cannot authenticate through EAS, so
        the model must route direct -- getattr's default (now honored for
        missing ConfigWrapper keys) is what makes this presence check work."""
        test_config = get_mock_config()
        test_config.data["eas"] = {"secret_key": "secret_key"}
        mocks = self.setUpWithConfig(test_config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            test_model = model()
        self.assertIsInstance(test_model, ChatOpenAI)
        mocks["token"].assert_not_called()

    def test_model_with_no_models_configured(self):
        test_config = get_mock_config()
        del test_config.data["models"]
        self.setUpWithConfig(test_config)
        with self.assertRaisesRegex(LCELModelException, "No models defined in config") as _:
            test_model = model()

    def test_model_with_undefined_llm(self):
        self.setUpWithConfig(get_mock_config())
        with self.assertRaisesRegex(
            LCELModelException, 'Model config at index "gpt5" is not defined'
        ) as _:
            test_model = model("gpt5")

    def test_model_with_unsupported_provider(self):
        """SYSTEMATIC-DEBUGGING FIX: unsupported provider must fail fast with the
        'not supported' message, not a misleading 'API key not found' error from
        an unrelated env var that the caller was never going to set. This also
        fixes a stale assertion: the actual message has always said 'not
        supported', never 'Not implemented' — the regex could not have matched
        even if the ordering bug were absent."""
        test_config = get_mock_config()
        # required to not modify dict instance
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        test_config.data["models"]["1"]["provider"] = "meta"
        self.setUpWithConfig(test_config)
        with self.assertRaisesRegex(LCELModelException, "not supported") as _:
            test_model = model()

    @patch("langchain.chat_models.init_chat_model")
    def test_get_direct_model_reraises_unexpected_exception(self, mock_init_chat_model):
        """BUG-4 regression: an unexpected exception from init_chat_model() (anything
        other than ImportError/ValueError) must be re-raised as LCELModelException with
        the original exception preserved via `from e`, not silently swallowed by a bare
        `except: pass` and left to fall through to manual init."""
        mock_init_chat_model.side_effect = RuntimeError("boom: unexpected failure")
        model_config = wrap_model_config(get_mock_config().data["models"]["1"])
        with self.assertRaisesRegex(
            LCELModelException, "Unexpected error initialising model"
        ) as cm:
            _get_direct_model_(model_config)
        # confirm the original exception is chained, not lost
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)
        # CODE-REVIEW regression: the wrapped error's retryability is unknown (it
        # could be a transient network/provider failure), so it must be the plain
        # base class -- NOT NonRetryableError-marked -- leaving retry wrappers free
        # to retry it.
        self.assertNotIsInstance(cm.exception, NonRetryableError)

    @patch("langchain.chat_models.init_chat_model")
    def test_get_direct_model_falls_through_on_expected_exceptions(self, mock_init_chat_model):
        """BUG-4 regression: expected ImportError/ValueError from init_chat_model() must
        still fall through to manual provider init (not be re-raised)."""
        mock_init_chat_model.side_effect = ValueError("Unable to infer model provider")
        model_config = wrap_model_config(get_mock_config().data["models"]["1"])
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _get_direct_model_(model_config)
        self.assertIsInstance(result, ChatOpenAI)

    def test_get_direct_model_safe_message_when_model_name_raises(self):
        """If the original failure inside _get_direct_model_'s try block was itself
        model_config.model_name raising AttributeError (a malformed config missing
        that attribute), the except-Exception handler's error message must not
        re-access that same attribute -- doing so would raise a second, uncaught
        AttributeError instead of the intended clean LCELModelException."""

        class _ModelNameRaises:
            provider = "openai"

            @property
            def model_name(self):
                raise AttributeError("model_name is not available")

        with self.assertRaisesRegex(
            LCELModelException, "Unexpected error initialising model"
        ) as cm:
            _get_direct_model_(_ModelNameRaises())
        self.assertIsInstance(cm.exception.__cause__, AttributeError)

    @patch("connectchain.lcel.model.logger")
    def test_unsupported_provider_does_not_log_fallback_warning(self, mock_logger):
        """A genuinely unknown provider must fail with the clear 'not supported' error
        WITHOUT first emitting the misleading 'falling back to manual provider init'
        warning -- there is no manual branch for it, so there is nothing to fall back
        to. Uses the REAL init_chat_model(): the provider name is passed through to it
        explicitly and its 'Unsupported model_provider' ValueError is what gets
        re-raised as the not-supported LCELModelConfigError."""
        model_config = wrap_model_config(
            {**get_mock_config().data["models"]["1"], "provider": "meta"}
        )
        with self.assertRaisesRegex(LCELModelConfigError, "not supported") as cm:
            _get_direct_model_(model_config)
        # The chained cause is init_chat_model()'s own rejection of the provider.
        self.assertIsInstance(cm.exception.__cause__, ValueError)
        mock_logger.warning.assert_not_called()

    @patch("langchain.chat_models.init_chat_model")
    def test_provider_without_manual_branch_reaches_init_chat_model_fast_path(
        self, mock_init_chat_model
    ):
        """CODE-REVIEW regression: providers outside the manual fallback set
        (mistralai, groq, ollama, bedrock, ...) are fully served by the
        init_chat_model() fast path and must NOT be pre-emptively rejected by the
        _SUPPORTED_PROVIDERS check -- that gate exists only for the manual fallback.
        The configured provider must be passed to init_chat_model() explicitly
        (model_provider=...) so a bogus provider still fails clearly there instead
        of silently constructing whatever model the name happens to infer to."""
        model_config = wrap_model_config(
            {"provider": "mistralai", "model_name": "mistral-large-latest"}
        )
        result = _get_direct_model_(model_config)
        self.assertIs(result, mock_init_chat_model.return_value)
        self.assertEqual(
            mock_init_chat_model.call_args.kwargs["model_provider"], "mistralai"
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_manual_set_provider_does_not_pass_model_provider(self, mock_init_chat_model):
        """Providers in the manual fallback set keep the baseline inference-based
        init_chat_model() call (no model_provider kwarg): e.g. 'google' is a valid
        connectchain provider but NOT a valid init_chat_model provider id
        (google_genai/google_vertexai), so passing it through would break a
        previously working path."""
        model_config = wrap_model_config({"provider": "openai", "model_name": "gpt-4"})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            _get_direct_model_(model_config)
        self.assertNotIn("model_provider", mock_init_chat_model.call_args.kwargs)

    def test_exception_hierarchy_retryability_split(self):
        """CODE-REVIEW regression: LCELModelException must be plain (retry-eligible)
        because it also wraps UNEXPECTED init failures of unknown retryability;
        only LCELModelConfigError -- raised for known-permanent config problems --
        carries the NonRetryableError marker. The subclass relationship keeps
        `except LCELModelException` catching both."""
        self.assertFalse(issubclass(LCELModelException, NonRetryableError))
        self.assertTrue(issubclass(LCELModelConfigError, LCELModelException))
        self.assertTrue(issubclass(LCELModelConfigError, NonRetryableError))

    def test_permanent_config_errors_are_non_retryable(self):
        """Representative known-permanent raise sites must use LCELModelConfigError
        (NonRetryableError) so retry wrappers fail fast on them."""
        # Unsupported provider.
        with self.assertRaises(LCELModelConfigError) as cm:
            _get_direct_model_(
                wrap_model_config({"provider": "meta", "model_name": "test_model"})
            )
        self.assertIsInstance(cm.exception, NonRetryableError)
        # Azure endpoint with missing api_version.
        with self.assertRaises(LCELModelConfigError) as cm:
            _get_direct_model_(
                wrap_model_config(
                    {
                        "provider": "openai",
                        "model_name": "gpt-4",
                        "api_base": "https://my-resource.openai.azure.com/",
                    }
                )
            )
        self.assertIsInstance(cm.exception, NonRetryableError)

    def test_model_azure_endpoint_without_api_version_raises(self):
        """An Azure-shaped api_base without api_version must fail loudly instead of
        silently falling back to a non-Azure ChatOpenAI client pointed at Azure."""
        test_config = get_mock_config()
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        test_config.data["models"]["1"]["bypass_eas"] = True
        test_config.data["models"]["1"]["api_base"] = "https://my-resource.openai.azure.com/"
        del test_config.data["models"]["1"]["api_version"]
        self.setUpWithConfig(test_config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
                test_model = model()

    def test_model_entrypoint_azure_realistic_model_name_without_api_version_raises(self):
        """Regression test: a realistic model_name (e.g. "gpt-4") makes
        langchain.chat_models.init_chat_model() succeed and return early on the fast
        path, so the api_version guard must fire BEFORE that path is even attempted --
        not only in the manual fallback branch reached when the fast path fails (which
        is all test_model_azure_endpoint_without_api_version_raises above exercised,
        via the fixture's non-standard model_name="test_model").

        This variant drives the full model() entry point (including bypass_eas
        routing into _get_direct_model_); its sibling
        test_model_azure_endpoint_realistic_model_name_without_api_version_raises
        below calls _get_direct_model_ directly. (Both previously shared one name,
        so Python silently discarded this one -- a dead test.)"""
        test_config = get_mock_config()
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        test_config.data["models"]["1"]["bypass_eas"] = True
        test_config.data["models"]["1"]["model_name"] = "gpt-4"
        test_config.data["models"]["1"]["api_base"] = "https://my-resource.openai.azure.com/"
        del test_config.data["models"]["1"]["api_version"]
        self.setUpWithConfig(test_config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
                test_model = model()

    def test_use_of_session_map(self):
        self.setUpWithConfig(get_mock_config())
        test_model = model()
        test_model2 = model()
        self.assertIs(test_model, test_model2)

    @patch("connectchain.lcel.model.ChatOpenAI", return_value=Mock(ChatOpenAI))
    # pylint: disable=unused-argument
    def test_unset_token_refresh_interval_resolved_before_network_call(self, *args):
        """CODE-REVIEW regression: when config.eas.token_refresh_interval is unset
        (the model() path resolves the missing key to None via getattr's default),
        the expiry interval must be
        resolved to SessionMap.DEFAULT_EXPIRES_IN at capture time -- BEFORE
        get_token_from_env()'s network call -- and passed to new_session() as an
        explicit int. Passing None through would make new_session() fall back to the
        singleton's CURRENT expires_in after the network call, inheriting whatever
        interval a concurrent request configured last: the exact race the
        capture-before-I/O comment in _get_openai_model_ claims to close."""
        test_config = get_mock_config()
        test_config.data["eas"] = {**test_config.data["eas"]}
        del test_config.data["eas"]["token_refresh_interval"]
        mocks = self.setUpWithConfig(test_config)

        # Fresh singleton state for this test; reset again afterwards so the
        # class-level singleton/cache never leaks into other tests.
        def _reset_singleton():
            SessionMap._instance = None  # pylint: disable=protected-access
            SessionMap.session_map.clear()

        _reset_singleton()
        self.addCleanup(_reset_singleton)

        def token_with_concurrent_reconfig(_index):
            # Simulate another request reconfiguring the singleton for a different
            # model's (much shorter) interval while this request is mid-network-call.
            SessionMap(5)
            return "test_token"

        mocks["token"].side_effect = token_with_concurrent_reconfig
        model()
        cached_expires_in = SessionMap.session_map["TEST_MODEL_ENV"][1]
        self.assertEqual(cached_expires_in, SessionMap.DEFAULT_EXPIRES_IN)

    def test_direct_azure_model_does_not_supply_api_version_twice(self):
        """Regression test: _get_direct_model_'s Azure branch must construct a real
        (non-mocked) AzureOpenAI without pydantic rejecting api_version as supplied
        twice (once as a direct kwarg, once inside model_kwargs)."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_base": "https://my-resource.openai.azure.com/",
                "api_version": "2024-02-01",
                "engine": "gpt-4",
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _get_direct_model_(model_config)
        self.assertIsInstance(result, AzureOpenAI)

    def test_model_azure_endpoint_realistic_model_name_without_api_version_raises(self):
        """Regression test: a realistic model_name (e.g. "gpt-4") makes
        langchain.chat_models.init_chat_model() succeed and return early on the fast
        path, so the api_version guard must fire BEFORE that path is even attempted --
        not only in the manual fallback branch reached when the fast path fails."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_base": "https://my-resource.openai.azure.com/",
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
                _get_direct_model_(model_config)

    def test_sovereign_cloud_azure_endpoint_detected(self):
        """CODE-REVIEW regression: Azure US Government / China endpoints
        (openai.azure.us, openai.azure.cn) must route to the Azure builder like the
        public cloud, not silently fall through to a plain ChatOpenAI."""
        for host in ("openai.azure.us", "openai.azure.cn"):
            with self.subTest(host=host):
                model_config = wrap_model_config(
                    {
                        "provider": "openai",
                        "model_name": "gpt-4",
                        "api_base": f"https://my-resource.{host}/",
                    }
                )
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    # Reaching the api_version guard proves Azure routing happened.
                    with self.assertRaisesRegex(LCELModelException, "api_version is required"):
                        _get_direct_model_(model_config)

    def test_explicit_azure_flag_forces_azure_routing(self):
        """CODE-REVIEW regression: APIM/custom domains can't be detected by hostname;
        `azure: true` on the model config must force Azure routing explicitly."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_base": "https://llm-gw.mycorp.example/",
                "azure": True,
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with self.assertRaisesRegex(LCELModelException, "api_version is required"):
                _get_direct_model_(model_config)

    def test_azure_lookalike_urls_are_not_misrouted_to_azure(self):
        """CODE-REVIEW regression: Azure detection must suffix-match the parsed
        HOSTNAME, not substring-match the whole URL. A URL carrying an Azure marker
        in its path, or a lookalike host merely containing a marker, must build a
        plain ChatOpenAI -- misrouting to the Azure builder would surface here as
        the 'api_version is required' guard firing (no api_version is configured)."""
        for api_base in (
            "https://myproxy.corp.com/openai.azure.com-compat",
            "https://notopenai.azure.com.evil.example/v1",
        ):
            with self.subTest(api_base=api_base):
                model_config = wrap_model_config(
                    {
                        "provider": "openai",
                        "model_name": "gpt-4",
                        "api_base": api_base,
                    }
                )
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    result = _get_direct_model_(model_config)
                self.assertIsInstance(result, ChatOpenAI)

    def test_schemeless_azure_api_base_still_detected(self):
        """A scheme-less api_base on a real Azure domain must still route to the
        Azure builder: urlparse() puts a scheme-less value entirely in .path
        (hostname=None), so _is_azure_endpoint_ re-parses it network-relative
        rather than silently classifying it as non-Azure. Reaching the
        api_version guard proves Azure routing happened."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_base": "my-resource.openai.azure.com/",
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with self.assertRaisesRegex(LCELModelException, "api_version is required"):
                _get_direct_model_(model_config)

    @patch("connectchain.lcel.model.logger")
    @patch("langchain.chat_models.init_chat_model")
    def test_api_key_env_unset_warns_and_falls_back_to_default_env_on_fast_path(
        self, mock_init_chat_model, mock_logger
    ):
        """CODE-REVIEW regression: a config naming api_key_env whose env var is unset
        must NOT fail hard on the init_chat_model fast path -- the baseline silently
        omitted api_key and let init_chat_model resolve the provider's default env var
        (e.g. OPENAI_API_KEY), which can work. The misconfiguration is surfaced via a
        logger.warning naming the unset variable, then init proceeds WITHOUT an
        explicit api_key so the provider-default resolution applies."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key_env": "MY_CUSTOM_KEY_VAR_THAT_IS_UNSET",
            }
        )
        env = {k: v for k, v in os.environ.items() if k != "MY_CUSTOM_KEY_VAR_THAT_IS_UNSET"}
        with patch.dict(os.environ, env, clear=True):
            result = _get_direct_model_(model_config)
        self.assertIs(result, mock_init_chat_model.return_value)
        self.assertNotIn("api_key", mock_init_chat_model.call_args.kwargs)
        mock_logger.warning.assert_called_once()
        self.assertTrue(
            any(
                "MY_CUSTOM_KEY_VAR_THAT_IS_UNSET" in str(arg)
                for arg in mock_logger.warning.call_args[0]
            ),
            "warning must name the unset env var",
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_temperature_forwarded_on_fast_path(self, mock_init_chat_model):
        """Companion to the manual-fallback temperature tests: the init_chat_model()
        fast path builds its temperature kwarg through the shared
        _temperature_kwargs_() helper (it used to hand-roll the same logic), so a
        configured temperature must be forwarded..."""
        model_config = wrap_model_config(
            {"provider": "openai", "model_name": "gpt-4", "temperature": 0.25}
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            _get_direct_model_(model_config)
        self.assertEqual(mock_init_chat_model.call_args.kwargs["temperature"], 0.25)

    @patch("langchain.chat_models.init_chat_model")
    def test_unset_temperature_not_forced_on_fast_path(self, mock_init_chat_model):
        """...and an unset temperature must be omitted entirely (each provider keeps
        its own default) rather than forced to None."""
        model_config = wrap_model_config({"provider": "openai", "model_name": "gpt-4"})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            _get_direct_model_(model_config)
        self.assertNotIn("temperature", mock_init_chat_model.call_args.kwargs)

    def test_temperature_honored_on_manual_fallback_path(self):
        """CODE-REVIEW regression: configured temperature was only honored on the
        init_chat_model fast path; the manual fallback ChatOpenAI dropped it."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                # Non-inferable name forces the manual fallback branch.
                "model_name": "test_model",
                "temperature": 0.25,
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _get_direct_model_(model_config)
        self.assertIsInstance(result, ChatOpenAI)
        self.assertEqual(result.temperature, 0.25)

    def test_temperature_honored_on_huggingface_fallback(self):
        """CODE-REVIEW regression: the configured-temperature fix
        (**_temperature_kwargs_()) was applied to the other provider constructors but
        the huggingface branch (HuggingFaceEndpoint) was missed, silently dropping a
        configured temperature. langchain_huggingface may not be installed in the
        test environment, so the import is satisfied via a sys.modules mock."""
        import sys  # pylint: disable=import-outside-toplevel

        mock_endpoint_cls = Mock(name="HuggingFaceEndpoint")
        mock_module = Mock(HuggingFaceEndpoint=mock_endpoint_cls)
        model_config = wrap_model_config(
            {
                "provider": "huggingface",
                # Non-inferable name forces the manual fallback branch.
                "model_name": "test-org/test-repo",
                "temperature": 0.25,
            }
        )
        with patch.dict(sys.modules, {"langchain_huggingface": mock_module}):
            with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "hf-test-key"}):
                result = _get_direct_model_(model_config)
        self.assertIs(result, mock_endpoint_cls.return_value)
        self.assertEqual(mock_endpoint_cls.call_args.kwargs["temperature"], 0.25)

    def test_unset_temperature_not_forced_on_huggingface_fallback(self):
        """Companion to the above: when no temperature is configured, the huggingface
        branch must not pass temperature at all (each provider keeps its own default,
        per _temperature_kwargs_) rather than forcing temperature=None."""
        import sys  # pylint: disable=import-outside-toplevel

        mock_endpoint_cls = Mock(name="HuggingFaceEndpoint")
        mock_module = Mock(HuggingFaceEndpoint=mock_endpoint_cls)
        model_config = wrap_model_config(
            {"provider": "huggingface", "model_name": "test-org/test-repo"}
        )
        with patch.dict(sys.modules, {"langchain_huggingface": mock_module}):
            with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "hf-test-key"}):
                _get_direct_model_(model_config)
        self.assertNotIn("temperature", mock_endpoint_cls.call_args.kwargs)

    def test_unquoted_yaml_date_api_version_is_coerced_to_string(self):
        """VERIFY finding: YAML parses an unquoted `api_version: 2024-02-01` as a
        datetime.date, which AzureOpenAI's pydantic validation rejects opaquely.
        The api_version guard must coerce it so the natural config spelling works."""
        import datetime  # pylint: disable=import-outside-toplevel

        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_base": "https://my-resource.openai.azure.com/",
                "api_version": datetime.date(2024, 2, 1),  # what yaml.safe_load produces
                "engine": "gpt-4",
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _get_direct_model_(model_config)
        self.assertIsInstance(result, AzureOpenAI)

    @patch("connectchain.lcel.model.ChatOpenAI", return_value=Mock(ChatOpenAI))
    # pylint: disable=unused-argument
    def test_model_eas_chat_missing_api_version_raises_clear_error(self, *args):
        """Regression test: the EAS/_get_chat_model_ path must raise a clear
        LCELModelException for a missing api_version instead of an opaque pydantic
        ValidationError from deep inside ChatOpenAI's constructor."""
        test_config = get_mock_config()
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        del test_config.data["models"]["1"]["api_version"]
        self.setUpWithConfig(test_config)
        with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
            test_model = model()

    @patch("connectchain.lcel.model.AzureOpenAI", return_value=Mock(AzureOpenAI))
    # pylint: disable=unused-argument
    def test_model_eas_azure_missing_api_version_raises_clear_error(self, *args):
        """Regression test: the EAS/_get_azure_model_ path must raise a clear
        LCELModelException for a missing api_version instead of an opaque pydantic
        ValidationError from deep inside AzureOpenAI's constructor."""
        test_config = get_mock_config()
        test_config.data["models"]["2"] = {**test_config.data["models"]["2"]}
        del test_config.data["models"]["2"]["api_version"]
        self.setUpWithConfig(test_config)
        with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
            test_model = model("2")

    @patch("connectchain.lcel.model.wrap_llm_with_proxy")
    def test_model_configured_with_no_proxy(self, mock_wrap_with_proxy):
        self.setUpWithConfig(get_mock_config())
        model()
        mock_wrap_with_proxy.assert_not_called()

    @patch("connectchain.lcel.model.wrap_llm_with_proxy")
    def test_model_configured_with_global_proxy(self, mock_wrap_with_proxy: Mock):
        test_config = get_mock_config()
        test_proxy_config = {"host": "localhost", "port": 8080}
        test_config.data["proxy"] = test_proxy_config
        self.setUpWithConfig(test_config)
        model_instance = model()
        mock_wrap_with_proxy.assert_called_once()
        self.assertIs(model_instance, mock_wrap_with_proxy.call_args[0][0])
        used_proxy_config = mock_wrap_with_proxy.call_args[0][1]
        self.assertEqual(used_proxy_config["host"], test_proxy_config["host"])
        self.assertEqual(used_proxy_config["port"], test_proxy_config["port"])

    @patch("connectchain.lcel.model.wrap_llm_with_proxy")
    def test_model_configured_with_model_only_proxy(self, mock_wrap_with_proxy: Mock):
        test_config = get_mock_config()
        test_proxy_config = {"host": "localhost", "port": 8080}
        # required to not modify dict instance
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        test_config.data["models"]["1"]["proxy"] = test_proxy_config
        self.setUpWithConfig(test_config)
        model_instance = model()
        mock_wrap_with_proxy.assert_called_once()
        self.assertIs(model_instance, mock_wrap_with_proxy.call_args[0][0])
        used_proxy_config = mock_wrap_with_proxy.call_args[0][1]
        self.assertEqual(used_proxy_config["host"], test_proxy_config["host"])
        self.assertEqual(used_proxy_config["port"], test_proxy_config["port"])

    @patch("connectchain.lcel.model.wrap_llm_with_proxy")
    def test_model_configured_with_model_override_proxy(self, mock_wrap_with_proxy: Mock):
        test_config = get_mock_config()
        test_global_proxy_config = {"host": "localhost", "port": 8080}
        test_model_proxy_config = {"host": "localhost", "port": 8080}
        test_config.data["proxy"] = test_global_proxy_config
        # required to not modify dict instance
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        test_config.data["models"]["1"]["proxy"] = test_model_proxy_config
        self.setUpWithConfig(test_config)
        model_instance = model()
        mock_wrap_with_proxy.assert_called_once()
        self.assertIs(model_instance, mock_wrap_with_proxy.call_args[0][0])
        used_proxy_config = mock_wrap_with_proxy.call_args[0][1]
        self.assertEqual(used_proxy_config["host"], test_model_proxy_config["host"])
        self.assertEqual(used_proxy_config["port"], test_model_proxy_config["port"])
