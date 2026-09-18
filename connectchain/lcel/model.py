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
"""LCEL model module"""
import logging
import os
from typing import Any, Optional
from urllib.parse import urlparse

from langchain.schema.language_model import BaseLanguageModel
from langchain_openai import AzureOpenAI, ChatOpenAI
from pydantic import SecretStr

from connectchain.utils import Config, SessionMap, get_token_from_env
from connectchain.utils.exceptions import NonRetryableError
from connectchain.utils.llm_proxy_wrapper import wrap_llm_with_proxy

logger = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = ("openai", "anthropic", "google", "cohere", "huggingface")


class LCELModelException(Exception):
    """Base exception for the LCEL model.

    Deliberately NOT marked NonRetryableError: this base class also wraps
    UNEXPECTED model-initialisation failures (see _get_direct_model_'s broad
    except) whose retryability is unknown -- e.g. a transient network or
    provider outage surfacing inside init_chat_model() -- so retry wrappers
    (connectchain.utils.retry) must stay free to retry it. Known-permanent
    configuration problems raise LCELModelConfigError instead; `except
    LCELModelException` still catches both via the subclass relationship.
    """


class LCELModelConfigError(LCELModelException, NonRetryableError):
    """A permanent configuration problem with the LCEL model: missing config
    entry, unset API-key env var, unsupported provider, missing api_version,
    or an uninstalled provider package. These will never succeed on retry, so
    the NonRetryableError marker tells connectchain.utils.retry's wrappers to
    re-raise immediately instead of burning retry attempts on nothing."""


def model(index: Any = "1") -> BaseLanguageModel:
    """
    Though name of this method may be confusing, the purpose is to keep
    the LCEL notation nearly intact
    """
    llm = _get_model_(index)
    return llm


def _get_model_(index: Any) -> BaseLanguageModel:
    """Get the model config based on the models defined in the config"""
    config = Config.from_env()
    try:
        models = config.models
    except AttributeError as ex:
        # Config raises AttributeError for a missing top-level key.
        raise LCELModelConfigError("No models defined in config") from ex
    model_config = models[index]
    if model_config is None:
        raise LCELModelConfigError(f'Model config at index "{index}" is not defined')

    # EAS routing requires an eas section carrying a truthy id_key, unless the
    # model opts out via bypass_eas. getattr's default genuinely works now that
    # Config/ConfigWrapper raise AttributeError for missing keys -- and also
    # for a section left empty in YAML (`eas:` parses to None, and attribute
    # access on a None-holding wrapper raises AttributeError too) -- so the
    # try/except probe cascade this replaces is no longer needed.
    eas_config = getattr(config, "eas", None)
    needs_eas = bool(getattr(eas_config, "id_key", None)) and not getattr(
        model_config, "bypass_eas", False
    )

    if model_config.provider != "openai":
        needs_eas = False

    if needs_eas:
        model_instance = _get_openai_model_(index, config, model_config)
    else:
        model_instance = _get_direct_model_(model_config)

    if model_instance is None:
        raise LCELModelConfigError("Not implemented")
    # A model-level proxy overrides the global one; both are optional, so both
    # reads use getattr's None default (missing keys raise AttributeError now).
    proxy_config = getattr(model_config, "proxy", None)
    if proxy_config is None:
        proxy_config = getattr(config, "proxy", None)
    if proxy_config is not None:
        wrap_llm_with_proxy(model_instance, proxy_config)
    return model_instance


def _get_openai_model_(index: Any, config: Any, model_config: Any) -> BaseLanguageModel:
    """Get the OpenAI LLM instance"""
    model_session_key = SessionMap.uuid_from_config(config, model_config)
    # Captured AND fully resolved now, before any I/O below, then passed explicitly
    # to new_session() -- reading session_map.expires_in again after
    # get_token_from_env()'s network call would race a concurrent request that
    # reconstructs the singleton for a different model's token_refresh_interval in
    # between. Resolving an unset token_refresh_interval (None) to the default HERE
    # matters for the same reason: passing None would make new_session() fall back
    # to the singleton's CURRENT expires_in after the network call, reopening the
    # race and inheriting whatever interval another model configured last.
    # token_refresh_interval is optional, hence getattr's None default.
    expires_in = getattr(config.eas, "token_refresh_interval", None)
    if expires_in is None:
        expires_in = SessionMap.DEFAULT_EXPIRES_IN
    session_map = SessionMap(expires_in)
    if os.getenv(model_session_key) is not None:
        # get_valid_llm() combines the expiry check and cache read into a single
        # locked operation instead of two (is_expired() then get_llm()).
        cached_llm = session_map.get_valid_llm(model_session_key)
        if cached_llm is not None:
            # Note: SessionMap is typed for LLMResult but we store LLM instances
            return cached_llm  # type: ignore[return-value]
    auth_token = get_token_from_env(index)
    os.environ[model_session_key] = auth_token
    if model_config.type == "chat":
        llm: BaseLanguageModel = _get_chat_model_(auth_token, model_config)
    else:
        llm = _get_azure_model_(auth_token, model_config)
    session_map.new_session(model_session_key, llm, expires_in)  # type: ignore[arg-type]
    return llm


# Azure OpenAI endpoint domains: public cloud, US Government, and China sovereign
# clouds. APIM/custom domains can't be detected by hostname -- configs using one
# should set `azure: true` on the model entry instead.
_AZURE_ENDPOINT_MARKERS = ("openai.azure.com", "openai.azure.us", "openai.azure.cn")


def _is_azure_endpoint_(model_config: Any, api_base: Any) -> bool:
    """Whether this config targets Azure OpenAI: either an api_base whose HOSTNAME is
    on a known Azure domain, or an explicit `azure: true` flag for custom/APIM domains.

    The hostname is extracted with urlparse and suffix-matched (exact host or a
    subdomain, i.e. host == marker or host.endswith("." + marker)) rather than
    substring-matched over the whole URL: a substring match would misroute lookalikes
    such as https://myproxy.corp.com/openai.azure.com-compat (marker in the path) or
    https://notopenai.azure.com.evil.example (marker embedded in a non-Azure host)
    to the Azure builder.
    """
    if getattr(model_config, "azure", None):
        return True
    if not api_base:
        return False
    parsed = urlparse(str(api_base))
    hostname = parsed.hostname
    if hostname is None:
        # urlparse quirk: a scheme-less value like "my-resource.openai.azure.com/x"
        # parses entirely into .path with no hostname. Re-parse as network-relative
        # ("//host/...") so a scheme-less Azure api_base is still recognised as
        # Azure instead of silently falling through to a plain OpenAI client.
        hostname = urlparse(f"//{api_base}").hostname
    if not hostname:
        return False
    # urlparse lowercases .hostname, so the comparison is already case-insensitive.
    return any(
        hostname == marker or hostname.endswith(f".{marker}")
        for marker in _AZURE_ENDPOINT_MARKERS
    )


def _require_api_version_(model_config: Any, api_base: Any) -> Any:
    """Validate api_version is configured before building an Azure client.

    Without this, AzureOpenAI/ChatOpenAI(model_kwargs={"api_version": None, ...}) fails
    with an opaque pydantic ValidationError ("Must provide either the api_version
    argument or the OPENAI_API_VERSION environment variable") deep inside a third-party
    constructor. Raising here gives a clear, actionable error instead.
    """
    api_version = getattr(model_config, "api_version", None)
    if not api_version:
        raise LCELModelConfigError(
            f"Azure OpenAI endpoint detected ({api_base}) but no api_version is "
            f"configured; api_version is required for Azure OpenAI."
        )
    # YAML parses an unquoted `api_version: 2024-02-01` as datetime.date, which the
    # AzureOpenAI/ChatOpenAI pydantic validators reject with an opaque error. Coerce
    # so both the quoted and the natural unquoted config spelling work.
    return str(api_version)


def _temperature_kwargs_(model_config: Any) -> dict:
    """{'temperature': t} when configured, else {} -- so an unset temperature keeps
    each provider's own default instead of being forced to None. getattr's None
    default covers a missing key (ConfigWrapper raises AttributeError for those)
    as well as an explicit `temperature: null`, and works the same for plain
    (non-ConfigWrapper) config objects."""
    temperature = getattr(model_config, "temperature", None)
    return {} if temperature is None else {"temperature": temperature}


def _azure_model_kwargs_(engine: Any) -> dict:
    """Shared model_kwargs for AzureOpenAI. api_version is deliberately not repeated
    here: both callers already pass it as a direct constructor kwarg, and AzureOpenAI's
    pydantic validation rejects api_version appearing a second time inside model_kwargs
    ('supplied twice')."""
    return {"engine": engine, "api_type": "azure"}


def _get_chat_model_(auth_token: str, model_config: Any) -> ChatOpenAI:
    """Get a ChatOpenAI instance"""
    api_version = _require_api_version_(model_config, model_config.api_base)
    return ChatOpenAI(
        model=model_config.model_name,
        api_key=SecretStr(auth_token) if auth_token else None,
        base_url=model_config.api_base,
        # api_version goes inside model_kwargs here (unlike _get_azure_model_ /
        # _get_direct_azure_model_) because ChatOpenAI has no top-level api_version kwarg.
        model_kwargs={**_azure_model_kwargs_(model_config.engine), "api_version": api_version},
        **_temperature_kwargs_(model_config),
    )


def _get_azure_model_(auth_token: str, model_config: Any) -> AzureOpenAI:
    """Get an AzureOpenAI instance"""
    api_version = _require_api_version_(model_config, model_config.api_base)
    return AzureOpenAI(
        model=model_config.model_name,
        api_key=SecretStr(auth_token) if auth_token else None,
        azure_endpoint=model_config.api_base,
        api_version=api_version,
        model_kwargs=_azure_model_kwargs_(model_config.engine),
        **_temperature_kwargs_(model_config),
    )


def _get_direct_azure_model_(model_config: Any, api_key: str, api_base: Any) -> AzureOpenAI:
    """Build a direct-API-key AzureOpenAI client.

    Called from _get_direct_model_ BEFORE the generic init_chat_model() fast path is
    even attempted: that fast path has no notion of azure_endpoint/api_version/engine,
    so routing an Azure-shaped api_base through it would silently succeed with a plain
    ChatOpenAI pointed at the Azure URL instead of a working (or clearly failing) Azure
    client.
    """
    api_version = _require_api_version_(model_config, api_base)
    engine = getattr(model_config, "engine", None) or model_config.model_name
    return AzureOpenAI(
        model=model_config.model_name,
        api_key=SecretStr(api_key),
        azure_endpoint=api_base,
        api_version=api_version,
        model_kwargs=_azure_model_kwargs_(engine),
        **_temperature_kwargs_(model_config),
    )


def _resolve_direct_api_key_(model_config: Any) -> str:
    """Resolve the API key for a direct (non-EAS) model from its configured or
    provider-default environment variable, raising a clear error if unset."""
    api_key_env = getattr(model_config, "api_key_env", None)
    if not api_key_env:
        # Default to {PROVIDER}_API_KEY pattern
        api_key_env = f"{model_config.provider.upper()}_API_KEY"

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise LCELModelConfigError(
            f"API key not found in environment variable: {api_key_env}. "
            f"Please set it in your .env file or environment."
        )
    return api_key


def _resolve_fast_path_api_key_(model_config: Any) -> Optional[str]:
    """Resolve an explicitly configured api_key_env for the init_chat_model()
    fast path, returning None whenever resolution should instead be left to
    init_chat_model()'s own provider-default handling.

    Companion to _resolve_direct_api_key_ (the manual-fallback resolver); the
    two deliberately differ on an UNSET named env var. Here we must not fail
    hard: init_chat_model() resolves the provider's own default env var (e.g.
    OPENAI_API_KEY) when no explicit api_key is passed, which may well work.
    The misconfiguration is surfaced loudly via logger.warning, then init
    proceeds without an explicit key. (The manual fallback path still
    hard-fails via _resolve_direct_api_key_ if no key is resolvable at all.)
    """
    api_key_env = getattr(model_config, "api_key_env", None)
    if not api_key_env:
        return None
    api_key = os.getenv(api_key_env)
    if api_key:
        return api_key
    logger.warning(
        "api_key_env is set to '%s' but that environment variable is unset; "
        "falling back to the provider's default API-key environment variable "
        "resolution.",
        api_key_env,
    )
    return None


def _get_direct_model_(model_config: Any) -> BaseLanguageModel:
    """Get a direct API model instance for any provider without EAS authentication.

    ImportError and ValueError from init_chat_model() are expected fallback
    conditions (missing provider package, bad model name, or init_chat_model
    itself unavailable on an old langchain): for providers with a manual
    initialisation branch below (_SUPPORTED_PROVIDERS) they are logged as a
    warning and fall through to that manual init; for any other provider there
    is nothing to fall back to, so a clear LCELModelConfigError is raised
    instead of the misleading fallback warning. Any other exception is
    unexpected and is re-raised as (retry-eligible) LCELModelException with
    the original traceback preserved via `raise ... from e`.
    """
    # Azure-shaped configs are routed straight to a dedicated Azure builder, BEFORE the
    # generic init_chat_model() fast path below is ever attempted. init_chat_model() has
    # no notion of azure_endpoint/api_version/engine, so letting an Azure api_base reach
    # it would silently succeed with a plain, non-Azure ChatOpenAI instead of a working
    # (or clearly failing) Azure client.
    api_base = getattr(model_config, "api_base", None)
    if model_config.provider == "openai" and _is_azure_endpoint_(model_config, api_base):
        azure_api_key = _resolve_direct_api_key_(model_config)
        return _get_direct_azure_model_(model_config, azure_api_key, api_base)

    # Resolved before the try block so the env lookup can't be mistaken for an
    # "unexpected" init error by the `except Exception` below.
    explicit_api_key = _resolve_fast_path_api_key_(model_config)

    try:
        from langchain.chat_models import init_chat_model  # pylint: disable=import-outside-toplevel

        model_name = model_config.model_name
        config_dict: dict = {}

        if api_base:
            config_dict["base_url"] = api_base
        config_dict.update(_temperature_kwargs_(model_config))
        if explicit_api_key:
            config_dict["api_key"] = explicit_api_key
        if model_config.provider not in _SUPPORTED_PROVIDERS:
            # Providers without a manual fallback branch below (mistralai, groq,
            # ollama, bedrock, ...) are still served by init_chat_model(). For
            # these, the configured provider is passed explicitly rather than
            # inferred from model_name, so a genuinely unknown provider fails
            # right here with init_chat_model()'s clear "Unsupported
            # model_provider=..." ValueError instead of silently constructing a
            # model for whatever provider the name happens to infer to (e.g.
            # provider 'meta' with model_name 'gpt-4' must not build a ChatOpenAI).
            config_dict["model_provider"] = model_config.provider

        return init_chat_model(model_name, **config_dict)

    except (ImportError, ValueError) as e:
        # Expected: missing provider package, unrecognised model/provider name, or
        # init_chat_model itself unavailable.
        if model_config.provider not in _SUPPORTED_PROVIDERS:
            # No manual initialisation branch exists for this provider, so the
            # "falling back" warning below would be misleading -- there is nothing
            # to fall back to. Fail fast with a clear error instead; the chained
            # original (`from e`) explains WHY the fast path refused it (unknown
            # provider vs. an uninstalled provider package).
            raise LCELModelConfigError(
                f"Provider '{model_config.provider}' not supported "
                f"({type(e).__name__}: {e}). Providers with a built-in fallback: "
                f"{', '.join(_SUPPORTED_PROVIDERS)}"
            ) from e
        # Log and fall through to the manual initialisation block below.
        logger.warning(
            "init_chat_model() failed for '%s' (%s: %s); falling back to manual provider init.",
            model_config.model_name,
            type(e).__name__,
            e,
        )
    except Exception as e:  # pylint: disable=broad-except
        # Unexpected failure: preserve the original traceback. Wrapped as the plain
        # (retry-eligible) LCELModelException, NOT LCELModelConfigError: the cause
        # here is by definition unknown -- it may be a transient network/provider
        # error -- so it must not be branded never-retryable. Use getattr for the
        # model name so constructing this message can never raise a second, uncaught
        # exception (e.g. when the original failure was model_config.model_name itself
        # raising AttributeError on a malformed config).
        raise LCELModelException(
            f"Unexpected error initialising model "
            f"'{getattr(model_config, 'model_name', '<unknown>')}': {e}"
        ) from e

    # ── Manual provider-specific initialisation fallback (Azure and unsupported
    # providers already handled above) ──────────────────────────────────────
    api_key = _resolve_direct_api_key_(model_config)

    if model_config.provider == "openai":
        return ChatOpenAI(
            model=model_config.model_name,
            api_key=api_key,
            base_url=api_base,  # Can be None for default OpenAI endpoint
            **_temperature_kwargs_(model_config),
        )

    elif model_config.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # pylint: disable=import-outside-toplevel

            return ChatAnthropic(
                model=model_config.model_name,
                anthropic_api_key=api_key,
                anthropic_api_url=getattr(model_config, "api_base", None),
                **_temperature_kwargs_(model_config),
            )
        except ImportError as exc:
            raise LCELModelConfigError(
                "langchain-anthropic not installed. Run: pip install langchain-anthropic"
            ) from exc

    elif model_config.provider == "google":
        try:
            from langchain_google_genai import (  # pylint: disable=import-outside-toplevel
                ChatGoogleGenerativeAI,
            )

            return ChatGoogleGenerativeAI(
                model=model_config.model_name,
                google_api_key=api_key,
                **_temperature_kwargs_(model_config),
            )
        except ImportError as exc:
            raise LCELModelConfigError(
                "langchain-google-genai not installed. Run: pip install langchain-google-genai"
            ) from exc

    elif model_config.provider == "cohere":
        try:
            from langchain_cohere import ChatCohere  # pylint: disable=import-outside-toplevel

            return ChatCohere(
                model=model_config.model_name,
                cohere_api_key=api_key,
                **_temperature_kwargs_(model_config),
            )
        except ImportError as exc:
            raise LCELModelConfigError(
                "langchain-cohere not installed. Run: pip install langchain-cohere"
            ) from exc

    elif model_config.provider == "huggingface":
        try:
            from langchain_huggingface import (  # pylint: disable=import-outside-toplevel
                HuggingFaceEndpoint,
            )

            return HuggingFaceEndpoint(
                repo_id=model_config.model_name,
                huggingfacehub_api_token=api_key,
                endpoint_url=getattr(model_config, "api_base", None),
                **_temperature_kwargs_(model_config),
            )
        except ImportError as exc:
            raise LCELModelConfigError(
                "langchain-huggingface not installed. Run: pip install langchain-huggingface"
            ) from exc

    # Defensive: fires only if a provider is added to _SUPPORTED_PROVIDERS without a
    # matching branch above; also satisfies mypy's non-exhaustive-return check.
    raise LCELModelConfigError(
        f"No initialisation branch for provider '{model_config.provider}'"
    )
