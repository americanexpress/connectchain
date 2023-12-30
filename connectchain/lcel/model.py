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
""" LCEL model module """
import os
from typing import Any
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import AzureOpenAI
from langchain.schema.language_model import BaseLanguageModel
from connectchain.utils import Config, get_token_from_env, SessionMap
from connectchain.utils.llm_proxy_wrapper import wrap_llm_with_proxy

class LCELModelException(BaseException):
    """Base exception for the LCEL model"""

def model(index: Any = '1') -> BaseLanguageModel:
    """
    Though name of this method may be confusing, the purpose is to keep
    the LCEL notation nearly intact
    """
    llm = _get_model_(index)
    return llm

def _get_model_(index):
    """Get the model config based on the models defined in the config"""
    config = Config.from_env()
    try:
        models = config.models
    except KeyError as ex:
        raise LCELModelException('No models defined in config') from ex
    model_config = models[index]
    if model_config is None:
        raise LCELModelException(f'Model config at index "{index}" is not defined')
    model_instance = None
    if model_config.provider == 'openai':
        model_instance = _get_openai_model_(index, config, model_config)
    if model_instance is None:
        raise LCELModelException('Not implemented')
    try:
        proxy_config = model_config.proxy
    except KeyError:
        # Proxy settings not required
        pass
    if proxy_config is None:
        try:
            proxy_config = config.proxy
            # Proxy settings not required
        except KeyError:
            pass
    if proxy_config is not None:
        wrap_llm_with_proxy(model_instance, proxy_config)
    return model_instance

def _get_openai_model_(index, config, model_config):
    """Get the OpenAI LLM instance"""
    model_session_key = SessionMap.uuid_from_config(config, model_config)
    auth_token = os.getenv(model_session_key)
    session_map = SessionMap(config.eas.token_refresh_interval)
    if auth_token is None or session_map.is_expired(model_session_key):
        auth_token = get_token_from_env(index)
        os.environ[model_session_key] = auth_token
        if model_config.type == 'chat':
            llm = _get_chat_model_(auth_token, model_config)
        else:
            llm = _get_azure_model_(auth_token, model_config)
        session_map.new_session(model_session_key, llm)
        return llm
    return session_map.get_llm(model_session_key)

def _get_chat_model_(auth_token, model_config):
    """Get a ChatOpenAI instance"""
    llm = ChatOpenAI(
        model_name=model_config.model_name,
        openai_api_key=auth_token,
        openai_api_base=model_config.api_base,
        model_kwargs={
            "engine": model_config.engine,
            "api_version": model_config.api_version,
            "api_type": "azure"
        })
    return llm

def _get_azure_model_(auth_token, model_config):
    """Get an AzureOpenAI instance"""
    llm = AzureOpenAI(
        model_name=model_config.model_name,
        openai_api_key=auth_token,
        openai_api_base=model_config.api_base,
        openai_api_version=model_config.api_version,
        model_kwargs={
            "engine": model_config.engine,
            "api_version": model_config.api_version,
            "api_type": "azure"
        })
    return llm
