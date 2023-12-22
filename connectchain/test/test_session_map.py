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
import unittest
from connectchain.utils import SessionMap

from .setup_utils import get_mock_config, wrap_model_config

class TestSessionMap(unittest.TestCase):
    """Unit testing the model LCEL method """

    def test_uuid_from_config(self):
        """Test that a PortableOrchestrator instance can be built with the default LLM"""
        test_config = get_mock_config()
        test_uuid = SessionMap.uuid_from_config(test_config, test_config.models['1'])
        test_uuid2 = SessionMap.uuid_from_config(test_config, test_config.models['2'])
        self.assertEqual(test_uuid, 'id_key_secret_key_openai_chat_engine_test_model_api_version')
        self.assertEqual(test_uuid2, 'id_key_secret_key_openai_azure_engine_test_model_other_api_version_other')

    def test_model_config_override(self):
        test_config = get_mock_config()
        test_model_config = wrap_model_config({
            'eas': {
                'id_key': 'mod_id',
                'secret_key': 'mod_sec' # EARLYBIRD-IGNORE
            },
            'provider': 'oss_provider',
            'model_name': 'some_model',
            'type': 'some_model_type',
            'engine': 'oss_engine',
            'api_version': 'latest'
        })
        test_uuid = SessionMap.uuid_from_config(test_config, test_model_config)
        self.assertEqual(test_uuid, 'mod_id_mod_sec_oss_provider_some_model_type_oss_engine_some_model_latest')
