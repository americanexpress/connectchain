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
from connectchain.utils.config import Config, ConfigWrapper

class MockConfig(Config):
    
    def __init__(self, data):
        self.data = data

MOCK_DATA = {
    'eas': {
        'id_key': 'id_key',
        'secret_key': 'secret_key',
        'token_refresh_interval': 100
    },
    'cert': {
        'cert_name': 'cert_name',
        'cert_path': 'cert_path',
        'cert_size': 1938
    },
    'models': {
        '1': {
            'provider': 'openai',
            'type': 'chat',
            'engine': 'engine',
            'model_name': 'test_model',
            'api_base': 'test_base',
            'api_version': 'api_version'
        },
        '2': {
            'provider': 'openai',
            'type': 'azure',
            'engine': 'engine',
            'model_name': 'test_model_other',
            'api_base': 'test_base_other',
            'api_version': 'api_version_other'
        }
    }
}

def wrap_model_config(model_config):
    return ConfigWrapper(model_config)

def get_mock_config(data=None):
    if data is None:
        data = { x: { **y } for x, y in MOCK_DATA.items() }
    return MockConfig(data)
