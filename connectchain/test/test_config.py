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
""" Unit tests for Config class by mocking the config.yml
 file read operation to return mock_config """

import unittest
import os
from connectchain.utils import Config

#pylint: disable=invalid-name duplicate-code
mock_config = """
eas:
    url: https://someurl/token
    scope: [
        /blabla/**::get,
        /blabla/**::post
        ]
    originator_source: digital-something
cert:
    cert_path: https://someurl.com
    cert_name: ./some_cert.crt
"""



class TestConfig(unittest.TestCase):
    """ Test Config class"""
    def test_config(self):
        "create mock config.yml file"
        with open("mock_config.yml", "w", encoding="utf-8") as f:
            f.write(mock_config)
        # create mock config object
        config = Config("mock_config.yml")
        # assert that the eas url is the same as the eas url in config
        self.assertEqual(config.eas.url, "https://someurl/token")
        # assert that the eas scope is the same as the eas scope in config
        self.assertEqual(config.eas.scope[0], "/blabla/**::get")
        # assert that the eas originator_source is the
        # same as the eas originator_source in config
        self.assertEqual(config.eas.originator_source, "digital-something")
        # assert that the cert cert_path is the same as the cert cert_path in config
        self.assertEqual(config.cert.cert_path, "https://someurl.com")
        # assert that the cert cert_name is the same as the cert cert_name in config
        self.assertEqual(config.cert.cert_name, "./some_cert.crt")
        # remove mock config.yml file
        os.remove("mock_config.yml")
