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
"""
This module provides a simple wrapper around a yaml config file.
"""
import os
import yaml

class ConfigException(BaseException):
    """Base exception for the config class"""

class Config:
    """ Config Class"""
    @staticmethod
    def from_env() -> str:
        """ Static method to get config from environment variable"""
        config_path = os.getenv("CONFIG_PATH")
        if config_path is None:
            raise ConfigException("CONFIG_PATH environment variable not set")

        return Config(config_path)

    def __init__(self, filepath):
        with open(filepath, 'r',encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def __getitem__(self, key):
        return ConfigWrapper(self.data[key])

    def __getattr__(self, key):
        return ConfigWrapper(self.data[key])


class ConfigWrapper:
    """Wrapper for Config Class"""
    def __init__(self, data):
        self.data = data

    def __getattr__(self, key):
        if key not in self.data:
            return None
        if isinstance(self.data[key], dict):
            return ConfigWrapper(self.data[key])
        return self.data[key]

    def __getitem__(self, key):
        if key not in self.data:
            return None
        if isinstance(self.data[key], dict):
            return ConfigWrapper(self.data[key])
        return self.data[key]


# Example usage:
if __name__ == '__main__':
    config = Config('../config/config.yml')
    print(config.eas.scope[0])  # Returns "/blabla/**::get"
    print(config.cert.cert_path)  # Returns "https://someurl.com"
