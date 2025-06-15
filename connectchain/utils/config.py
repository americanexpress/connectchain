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
from typing import Any, Dict, Union

import yaml


class ConfigException(BaseException):
    """Base exception for the config class"""


class Config:
    """Config Class"""

    def __init__(self, filepath: str) -> None:
        """Initialize config with YAML file path."""
        with open(filepath, "r", encoding="utf-8") as f:
            self.data: Dict[str, Any] = yaml.safe_load(f)

    @staticmethod
    def from_env() -> "Config":
        """Static method to get config from environment variable"""
        config_path = os.getenv("CONFIG_PATH")
        if config_path is None:
            raise ConfigException("CONFIG_PATH environment variable not set")

        return Config(config_path)

    def __getitem__(self, key: str) -> "ConfigWrapper":
        """Get config item by key."""
        return ConfigWrapper(self.data[key])

    def __getattr__(self, key: str) -> "ConfigWrapper":
        """Get config attribute by key."""
        return ConfigWrapper(self.data[key])


class ConfigWrapper:
    """Wrapper for Config Class"""

    def __init__(self, data: Any) -> None:
        """Initialize config wrapper with data."""
        self.data = data

    def __getattr__(self, key: str) -> Union["ConfigWrapper", Any, None]:
        """Get attribute by key, returning None if not found."""
        if key not in self.data:
            return None
        if isinstance(self.data[key], dict):
            return ConfigWrapper(self.data[key])
        return self.data[key]

    def __getitem__(self, key: Union[str, int]) -> Union["ConfigWrapper", Any, None]:
        """Get item by key, returning None if not found."""
        if isinstance(key, int):
            # Handle list/array indexing
            if isinstance(self.data, list) and 0 <= key < len(self.data):
                if isinstance(self.data[key], dict):
                    return ConfigWrapper(self.data[key])
                return self.data[key]
            return None

        # Handle dictionary key access
        if key not in self.data:
            return None
        if isinstance(self.data[key], dict):
            return ConfigWrapper(self.data[key])
        return self.data[key]


# Example usage:
if __name__ == "__main__":
    config = Config("../config/config.yml")
    scope = config.eas.scope
    if scope:
        print(scope[0])  # Returns "/blabla/**::get"
    cert_path = config.cert.cert_path
    if cert_path:
        print(cert_path)  # Returns "https://someurl.com"
