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

from .exceptions import NonRetryableError


class ConfigException(Exception, NonRetryableError):
    """Base exception for the config class. A missing/malformed config file will never
    succeed on retry -- see NonRetryableError."""


def _require_key_(data: Any, key: str, owner: str) -> Any:
    """Attribute-lookup core shared by Config and ConfigWrapper: validate that
    `data` is a mapping containing `key` and return data[key].

    Both failure modes raise AttributeError -- never KeyError or TypeError --
    because AttributeError is the only exception type Python's own
    optional-attribute tools understand: hasattr() returns False and
    getattr(obj, key, default) returns the default ONLY for AttributeError;
    anything else propagates as a crash out of those very calls.

    * Non-mapping `data`: a YAML section left empty (e.g. a bare `eas:` line)
      parses to None, and a scalar-valued section has no sub-keys to read.
      `key in data` would raise TypeError on None (or silently do a substring
      test on a str); a clear AttributeError instead makes such sections look
      "attribute-less", which is what they are.
    * Missing `key`: intentional strictness -- see ConfigWrapper's docstring.
    """
    if not isinstance(data, dict):
        raise AttributeError(
            f"Cannot read config key '{key}': {owner} holds "
            f"{type(data).__name__} data, not a mapping"
        )
    if key not in data:
        raise AttributeError(f"Config key '{key}' not found in {owner}")
    return data[key]


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
        """Get config item by key. Raises KeyError if absent (dict-style access
        keeps dict-style errors)."""
        return ConfigWrapper(self.data[key])

    def __getattr__(self, key: str) -> "ConfigWrapper":
        """Get config attribute by key.

        Raises AttributeError for a missing top-level key. (Formerly KeyError,
        which broke hasattr()/getattr(default) on the root config object: they
        only swallow AttributeError, so the KeyError escaped through them.)
        Same strictness contract as ConfigWrapper.__getattr__ -- see its class
        docstring."""
        return ConfigWrapper(_require_key_(self.data, key, "Config"))


class ConfigWrapper:
    """Wrapper providing attribute- and item-style access to a parsed YAML
    config tree (used for sections/models pulled out of Config).

    Attribute access is STRICT by design: reading a key that is absent -- or
    reading any key from a section whose data is not a mapping, e.g. a YAML
    section left empty, which parses to None -- raises AttributeError. This is
    intentional strictness so misconfigurations fail loudly at the point of
    first use. The previous behavior of silently returning None for missing
    keys defeated Python's optional-attribute tools for every consumer:
    hasattr() was always True and getattr(obj, key, default) never returned
    its default, so call sites could not distinguish "unset" from "set", and
    misconfigurations surfaced as far-away crashes on unexpected Nones. With
    AttributeError, hasattr()/getattr(default) genuinely work; use them for
    keys that are legitimately optional.

    Item access (wrapper["key"] / wrapper[0]) intentionally KEEPS the lenient
    None-return for missing keys/indices: it is the explicit "give me the
    value if present" accessor (e.g. `models[index]` followed by an is-None
    check), and dict-style lookup has no hasattr()-style idiom for the
    None-return to defeat.
    """

    def __init__(self, data: Any) -> None:
        """Initialize config wrapper with data."""
        self.data = data

    def __getattr__(self, key: str) -> Union["ConfigWrapper", Any]:
        """Get attribute by key; raises AttributeError if absent or if the
        wrapped data is not a mapping (see class docstring)."""
        value = _require_key_(self.data, key, "ConfigWrapper")
        if isinstance(value, dict):
            return ConfigWrapper(value)
        return value

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
