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
"""Unit tests for Config class by mocking the config.yml
file read operation to return mock_config"""

import os
import unittest

from connectchain.utils import Config
from connectchain.utils.config import ConfigWrapper

# pylint: disable=invalid-name duplicate-code
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
    """Test Config class"""

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


class TestConfigWrapperStrictness(unittest.TestCase):
    """Regression suite for the deliberate ConfigWrapper strictness change:
    attribute access on a missing key raises AttributeError instead of
    silently returning None. The old None-return defeated hasattr() (always
    True) and getattr(obj, key, default) (default never returned) for every
    consumer; these tests pin the new contract down."""

    def test_missing_key_raises_attribute_error_naming_the_key(self):
        """A missing key must raise AttributeError (so misconfigurations fail
        loudly at first use) and the message must name the offending key."""
        wrapper = ConfigWrapper({"present": 1})
        with self.assertRaisesRegex(AttributeError, "missing_key"):
            _ = wrapper.missing_key

    def test_hasattr_reflects_actual_key_presence(self):
        """hasattr() was previously True for EVERY key (None-return); it must
        now genuinely reflect presence."""
        wrapper = ConfigWrapper({"present": 1})
        self.assertTrue(hasattr(wrapper, "present"))
        self.assertFalse(hasattr(wrapper, "absent"))

    def test_getattr_default_is_honored(self):
        """getattr's default was previously dead code (None always returned);
        it must now be returned for missing keys, while present keys still
        resolve to their values."""
        wrapper = ConfigWrapper({"present": 1})
        self.assertEqual(getattr(wrapper, "present", "default"), 1)
        self.assertEqual(getattr(wrapper, "absent", "default"), "default")
        self.assertIsNone(getattr(wrapper, "absent", None))

    def test_nested_wrapper_behavior(self):
        """Dict values come back wrapped (so the strict contract applies at
        every nesting level); scalars come back raw; a missing nested key
        raises just like a missing top-level one."""
        wrapper = ConfigWrapper({"section": {"key": "value"}})
        nested = wrapper.section
        self.assertIsInstance(nested, ConfigWrapper)
        self.assertEqual(nested.key, "value")
        with self.assertRaisesRegex(AttributeError, "other_key"):
            _ = nested.other_key
        self.assertFalse(hasattr(nested, "other_key"))

    def test_none_data_raises_clear_attribute_error_not_type_error(self):
        """A YAML section left empty (e.g. a bare `eas:` line) parses to None.
        Attribute access on such a wrapper used to crash with TypeError
        ("argument of type 'NoneType' is not iterable" from `key in None`);
        it must now raise a clear AttributeError -- which also means
        hasattr()/getattr(default) treat the empty section as attribute-less
        instead of blowing up."""
        wrapper = ConfigWrapper(None)
        with self.assertRaisesRegex(AttributeError, "id_key"):
            _ = wrapper.id_key
        self.assertFalse(hasattr(wrapper, "id_key"))
        self.assertIsNone(getattr(wrapper, "id_key", None))

    def test_non_mapping_data_raises_attribute_error(self):
        """Scalar- and list-valued sections have no sub-keys to read: attribute
        access must raise AttributeError (a str used to be substring-probed by
        `key in data`; a list raised TypeError)."""
        for data in ("scalar-value", [1, 2, 3], 42):
            with self.subTest(data=data):
                wrapper = ConfigWrapper(data)
                with self.assertRaisesRegex(AttributeError, "some_key"):
                    _ = wrapper.some_key
                self.assertFalse(hasattr(wrapper, "some_key"))

    def test_item_access_stays_lenient(self):
        """The strictness change is scoped to ATTRIBUTE access only: item
        access is the explicit optional accessor (e.g. models[index] followed
        by an is-None check) and must keep returning None for missing keys
        and out-of-range list indices."""
        wrapper = ConfigWrapper({"present": 1, "items": [10]})
        self.assertIsNone(wrapper["absent"])
        self.assertEqual(wrapper["present"], 1)
        self.assertIsNone(ConfigWrapper([10])[5])

    def test_config_missing_top_level_key_raises_attribute_error(self):
        """Config.__getattr__ used to raise KeyError for a missing top-level
        key, which escaped straight through hasattr()/getattr(default) (they
        only swallow AttributeError). It must now follow the same
        AttributeError contract as ConfigWrapper so those idioms work on the
        root config object too (e.g. getattr(config, "proxy", None))."""
        with open("mock_config_strict.yml", "w", encoding="utf-8") as f:
            f.write(mock_config)
        self.addCleanup(os.remove, "mock_config_strict.yml")
        config = Config("mock_config_strict.yml")
        with self.assertRaisesRegex(AttributeError, "proxy"):
            _ = config.proxy
        self.assertFalse(hasattr(config, "proxy"))
        self.assertIsNone(getattr(config, "proxy", None))
        self.assertTrue(hasattr(config, "eas"))

    def test_empty_yaml_section_is_treated_as_attribute_less(self):
        """End-to-end through Config: a section present but empty (`proxy:`)
        wraps None; reading keys from it raises AttributeError (not TypeError)
        so presence probes via getattr/hasattr degrade gracefully."""
        with open("mock_config_empty_section.yml", "w", encoding="utf-8") as f:
            f.write(mock_config + "\nproxy:\n")
        self.addCleanup(os.remove, "mock_config_empty_section.yml")
        config = Config("mock_config_empty_section.yml")
        empty_section = config.proxy  # present, wraps None
        self.assertIsInstance(empty_section, ConfigWrapper)
        with self.assertRaisesRegex(AttributeError, "host"):
            _ = empty_section.host
        self.assertIsNone(getattr(empty_section, "host", None))
