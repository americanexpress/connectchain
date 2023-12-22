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
"""Unit test for token_util"""
#pylint: disable=unused-import unused-variable unused-argument protected-access
import os
import asyncio
from datetime import datetime, timedelta
from unittest import TestCase
from unittest.mock import patch, ANY
from connectchain.utils import get_token_from_env, TokenUtil, UtilException
from .setup_utils import get_mock_config, wrap_model_config

class TestTokenUtil(TestCase):
    """Unit Test Class to test TokenUtil"""

    def test_get_token(self):
        token_util = TokenUtil("test_id", "test_secret", get_mock_config())
        valid_response = [{ 'authorization_token': 'test_token' }, 200]

        @patch('os.path.exists', return_value=True)
        @patch(f'{__name__}.TokenUtil._TokenUtil__aio_http_post', return_value=valid_response)
        @patch.object(token_util, '_TokenUtil__get_signature', return_value='signature')
        @patch.object(token_util, '_TokenUtil__retrieve_cert', return_value=None)
        def test_certificate_exists(mock_retrieve_cert, mock_sign, mock_post, *args):
            test_token = asyncio.run(token_util.get_token(token_util.config.models['1']))
            self.assertEqual(test_token, 'Bearer test_token')
            mock_retrieve_cert.assert_not_called()
            mock_sign.assert_called_once()
            mock_post.assert_called_once()

        @patch('os.path.exists', return_value=False)
        @patch(f'{__name__}.TokenUtil._TokenUtil__aio_http_post', return_value=valid_response)
        @patch.object(token_util, '_TokenUtil__get_signature', return_value='signature')
        @patch.object(token_util, '_TokenUtil__retrieve_cert', return_value=None)
        def test_certificate_does_not_exist(mock_retrieve_cert, mock_sign, mock_post, *args):
            test_token = asyncio.run(token_util.get_token(token_util.config.models['1']))
            self.assertEqual(test_token, 'Bearer test_token')
            mock_retrieve_cert.assert_called_once()
            mock_sign.assert_called_once()
            mock_post.assert_called_once()

        @patch.object(token_util, '_TokenUtil__get_signature', return_value='signature')
        @patch.object(token_util, '_TokenUtil__retrieve_cert', return_value=None)
        @patch(f'{__name__}.TokenUtil._TokenUtil__aio_http_post', return_value=valid_response)
        @patch('os.path.exists', return_value=False)
        def test_model_config_overrides(os_path_mock, mock_post, *args):
            test_model_url = 'test_url'
            test_model_cert_name = 'test_name'
            test_model_config = wrap_model_config({
                'eas': {
                    'url': test_model_url
                },
                'cert': {
                    'cert_name': test_model_cert_name,
                }
            })
            asyncio.run(token_util.get_token(test_model_config))
            os_path_mock.assert_called_once_with(test_model_cert_name)
            mock_post.assert_called_once_with(ANY, ANY, test_model_url, ANY, ANY, ANY)

        test_certificate_exists()
        test_certificate_does_not_exist()
        test_model_config_overrides()

    def test_retrieve_cert(self):
        token_util = TokenUtil("test_id", "test_secret", get_mock_config())
        cert_expire_tomorrow = datetime.now() + timedelta(days=1)

        @patch('os.path.getsize', return_value=0)
        @patch(f'connectchain.utils.token_util.urllib', return_value=None)
        def test_fails_if_file_empty(*args):
            with self.assertRaisesRegex(UtilException, "Failed to Download the certificate"):
                token_util._TokenUtil__retrieve_cert(token_util.config.models['1'])

        @patch(f'{__name__}.TokenUtil.read_cert', return_value="mock_cert")
        @patch(f'{__name__}.TokenUtil.get_cert_expiration', return_value=datetime.now())
        @patch(f'connectchain.utils.token_util.urllib', return_value=None)
        @patch('os.path.getsize', return_value=token_util.config.cert.cert_size)
        def test_fails_if_expired(*args):
            with self.assertRaisesRegex(UtilException, "Certificate expired, please renew"):
                token_util._TokenUtil__retrieve_cert(token_util.config.models['1'])

        @patch(f'{__name__}.TokenUtil.read_cert', return_value="mock_cert")
        @patch(f'{__name__}.TokenUtil.get_cert_expiration', return_value=cert_expire_tomorrow)
        @patch(f'connectchain.utils.token_util.urllib', return_value=None)
        @patch('os.path.getsize', return_value=token_util.config.cert.cert_size)
        def test_success(*args):
            token_util._TokenUtil__retrieve_cert(token_util.config.models['1'])

        @patch(f'{__name__}.TokenUtil.get_cert_expiration', return_value=cert_expire_tomorrow)
        @patch(f'{__name__}.TokenUtil.read_cert', return_value="mock_cert")
        @patch(f'connectchain.utils.token_util.urllib.request.urlretrieve', return_value=None)
        def test_model_config_default(mock_urllib, mock_read_cert, *args): # pylint: disable=unused-variable
            with patch('os.path.getsize', return_value=token_util.config.cert.cert_size) as _:
                token_util._TokenUtil__retrieve_cert(token_util.config.models['1'])
                mock_urllib.assert_called_once_with(token_util.config.cert.cert_path, token_util.config.cert.cert_name)
                mock_read_cert.assert_called_once_with(token_util.config.cert.cert_name)

        @patch(f'{__name__}.TokenUtil.get_cert_expiration', return_value=cert_expire_tomorrow)
        @patch(f'{__name__}.TokenUtil.read_cert', return_value="mock_cert")
        @patch(f'connectchain.utils.token_util.urllib.request.urlretrieve', return_value=None)
        def test_model_config_override(mock_urllib, mock_read_cert, *args): # pylint: disable=unused-variable
            test_model_cert_path = 'test_path'
            test_model_cert_name = 'test_name'
            test_model_cert_size = 100
            test_model_config = wrap_model_config({
                'cert': {
                    'cert_path': test_model_cert_path,
                    'cert_name': test_model_cert_name,
                    'cert_size': test_model_cert_size
                }
            })
            with patch('os.path.getsize', return_value=test_model_cert_size) as _:
                token_util._TokenUtil__retrieve_cert(test_model_config)
                mock_urllib.assert_called_once_with(test_model_cert_path, test_model_cert_name)
                mock_read_cert.assert_called_once_with(test_model_cert_name)

        test_fails_if_file_empty()
        test_fails_if_expired()
        test_success()
        test_model_config_default()
        test_model_config_override()

    """Tests for `get_token_from_env`"""

    @patch('connectchain.utils.token_util.Config.from_env', return_value=get_mock_config({}))
    def test_token_from_env_no_models(self, *args):
        """ Test for missing id_key environment variable"""
        with self.assertRaisesRegex(UtilException, 'No models defined in config') as _:
            get_token_from_env('other')

    @patch('connectchain.utils.token_util.Config.from_env', return_value=get_mock_config({ 'models': {} }))
    def test_token_from_env_no_model_at_index(self, *args):
        """ Test for missing id_key environment variable"""
        with self.assertRaisesRegex(UtilException, 'Model config at index "other" is not defined') as _:
            get_token_from_env('other')

    @patch('connectchain.utils.token_util.Config.from_env', return_value=get_mock_config({ 'models': { 'other': { 'eas': { 'id_key': 'missing_key' } }} }))
    def test_token_from_env_no_id_key(self, *args):
        """ Test for missing id_key environment variable"""
        with self.assertRaisesRegex(UtilException, 'Environment variable id key "missing_key" not set for model index other') as _:
            get_token_from_env('other')

    @patch.dict(os.environ, { 'my_id': 'test_id' })
    @patch('connectchain.utils.token_util.Config.from_env', return_value=get_mock_config(
        { 'models': { 'other': { 'eas': { 'id_key': 'my_id', 'secret_key': 'missing_key' } }} }))
    def test_token_from_env_no_secret_key(self, *args):
        """ Test for missing id_key environment variable"""
        with self.assertRaisesRegex(UtilException, 'Environment variable secret key "missing_key" not set for model index other') as _:
            get_token_from_env('other')

    @patch.dict(os.environ, { 'my_id': 'test_id', 'my_secret': 'test_secret' })
    @patch('connectchain.utils.TokenUtil.__init__', return_value=None)
    @patch('connectchain.utils.token_util.Config.from_env', return_value=get_mock_config(
        { 'models': { 'other': { 'eas': { 'id_key': 'my_id', 'secret_key': 'my_secret' } }} }))
    @patch('connectchain.utils.TokenUtil.get_token')
    def test_token_from_env_calls_get_token(self, get_token_mock, *args):
        """ Test for missing id_key environment variable"""
        get_token_from_env('other')
        get_token_mock.assert_called_once()
