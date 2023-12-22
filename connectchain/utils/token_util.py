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
"""token_util is the utility class to get the bearer token from the environment variables"""
from datetime import datetime
from typing import Final, Any
import asyncio
import os
import time
import hashlib
import hmac
import base64
import uuid
import urllib
import aiohttp
from dotenv import load_dotenv, find_dotenv
from OpenSSL import crypto as c
from connectchain.utils import Config

# There are 3 environment variables that need to be set: CONFIG_PATH:
# path to the config file Consumer ID: the consumer integration ID.
# The environment variable name is defined in the config file under
# eas.id_key Consumer
# Secret: the consumer secret. The environment variable name is
# defined in the config file under eas.secret_key

class UtilException(BaseException):
    """Custom exception class for token_util"""


def get_token_from_env(index: Any = '1') -> str:
    """convenience method to get token from environment variables synchronously"""
    config = Config.from_env()
    try:
        models = config.models
    except KeyError as ex:
        raise UtilException('No models defined in config') from ex
    model_config = models[index]
    if model_config is None:
        raise UtilException(f'Model config at index "{index}" is not defined')
    consumer_id_key = None
    consumer_secret_key = None
    if model_config.eas:
        consumer_id_key = model_config.eas.id_key
        consumer_secret_key = model_config.eas.secret_key
    if consumer_id_key is None:
        consumer_id_key = config.eas.id_key
    consumer_id = os.getenv(f"{consumer_id_key}")
    if consumer_id is None:
        raise UtilException(f'Environment variable id key "{consumer_id_key}" not set for model index {index}')
    if consumer_secret_key is None:
        consumer_secret_key = config.eas.secret_key
    consumer_secret = os.getenv(f"{consumer_secret_key}")
    if consumer_secret is None:
        raise UtilException(f'Environment variable secret key "{consumer_secret_key}" not set for model index {index}')
    return asyncio.run(TokenUtil(consumer_id, consumer_secret, config).get_token(model_config))


# pylint: disable=too-few-public-methods
class TokenUtil:
    """TokenUtil class to get bearer token from environment variables"""
    __SERVICE_VERSION: Final[str] = "2"
    __BYTE_ARRAY_ENCODING: Final[str] = "utf-8"

    # create constructor that takes Config class as parameter
    def __init__(self, consumer_id: str, consumer_secret: str, config: Config):
        self.consumer_id = consumer_id
        self.consumer_secret = consumer_secret
        self.config = config

    def __retrieve_cert(self, model_config):
        """retrieve certificate from the url in the config file if it does not exist locally"""
        cert_path = None
        cert_name = None
        cert_size = None
        if model_config.cert:
            cert_path = model_config.cert.cert_path
            cert_name = model_config.cert.cert_name
            cert_size = model_config.cert.cert_size
        if cert_path is None:
            cert_path = self.config.cert.cert_path
        if cert_name is None:
            cert_name = self.config.cert.cert_name
        if cert_size is None:
            cert_size = self.config.cert.cert_size
        urllib.request.urlretrieve(cert_path, cert_name)
        # check whether the certificate exists locally
        if not os.path.getsize("./" + cert_name) == cert_size:
            raise UtilException("Failed to Download the certificate")
        # check the expiration date of the certificate
        cert_data = TokenUtil.read_cert(cert_name)
        cert_expires = TokenUtil.get_cert_expiration(cert_data)
        if cert_expires < datetime.now():
            raise UtilException("Certificate expired, please renew")

    @staticmethod
    def read_cert(cert_name):
        """read certificate from the local file"""
        with open(cert_name, "r", encoding="utf-8") as reader:
            cert_data = reader.read()
        return cert_data

    @staticmethod
    def get_cert_expiration(cert_data):
        """ get the expiration date of the certificate """
        date_str = c.load_certificate(c.FILETYPE_PEM, cert_data).get_notAfter().decode("UTF-8")
        return datetime.strptime(date_str, "%Y%m%d%H%M%SZ")

    def __service_payload(self, model_config) -> dict[str, dict[str, any]]:
        """payload to get the bearer token"""
        scope = None
        originator_source = None
        if model_config.eas:
            scope = model_config.eas.scope
            originator_source = model_config.eas.originator_source
        if scope is None:
            scope = self.config.eas.scope
        if originator_source is None:
            originator_source = self.config.eas.originator_source
        return {
            'scope': scope,
            'additional_claims': {'originator_source': originator_source}
        }

    @staticmethod
    def __headers(correlation_id, app_id, version, signature, timestamp):
        """headers to get the bearer token"""
        return {
            'Content-Type': 'application/json',
            'X-Auth-AppID': app_id,
            'X-Auth-Version': version,
            'X-Auth-Signature': signature,
            'X-Auth-Timestamp': str(timestamp),
            'X-CorrelationID': correlation_id
        }

    @staticmethod
    async def __aio_http_post(correlation_id,  # pylint: disable=unused-argument, too-many-arguments
                              sor_name,  # pylint: disable=unused-argument
                              url,
                              json,
                              req_headers,
                              timeout,
                              success_codes=(200,),  # pylint: disable=unused-argument
                              cookies=None,
                              proxies=None) -> tuple:
        """aiohttp post method"""
        async with aiohttp.ClientSession() as session:
            start_time = datetime.now()  # pylint: disable=unused-argument, unused-variable
            async with session.post(url, json=json, headers=req_headers, timeout=timeout, ssl=False,
                                    cookies=cookies,
                                    proxy=proxies) as response:
                return await response.json(content_type=None), response.status

    @staticmethod
    def __response_builder(out, status_code) -> str:
        if status_code == 200:
            return f"Bearer {out['authorization_token']}"
        raise UtilException(out['description'])

    def __get_signature(self, version, timestamp):
        message = f'{self.consumer_id}-{version}-{str(timestamp)}'
        input_byte = bytearray(message, TokenUtil.__BYTE_ARRAY_ENCODING)
        decoded_secret = base64.b64decode(self.consumer_secret)
        signature = base64.urlsafe_b64encode(hmac.new(decoded_secret, input_byte, digestmod=hashlib.sha256).digest())
        signature = signature[:-1].decode(TokenUtil.__BYTE_ARRAY_ENCODING)
        return signature

    async def get_token(self, model_config) -> str:
        """async method to get the bearer token"""
        cert_name = None
        if model_config.cert:
            cert_name = model_config.cert.cert_name
        if cert_name is None:
            cert_name = self.config.cert.cert_name
        if not os.path.exists(cert_name):
            self.__retrieve_cert(model_config)
        correlation_id = uuid.uuid1().hex
        version = TokenUtil.__SERVICE_VERSION
        timestamp = int(time.time() * 1000.0)
        signature = self.__get_signature(version, timestamp)
        sor_name = "dummy"
        eas_url = None
        if model_config.eas:
            eas_url = model_config.eas.url
        if eas_url is None:
            eas_url = self.config.eas.url
        timeout = 5

        response = await TokenUtil.__aio_http_post(
            correlation_id,
            sor_name,
            eas_url,
            self.__service_payload(model_config),
            TokenUtil.__headers(correlation_id, self.consumer_id, version, signature, timestamp),
            timeout)

        return TokenUtil.__response_builder(response[0], response[1])


if __name__ == "__main__":  # pragma: no cover
    load_dotenv(find_dotenv())
    auth_token = get_token_from_env()
    print(auth_token)
