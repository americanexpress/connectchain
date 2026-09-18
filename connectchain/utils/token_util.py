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
import asyncio
import base64
import hashlib
import hmac
import os
import time
import urllib
import uuid
from datetime import datetime
from typing import Any, Dict, Final

import aiohttp
from dotenv import find_dotenv, load_dotenv
from OpenSSL import crypto as c

from connectchain.utils import Config
from connectchain.utils.exceptions import NonRetryableError

# There are 3 environment variables that need to be set: CONFIG_PATH:
# path to the config file Consumer ID: the consumer integration ID.
# The environment variable name is defined in the config file under
# eas.id_key Consumer
# Secret: the consumer secret. The environment variable name is
# defined in the config file under eas.secret_key


class UtilException(Exception, NonRetryableError):
    """Custom exception class for token_util. Raised for permanent config errors
    (missing models, undefined index, unset env keys, expired cert) that can
    never succeed on retry -- see NonRetryableError."""


class EASAuthError(Exception):
    """Raised when a call to the EAS auth service fails (any non-200 response).

    Deliberately a plain, retry-eligible Exception -- NOT a UtilException --
    because the retryability of an auth-service failure is unknown at raise
    time: a transient 5xx or throttling response may well succeed on retry.
    UtilException carries the NonRetryableError marker (it is reserved for
    permanent config errors), so raising it here would make retry-wrapped
    token fetches fail fast on recoverable server errors."""


def get_token_from_env(index: Any = "1") -> str:
    """convenience method to get token from environment variables synchronously"""
    config = Config.from_env()
    try:
        models = config.models
    except AttributeError as ex:
        # Config raises AttributeError for a missing top-level key.
        raise UtilException("No models defined in config") from ex
    model_config = models[index]
    if model_config is None:
        raise UtilException(f'Model config at index "{index}" is not defined')
    # A partial per-model eas section overrides the global one key-by-key; each
    # key is therefore optional at each level, hence getattr's None defaults
    # (missing keys raise AttributeError now). The global `config.eas` section
    # itself stays a strict read: reaching the fallback without one is a real
    # misconfiguration that should fail loudly.
    model_eas = getattr(model_config, "eas", None)
    consumer_id_key = getattr(model_eas, "id_key", None)
    consumer_secret_key = getattr(model_eas, "secret_key", None)
    if consumer_id_key is None:
        consumer_id_key = getattr(config.eas, "id_key", None)
    consumer_id = os.getenv(f"{consumer_id_key}")
    if consumer_id is None:
        raise UtilException(
            f'Environment variable id key "{consumer_id_key}" not set for model index {index}'
        )
    if consumer_secret_key is None:
        consumer_secret_key = getattr(config.eas, "secret_key", None)
    consumer_secret = os.getenv(f"{consumer_secret_key}")
    if consumer_secret is None:
        raise UtilException(
            f'Environment variable secret key "{consumer_secret_key}" not set for model index {index}'
        )
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

    def __retrieve_cert(self, model_config: Any) -> None:
        """retrieve certificate from the url in the config file if it does not exist locally"""
        # Per-model cert settings override the global ones key-by-key, and every
        # key is optional at each level (the guards below skip whatever is
        # unset), hence getattr's None defaults throughout. The global
        # `self.config.cert` section reads stay strict on purpose: they are
        # only reached when a key is otherwise unresolved, and a missing global
        # cert section then fails loudly rather than silently skipping cert
        # verification.
        model_cert = getattr(model_config, "cert", None)
        cert_path = getattr(model_cert, "cert_path", None)
        cert_name = getattr(model_cert, "cert_name", None)
        cert_size = getattr(model_cert, "cert_size", None)
        if cert_path is None:
            cert_path = getattr(self.config.cert, "cert_path", None)
        if cert_name is None:
            cert_name = getattr(self.config.cert, "cert_name", None)
        if cert_size is None:
            cert_size = getattr(self.config.cert, "cert_size", None)
        if cert_path and cert_name:
            urllib.request.urlretrieve(str(cert_path), str(cert_name))
        # check whether the certificate exists locally
        if cert_name and cert_size and not os.path.getsize("./" + str(cert_name)) == cert_size:
            raise UtilException("Failed to Download the certificate")
        # check the expiration date of the certificate
        if cert_name:
            cert_data = TokenUtil.read_cert(str(cert_name))
            cert_expires = TokenUtil.get_cert_expiration(cert_data)
            if cert_expires < datetime.now():
                raise UtilException("Certificate expired, please renew")

    @staticmethod
    def read_cert(cert_name: str) -> str:
        """read certificate from the local file"""
        with open(cert_name, "r", encoding="utf-8") as reader:
            cert_data = reader.read()
        return cert_data

    @staticmethod
    def get_cert_expiration(cert_data: str) -> datetime:
        """get the expiration date of the certificate"""
        date_str = c.load_certificate(c.FILETYPE_PEM, cert_data).get_notAfter().decode("UTF-8")
        return datetime.strptime(date_str, "%Y%m%d%H%M%SZ")

    def __service_payload(self, model_config: Any) -> Dict[str, Any]:
        """payload to get the bearer token"""
        # Same per-model-overrides-global, key-by-key optionality as elsewhere
        # in this module, hence getattr's None defaults.
        model_eas = getattr(model_config, "eas", None)
        scope = getattr(model_eas, "scope", None)
        originator_source = getattr(model_eas, "originator_source", None)
        if scope is None:
            scope = getattr(self.config.eas, "scope", None)
        if originator_source is None:
            originator_source = getattr(self.config.eas, "originator_source", None)
        return {"scope": scope, "additional_claims": {"originator_source": originator_source}}

    @staticmethod
    def __headers(
        correlation_id: str, app_id: str, version: str, signature: str, timestamp: int
    ) -> Dict[str, str]:
        """headers to get the bearer token"""
        return {
            "Content-Type": "application/json",
            "X-Auth-AppID": app_id,
            "X-Auth-Version": version,
            "X-Auth-Signature": signature,
            "X-Auth-Timestamp": str(timestamp),
            "X-CorrelationID": correlation_id,
        }

    @staticmethod
    async def __aio_http_post(
        correlation_id: str,  # pylint: disable=unused-argument, too-many-arguments
        sor_name: str,  # pylint: disable=unused-argument
        url: str,
        json: Any,
        req_headers: Dict[str, str],
        timeout: Any,
        success_codes: tuple = (200,),  # pylint: disable=unused-argument
        cookies: Any = None,
        proxies: Any = None,
    ) -> tuple:
        """aiohttp post method"""
        async with aiohttp.ClientSession() as session:
            start_time = datetime.now()  # pylint: disable=unused-argument, unused-variable
            async with session.post(
                url,
                json=json,
                headers=req_headers,
                timeout=timeout,
                ssl=False,
                cookies=cookies,
                proxy=proxies,
            ) as response:
                return await response.json(content_type=None), response.status

    @staticmethod
    def __response_builder(out: Any, status_code: int) -> str:
        if status_code == 200:
            return f"Bearer {out['authorization_token']}"
        # Non-200 from the auth service may be transient (5xx, throttling), so
        # raise the retry-eligible EASAuthError rather than the
        # NonRetryableError-marked UtilException -- see EASAuthError.
        raise EASAuthError(out["description"])

    def __get_signature(self, version: str, timestamp: int) -> str:
        message = f"{self.consumer_id}-{version}-{str(timestamp)}"
        input_byte = bytearray(message, TokenUtil.__BYTE_ARRAY_ENCODING)
        decoded_secret = base64.b64decode(self.consumer_secret)
        signature = base64.urlsafe_b64encode(
            hmac.new(decoded_secret, input_byte, digestmod=hashlib.sha256).digest()
        )
        signature_str = signature[:-1].decode(TokenUtil.__BYTE_ARRAY_ENCODING)
        return signature_str

    async def get_token(self, model_config: Any) -> str:
        """async method to get the bearer token"""
        # Per-model override falls back to the global cert section (see
        # __retrieve_cert for why the reads use getattr's None default).
        cert_name = getattr(getattr(model_config, "cert", None), "cert_name", None)
        if cert_name is None:
            cert_name = getattr(self.config.cert, "cert_name", None)
        if cert_name and not os.path.exists(str(cert_name)):
            self.__retrieve_cert(model_config)
        correlation_id = uuid.uuid1().hex
        version = TokenUtil.__SERVICE_VERSION
        timestamp = int(time.time() * 1000.0)
        signature = self.__get_signature(version, timestamp)
        sor_name = "dummy"
        # Per-model override falls back to the global eas section, key-by-key.
        eas_url = getattr(getattr(model_config, "eas", None), "url", None)
        if eas_url is None:
            eas_url = getattr(self.config.eas, "url", None)
        timeout = 5

        response = await TokenUtil.__aio_http_post(
            correlation_id,
            sor_name,
            str(eas_url),
            self.__service_payload(model_config),
            TokenUtil.__headers(correlation_id, self.consumer_id, version, signature, timestamp),
            timeout,
        )

        return TokenUtil.__response_builder(response[0], response[1])


if __name__ == "__main__":  # pragma: no cover
    load_dotenv(find_dotenv())
    auth_token = get_token_from_env()
    print(auth_token)
