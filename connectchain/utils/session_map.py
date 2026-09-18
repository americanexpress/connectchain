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
"""This module is used to keep track of the session expiration time"""
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from langchain.schema import LLMResult


class SessionMap:
    """Singleton tracking session expiration time for cached LLM instances.

    is_expired()/get_valid_llm() return safely for an unregistered
    session_id rather than raising KeyError (get_llm() is the one
    exception -- see its docstring). _instance_lock guards first
    construction via double-checked locking; self._lock (created once
    construction completes) guards all session_map reads/writes.
    """

    DEFAULT_EXPIRES_IN: int = 900

    _instance: Optional["SessionMap"] = None
    _instance_lock: threading.Lock = threading.Lock()
    _lock: threading.Lock
    # Each entry stores the expires_in that was active when it was cached, not the
    # singleton's current value -- a later SessionMap(different_interval) call must not
    # retroactively change the expiry policy for sessions cached under a prior interval.
    session_map: Dict[str, Tuple[datetime, int, LLMResult]] = {}
    expires_in: int = DEFAULT_EXPIRES_IN

    def __new__(cls, expires_in: Optional[int] = None) -> "SessionMap":
        """Return the singleton, optionally reconfiguring its default expiry.

        `expires_in=None` (the default) means "don't reconfigure": a bare
        SessionMap() call -- e.g. constructed just to read the cache -- must not
        silently reset a previously configured interval back to the default.
        Passing an explicit value reconfigures the default used for sessions
        cached from here on; already-cached sessions keep the expires_in they
        were cached under (captured per-entry in new_session()).
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    # Fully build the instance on a local var before publishing it to
                    # cls._instance. A racing thread's outer `if cls._instance is None`
                    # check (above, outside the lock) reads cls._instance directly, so
                    # publishing a partially-built instance (e.g. before _lock is set)
                    # would let that thread skip the lock entirely and use an instance
                    # missing _lock, raising AttributeError.
                    new_instance = super(SessionMap, cls).__new__(cls)
                    new_instance.expires_in = (
                        expires_in if expires_in is not None else cls.DEFAULT_EXPIRES_IN
                    )
                    new_instance._lock = threading.Lock()
                    cls._instance = new_instance
                    return cls._instance
        if expires_in is not None:
            # Reconfigure only when the value actually changes. Every EAS-routed
            # model() call constructs SessionMap(expires_in) on its hot path, and
            # the value is almost always the one already configured, so
            # unconditionally taking self._lock -- shared with every session_map
            # cache read/write -- just to rewrite an identical int was pure lock
            # contention. The preliminary comparison deliberately reads
            # expires_in WITHOUT the lock: the read is a single atomic attribute
            # fetch, and the worst a racing reconfiguration can do is make the
            # comparison see the older or newer of two concurrently requested
            # values -- the same last-writer-wins outcome the unconditional
            # locked write had, so the benign race changes no observable
            # semantics. The write itself stays locked so it cannot interleave
            # with new_session()'s locked fallback read of self.expires_in.
            if cls._instance.expires_in != expires_in:
                with cls._instance._lock:
                    cls._instance.expires_in = expires_in
        return cls._instance

    def new_session(
        self, session_id: str, llm: LLMResult, expires_in: Optional[int] = None
    ) -> None:
        """Save new session for later.

        Pass `expires_in` explicitly when the caller captured it before doing
        anything slow (e.g. a network call): reading self.expires_in only at this
        point would race a concurrent SessionMap(other_interval) call made by another
        request in between, silently applying the wrong expiry to this session.
        The None-fallback below is retained only for backward compatibility with
        external callers; connectchain's own call sites always pass an explicit
        int (resolving an unset config interval to DEFAULT_EXPIRES_IN at capture
        time) precisely so this racy read is never exercised.
        """
        with self._lock:
            if expires_in is None:
                expires_in = self.expires_in
            self.session_map[session_id] = (datetime.now(), expires_in, llm)

    @staticmethod
    def _is_stale(entry: Tuple[datetime, int, LLMResult]) -> bool:
        """Whether a cached entry has exceeded the expires_in it was cached under.

        Shared by is_expired() and get_valid_llm() so the staleness rule lives
        in exactly one place. Pure function of the already-fetched entry tuple;
        no lock needed.
        """
        cached_at, expires_in, _ = entry
        return (datetime.now() - cached_at).total_seconds() > expires_in

    def is_expired(self, session_id: str) -> bool:
        """Check if the session is expired.

        Returns True (treat as expired) when session_id is not yet registered
        so that callers trigger a fresh token acquisition rather than raising
        a KeyError.
        """
        with self._lock:
            entry = self.session_map.get(session_id)
            return entry is None or self._is_stale(entry)

    def get_llm(self, session_id: str) -> LLMResult:
        """Get the LLM instance from the session.

        Precondition: caller must confirm is_expired(session_id) returns
        False first (or use get_valid_llm() instead, which does this
        atomically). Raises KeyError if session_id is not registered.
        """
        with self._lock:
            return self.session_map[session_id][2]

    def get_valid_llm(self, session_id: str) -> Optional[LLMResult]:
        """Return the cached LLM for session_id if it exists and is not expired,
        else None. Checks existence and expiry atomically under one lock
        acquisition, unlike calling is_expired() then get_llm() separately."""
        with self._lock:
            entry = self.session_map.get(session_id)
            if entry is None or self._is_stale(entry):
                return None
            return entry[2]

    @staticmethod
    def uuid_from_config(config: Any, model_config: Any) -> str:
        """Generate a stable cache key from the config.

        A partial per-model eas section overrides the global one key-by-key,
        falling back to the global section for whichever keys it omits. Every
        read here uses getattr's None default: this is a cache-key builder,
        not a validator -- keys that are genuinely required are validated at
        their point of use, and a missing optional key must keep rendering as
        the literal "None" placeholder in the key (as it always has) rather
        than raising now that Config/ConfigWrapper are strict about missing
        attributes."""
        model_eas = getattr(model_config, "eas", None)
        env_id_key = getattr(model_eas, "id_key", None)
        if env_id_key is None:
            env_id_key = getattr(getattr(config, "eas", None), "id_key", None)
        env_secret_key = getattr(model_eas, "secret_key", None)
        if env_secret_key is None:
            env_secret_key = getattr(getattr(config, "eas", None), "secret_key", None)
        env_uuid = f"{env_id_key}_{env_secret_key}"
        model_uuid = "_".join(
            str(getattr(model_config, key, None))
            for key in ("provider", "type", "engine", "model_name", "api_version")
        )
        return f"{env_uuid}_{model_uuid}"
