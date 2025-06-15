"""This module provides a retry mechanism for LCEL-compliant Runnables."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable.utils import Input

from ..utils import abase_retry, base_retry


class LCELRetry(Runnable):
    """This class provides a retry mechanism for LCEL-compliant Runnables.

    Args:
        runnable (Runnable): The runnable to retry.
        max_retry (int): The maximum number of retries.
        sleep_time (int): The time to sleep between retries.
        exceptions (Union[List[Exception], Exception]): The exceptions to catch.
        ebo (bool): Whether to use exponential backoff.
        log_func (callable): The function to use for logging."""

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        runnable: Runnable,
        max_retry: int = 3,
        sleep_time: int = 1,
        exceptions: Union[Tuple[type[BaseException], ...], type[BaseException]] = Exception,
        ebo: bool = False,
        log_func: Any = print,
    ) -> None:
        # Python enforces exceptions must be a tuple.
        if isinstance(exceptions, (list, List)):
            exceptions = tuple(exceptions)  # type: ignore[unreachable]
        self.runnable = runnable
        self.max_retry = max_retry
        self.exceptions = exceptions
        self.sleep_time = sleep_time
        self.ebo = ebo
        # Set up logging function
        if hasattr(log_func, "debug"):
            self.log_func = log_func.debug
        else:
            # For callable objects (including print) or anything else, use as-is
            self.log_func = log_func

    def _get_kwargs_(
        self,
        func: Callable[..., Any],
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        return {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "max_retry": self.max_retry,
            "sleep_time": self.sleep_time,
            "exceptions": self.exceptions,
            "ebo": self.ebo,
            "log_func": self.log_func,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        def call_wrapper(*a: Any, **kw: Any) -> Any:
            # Call the runnable's __call__ method if it exists, otherwise use invoke
            if hasattr(self.runnable, "__call__"):
                return self.runnable.__call__(*a, **kw)
            return self.runnable.invoke(*a, **kw)

        return base_retry(**self._get_kwargs_(call_wrapper, args, kwargs))

    async def __acall__(self, *args: Any, **kwargs: Any) -> Any:
        async def async_wrapper(*a: Any, **kw: Any) -> Any:
            return await self.runnable.ainvoke(*a, **kw)

        return await abase_retry(**self._get_kwargs_(async_wrapper, args, kwargs))

    # pylint: disable=W0622, W0221 # LangChain overrode input
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Any:
        return base_retry(
            **self._get_kwargs_(self.runnable.invoke, kwargs={"input": input, "config": config})
        )

    # pylint: disable=W0622 # LangChain overrode input
    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        return await abase_retry(
            **self._get_kwargs_(
                self.runnable.ainvoke, kwargs={"input": input, "config": config, **kwargs}
            )
        )
