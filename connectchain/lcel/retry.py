""" This module provides a retry mechanism for LCEL-compliant Runnables. """
from typing import Any, Union, Tuple, List, Optional, Callable
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable.utils import Input, Output

from ..utils import base_retry, abase_retry


class LCELRetry(Runnable):
    """This class provides a retry mechanism for LCEL-compliant Runnables.
    
    Args:
        runnable (Runnable): The runnable to retry.
        max_retry (int): The maximum number of retries.
        sleep_time (int): The time to sleep between retries.
        exceptions (Union[List[Exception], Exception]): The exceptions to catch.
        ebo (bool): Whether to use exponential backoff.
        log_func (callable): The function to use for logging."""

    def __init__( # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        runnable,
        max_retry=3,
        sleep_time=1,
        exceptions: Union[Tuple[Exception], Exception] = Exception,
        ebo: bool = False,
        log_func: Callable = print,
    ):
        # Python enforces exceptions must be a tuple.
        if isinstance(exceptions, List):
            exceptions = tuple(exceptions)
        self.runnable = runnable
        self.max_retry = max_retry
        self.exceptions = exceptions
        self.sleep_time = sleep_time
        self.ebo = ebo
        if callable(log_func):
            self.log_func = log_func
        elif hasattr(log_func, "log"):
            self.log_func = log_func.debug
        else:
            self.log_func = print

    def _get_kwargs_(self, func, args=None, kwargs=None):
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
        return base_retry(**self._get_kwargs_(self.runnable.__call__, args, kwargs))

    async def __acall__(self, *args: Any, **kwargs: Any) -> Any:
        return await abase_retry(**self._get_kwargs_(self.runnable.__acall__, args, kwargs))

    # pylint: disable=W0622, W0221 # LangChain overrode input
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return base_retry(**self._get_kwargs_(self.runnable.invoke, kwargs={ "input": input }))

    # pylint: disable=W0622 # LangChain overrode input
    async def ainvoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        return await abase_retry(**self._get_kwargs_(self.runnable.ainvoke, kwargs={ "input": input, **kwargs }))
