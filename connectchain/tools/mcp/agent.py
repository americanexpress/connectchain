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


"""MCPToolAgent: LCEL-compatible agent for executing MCP tools based on LLM decisions.
This module provides the MCPToolAgent class, which integrates with LCEL and executes
MCP tools as requested by language model outputs. It supports both synchronous and
asynchronous invocation, error handling, and tool result aggregation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage

from ...lcel import model

logger = logging.getLogger(__name__)


class MCPToolAgent(Runnable):  # pylint: disable=redefined-builtin
    """LCEL-compatible agent that can execute MCP tools based on LLM decisions.
    Attributes:
        model_id (str): The model identifier.
        tools (Dict[str, BaseTool]): Mapping of tool names to tool instances.
    """

    def __init__(self, model_id: str, tools: List[BaseTool]):
        self.model_id = model_id
        self.tools: Dict[str, BaseTool] = {}
        for tool in tools:
            if tool.name in self.tools:
                logger.warning(
                    "Duplicate tool name '%s' from multiple servers; the later one "
                    "overrides the earlier one.",
                    tool.name,
                )
            self.tools[tool.name] = tool

    async def ainvoke(
        self,
        input: Any,  # pylint: disable=redefined-builtin
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> dict:
        """Process input through LLM and execute any requested tools."""
        llm = model(self.model_id)
        if hasattr(llm, "bind_tools"):
            llm = llm.bind_tools(list(self.tools.values()))

        # Forward **kwargs so caller-supplied run options (e.g. `stop`, provider-specific
        # parameters) actually reach the underlying model instead of being silently dropped.
        response = await llm.ainvoke(input, config, **kwargs)

        # Extract the content once -- it's the same expression on every return path, and
        # duplicating it at each return site invites the two copies drifting apart.
        content = response.content if isinstance(response, BaseMessage) else str(response)

        if not hasattr(response, "tool_calls") or not response.tool_calls:
            # Match the declared `-> dict` contract on every path, not just when tools
            # are requested -- otherwise callers can't uniformly do result["content"].
            return {"content": content, "tool_results": []}

        async def execute_tool(tool_name: str, tool_args: Any) -> dict:
            """Run a single tool, mapping any failure to an error entry so one failing
            tool never affects its siblings."""
            try:
                result = await self.tools[tool_name].ainvoke(tool_args)
                return {"tool": tool_name, "result": result}
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "Tool '%s' execution failed: %s", tool_name, e
                )
                return {"tool": tool_name, "error": str(e)}

        pending = []
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
            tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else tool_call.args

            if tool_name in self.tools:
                pending.append(execute_tool(tool_name, tool_args))
            else:
                # Unknown tools only produce a warning; they contribute no entry to
                # tool_results (preserving the long-standing behavior).
                logger.warning("Unknown tool requested: %s", tool_name)

        # Run the requested tools concurrently rather than awaiting them one by one.
        # gather() is safe here because the tool calls are independent of each other
        # and MCP sessions handle concurrent calls; per-tool failures are already
        # absorbed inside execute_tool, so one slow or failing tool cannot affect the
        # rest. gather() also preserves ordering: results arrive in the same order as
        # response.tool_calls (minus unknown tools), regardless of completion order.
        results = list(await asyncio.gather(*pending))

        return {"content": content, "tool_results": results}

    def invoke(
        self,
        input: Any,  # pylint: disable=redefined-builtin
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> dict:
        """Synchronous wrapper for the asynchronous ainvoke function."""
        try:
            return asyncio.run(self.ainvoke(input, config, **kwargs))
        except RuntimeError as e:
            raise RuntimeError(
                "MCPToolAgent.invoke() failed. If you are running from within an event loop "
                "(e.g., in a web server or async application), please use the async ainvoke() "
                "method with await instead."
            ) from e

    def _call(
        self,
        input_data: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> dict:
        raise NotImplementedError(
            "MCPToolAgent does not support hidden `_call`. Please use a supported method (`invoke`, etc.)."
        )

    def batch(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,  # pylint: disable=redefined-builtin
    ) -> List[dict]:
        raise NotImplementedError(
            "MCPToolAgent does not support `batch`. Please use a supported method (`invoke`, etc.)."
        )

    async def abatch(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,  # pylint: disable=redefined-builtin
    ) -> List[dict]:
        raise NotImplementedError(
            "MCPToolAgent does not support `abatch`. Please use a supported method (`invoke`, etc.)."
        )
