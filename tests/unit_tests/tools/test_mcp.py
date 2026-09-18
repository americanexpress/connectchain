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

"""Tests for MCP tools."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Skip tests if langchain-mcp-adapters not installed
pytest.importorskip("langchain_mcp_adapters")

from langchain.schema import AIMessage
from langchain.tools import BaseTool

from connectchain.tools.mcp.agent import MCPToolAgent
from connectchain.tools.mcp.loader import MCPToolLoader
from connectchain.utils.config import Config


class TestMCPToolAgent:
    """Test cases for MCPToolAgent."""

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        add_tool = Mock(spec=BaseTool)
        add_tool.name = "add"
        add_tool.ainvoke = AsyncMock(return_value=10)

        multiply_tool = Mock(spec=BaseTool)
        multiply_tool.name = "multiply"
        multiply_tool.ainvoke = AsyncMock(return_value=20)

        return [add_tool, multiply_tool]

    @pytest.mark.asyncio
    async def test_no_tool_calls(self, mock_tools):
        """Test agent behavior when LLM doesn't request tools."""
        with patch("connectchain.tools.mcp.agent.model") as mock_model:
            # Mock LLM response without tool calls
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="The answer is 42")
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.bind_tools = Mock(return_value=mock_llm)
            mock_model.return_value = mock_llm

            agent = MCPToolAgent("1", mock_tools)
            result = await agent.ainvoke({"query": "What is 2+2?"})

            assert result["content"] == "The answer is 42"
            assert result["tool_results"] == []
            assert mock_llm.bind_tools.called

    @pytest.mark.asyncio
    async def test_with_tool_calls(self, mock_tools):
        """Test agent executes requested tools."""
        with patch("connectchain.tools.mcp.agent.model") as mock_model:
            # Mock LLM response with tool calls
            mock_llm = AsyncMock()
            mock_response = AIMessage(
                content="I'll calculate that for you.",
                tool_calls=[
                    {"id": "1", "name": "add", "args": {"a": 5, "b": 5}},
                    {"id": "2", "name": "multiply", "args": {"a": 2, "b": 10}},
                ],
            )
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.bind_tools = Mock(return_value=mock_llm)
            mock_model.return_value = mock_llm

            agent = MCPToolAgent("1", mock_tools)
            result = await agent.ainvoke({"query": "Calculate 5+5 and 2*10"})

            assert result["content"] == "I'll calculate that for you."
            assert len(result["tool_results"]) == 2
            assert result["tool_results"][0] == {"tool": "add", "result": 10}
            assert result["tool_results"][1] == {"tool": "multiply", "result": 20}

    @pytest.mark.asyncio
    async def test_unknown_tool_requested(self, mock_tools):
        """Test agent handles unknown tool gracefully."""
        with patch("connectchain.tools.mcp.agent.model") as mock_model:
            # Mock LLM response requesting unknown tool
            mock_llm = AsyncMock()
            mock_response = AIMessage(
                content="Using unknown tool",
                tool_calls=[{"id": "1", "name": "unknown_tool", "args": {}}],
            )
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.bind_tools = Mock(return_value=mock_llm)
            mock_model.return_value = mock_llm

            agent = MCPToolAgent("1", mock_tools)
            result = await agent.ainvoke({"query": "Use unknown tool"})

            assert result["content"] == "Using unknown tool"
            assert len(result["tool_results"]) == 0

    @pytest.mark.asyncio
    async def test_tool_execution_error(self, mock_tools):
        """Test agent handles tool execution errors."""
        # Make add tool raise an error
        mock_tools[0].ainvoke = AsyncMock(side_effect=Exception("Tool failed"))

        with patch("connectchain.tools.mcp.agent.model") as mock_model:
            mock_llm = AsyncMock()
            mock_response = AIMessage(
                content="Calculating",
                tool_calls=[{"id": "1", "name": "add", "args": {"a": 1, "b": 2}}],
            )
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.bind_tools = Mock(return_value=mock_llm)
            mock_model.return_value = mock_llm

            agent = MCPToolAgent("1", mock_tools)
            result = await agent.ainvoke({"query": "Add 1+2"})

            assert result["tool_results"][0]["tool"] == "add"
            assert result["tool_results"][0]["error"] == "Tool failed"

    @pytest.mark.asyncio
    async def test_tool_calls_run_concurrently_in_order(self, mock_tools):
        """Regression test: independent tool calls must run concurrently, and results
        must stay in tool_calls order even when the first tool finishes last.

        Previously each tool was awaited sequentially in a for loop, so a slow tool
        blocked every tool after it. The rendezvous below is only satisfiable under
        concurrent execution: 'add' (requested first) refuses to finish until
        'multiply' (requested second) has started. Sequential execution would never
        start 'multiply' while 'add' is pending, so 'add' would time out and surface
        as an error entry instead of a result. asyncio.Event keeps this deterministic
        -- no wall-clock sleeps or timing races."""
        started = {"add": asyncio.Event(), "multiply": asyncio.Event()}

        async def slow_add(_args):
            started["add"].set()
            # Completes only once 'multiply' has started -- proof both tools were
            # in flight at the same time. The timeout is a safety net so a regression
            # to sequential execution fails the assertions below instead of hanging.
            await asyncio.wait_for(started["multiply"].wait(), timeout=2)
            return 10

        async def fast_multiply(_args):
            started["multiply"].set()
            return 20

        mock_tools[0].ainvoke = AsyncMock(side_effect=slow_add)
        mock_tools[1].ainvoke = AsyncMock(side_effect=fast_multiply)

        with patch("connectchain.tools.mcp.agent.model") as mock_model:
            mock_llm = AsyncMock()
            mock_response = AIMessage(
                content="I'll calculate that for you.",
                tool_calls=[
                    {"id": "1", "name": "add", "args": {"a": 5, "b": 5}},
                    {"id": "2", "name": "multiply", "args": {"a": 2, "b": 10}},
                ],
            )
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.bind_tools = Mock(return_value=mock_llm)
            mock_model.return_value = mock_llm

            agent = MCPToolAgent("1", mock_tools)
            result = await agent.ainvoke({"query": "Calculate 5+5 and 2*10"})

            # Both tools ran (and overlapped -- otherwise slow_add couldn't return 10).
            assert started["add"].is_set()
            assert started["multiply"].is_set()
            # Ordering matches tool_calls order even though 'add' finished last.
            assert result["tool_results"] == [
                {"tool": "add", "result": 10},
                {"tool": "multiply", "result": 20},
            ]

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_to_llm(self, mock_tools):
        """Regression test: caller-supplied kwargs must reach the underlying llm.ainvoke.

        Previously ainvoke accepted **kwargs but never forwarded them, so run
        options (e.g. `stop`) were silently dropped."""
        with patch("connectchain.tools.mcp.agent.model") as mock_model:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="The answer is 42")
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.bind_tools = Mock(return_value=mock_llm)
            mock_model.return_value = mock_llm

            agent = MCPToolAgent("1", mock_tools)
            await agent.ainvoke({"query": "What is 2+2?"}, None, stop=["\n"], temperature=0.1)

            mock_llm.ainvoke.assert_awaited_once_with(
                {"query": "What is 2+2?"}, None, stop=["\n"], temperature=0.1
            )

    def test_invoke_runtime_error(self, mock_tools):
        """Test synchronous invoke raises RuntimeError with helpful message."""
        with patch("connectchain.tools.mcp.agent.asyncio.run") as mock_asyncio_run:
            # Simulate the RuntimeError that occurs when asyncio.run() is called from within an event loop
            mock_asyncio_run.side_effect = RuntimeError("asyncio.run() cannot be called from a running event loop")
            
            agent = MCPToolAgent("1", mock_tools)

            with pytest.raises(RuntimeError, match="MCPToolAgent.invoke\\(\\) failed"):
                agent.invoke({"query": "test"})


class TestMCPToolLoader:
    """Test cases for MCPToolLoader."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with MCP settings."""
        config = Mock(spec=Config)
        config.data = {
            "mcp": {
                "servers": {
                    "math_tools": {
                        "command": "python",
                        "args": ["math_server.py"],
                        "transport": "stdio",
                    },
                    "web_tools": {
                        "url": "https://example.com",
                        "transport": "streamable-http",
                        "auth": {"type": "bearer"},
                    },
                }
            }
        }
        config.proxy = "http://proxy.company.com:8080"
        config.eas = "https://eas.company.com"
        config.cert = "/path/to/cert.pem"
        return config

    @pytest.mark.asyncio
    async def test_load_all_tools(self, mock_config):
        """Test loading tools from all configured servers."""
        with patch("connectchain.tools.mcp.loader.MultiServerMCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools = AsyncMock(
                return_value=[Mock(name="add"), Mock(name="multiply")]
            )
            mock_client_class.return_value = mock_client

            loader = MCPToolLoader(mock_config)
            tools = await loader.load_tools()

            assert len(tools) == 2
            assert mock_client_class.called
            assert mock_client.get_tools.called

    @pytest.mark.asyncio
    async def test_load_specific_tools(self, mock_config):
        """Test loading tools from specific servers."""
        with patch("connectchain.tools.mcp.loader.MultiServerMCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools = AsyncMock(return_value=[Mock(name="add")])
            mock_client_class.return_value = mock_client

            loader = MCPToolLoader(mock_config)
            await loader.load_tools(["math_tools"])

            # Check that only math_tools was passed to client
            call_args = mock_client_class.call_args[0][0]
            assert "math_tools" in call_args
            assert "web_tools" not in call_args


    @pytest.mark.asyncio
    async def test_no_servers_configured(self):
        """Test behavior when no MCP servers are configured."""
        config = Mock(spec=Config)
        config.data = {}

        loader = MCPToolLoader(config)
        tools = await loader.load_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_close(self, mock_config):
        """Test cleanup of MCP client."""
        loader = MCPToolLoader(mock_config)
        loader.client = Mock()

        await loader.close()
        assert loader.client is None