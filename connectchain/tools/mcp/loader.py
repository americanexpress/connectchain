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


"""Load MCP tools from configured servers."""

import logging
from typing import List, Optional, cast

from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ...utils.config import Config


class MCPToolLoader:
    """Load MCP tools from configured servers."""

    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[MultiServerMCPClient] = None

    async def load_tools(self, server_names: Optional[List[str]] = None) -> List[BaseTool]:
        """Load tools from MCP servers configured in the YAML file."""
        mcp_config = self.config.data.get("mcp", {})
        servers = mcp_config.get("servers", {})

        if not servers:
            logging.getLogger(__name__).warning(
                "No servers found for tool loading. Check MCP configuration."
            )
            return []

        # Filter servers if specific names requested. `is not None` (not truthiness) so an
        # explicit empty list means "connect to zero servers", not "no filter requested".
        if server_names is not None:
            servers = {k: v for k, v in servers.items() if k in server_names}

        # Pass server configs directly to MultiServerMCPClient
        self.client = MultiServerMCPClient(servers)
        tools = await self.client.get_tools()
        return cast(List[BaseTool], tools)

    async def close(self) -> None:
        """Clean up MCP client connections."""
        if self.client:
            self.client = None
