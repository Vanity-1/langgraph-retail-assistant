# src/web_search_mcp.py
import os
import asyncio
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)

BRAVE_MODEL_NAME = "brave-search"


@tool
def brave_search_unavailable(query: str) -> str:
    """Fallback when no BRAVE_API_KEY is configured or the MCP server fails.

    Explains that live web search is disabled so the agent can degrade gracefully
    instead of silently returning empty results.
    """
    return (
        "Live web search is unavailable (no BRAVE_API_KEY or the Brave MCP server "
        "could not be started). For up-to-date web information, enable it by setting "
        "BRAVE_API_KEY in .env and restarting the app."
    )


async def _load_brave_tool():
    """Load the Brave Search MCP tool when an API key is present.

    Behavior:
    1. Reads BRAVE_API_KEY from the environment.
    2. Without a key, returns a fallback tool explaining web search is disabled.
    3. With a key, connects to the Brave MCP server (via npx stdio transport).
    4. On any error, logs the failure and returns the fallback tool.

    Returns:
        List of tools (always a list, never raises).
    """
    brave_api_key = os.environ.get("BRAVE_API_KEY", "").strip()

    if not brave_api_key:
        logger.info("BRAVE_API_KEY missing; returning fallback web search tool.")
        return [brave_search_unavailable]

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                BRAVE_MODEL_NAME: {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@brave/brave-search-mcp-server",
                        "--transport",
                        "stdio",
                        "--brave-api-key",
                        brave_api_key,
                    ],
                    "transport": "stdio",
                }
            }
        )
        tools = await client.get_tools()
        logger.info("Loaded %d tools from Brave MCP server.", len(tools))
        return tools
    except Exception as e:  # pragma: no cover - network/env dependent
        logger.warning("Failed to start Brave MCP server: %s. Using fallback.", e)
        return [brave_search_unavailable]


def get_brave_web_search_tool_sync():
    """Safe sync wrapper for Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an event loop → schedule and return immediately if possible
        future = asyncio.ensure_future(_load_brave_tool())
        try:
            return future.result()
        except asyncio.InvalidStateError:
            return [brave_search_unavailable]
    else:
        return asyncio.run(_load_brave_tool())
