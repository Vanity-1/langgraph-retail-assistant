# src/assistants.py
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from .prompts import sales_rep_prompt, support_prompt
from .state import State
from .tools import (
    DEFAULT_USER_ID,
    EscalateToHuman,
    RouteToCustomerSupport,
    cart_tool,
    search_tool,
    structured_search_tool,
    view_cart,
    set_thread_id,
    set_user_id,
)

load_dotenv()

# LLM is created lazily (first use) so importing this module does not require a
# configured API key. This lets offline tooling / tests import the graph, and
# lets the app surface a clear, actionable message at runtime instead of failing
# at import time.
_llm = None


def _get_llm():
    """Build (once) and return the configured chat model."""
    global _llm
    if _llm is not None:
        return _llm

    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )
        return _llm

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            api_key=OPENAI_API_KEY,
            base_url=os.environ.get("OPENAI_BASE_URL"),  # None → OpenAI default
            temperature=0,
        )
        return _llm

    raise ValueError(
        "No API Key found. Please set GOOGLE_API_KEY or OPENAI_API_KEY in .env"
    )

# Tool registration
sales_tools = [
    RouteToCustomerSupport,
    search_tool,
    structured_search_tool,
    cart_tool,
    view_cart,
]
support_tools = [EscalateToHuman]

# Runnable pipelines (built lazily via _get_llm, mirroring the lazy LLM init so
# the module imports without a configured API key).
_sales_runnable = None
_support_runnable = None


def _get_sales_runnable():
    global _sales_runnable
    if _sales_runnable is None:
        _sales_runnable = sales_rep_prompt.partial(time=datetime.now) | _get_llm().bind_tools(
            sales_tools
        )
    return _sales_runnable


def _get_support_runnable():
    global _support_runnable
    if _support_runnable is None:
        _support_runnable = support_prompt.partial(time=datetime.now) | _get_llm().bind_tools(
            support_tools
        )
    return _support_runnable


async def sales_assistant(
    state: State, config: RunnableConfig, runnable=None
) -> dict:
    """
    LangGraph node function for running the sales assistant LLM agent.
    """
    if runnable is None:
        runnable = _get_sales_runnable()

    # Set context from config
    thread_id = config["configurable"].get("thread_id")
    if thread_id:
        set_thread_id(thread_id)

    # Set default user
    set_user_id(DEFAULT_USER_ID)

    # Run the agent (Supports both sync and async runnables)
    if hasattr(runnable, "ainvoke"):
        result = await runnable.ainvoke(state, config=config)
    else:
        result = runnable.invoke(state, config=config)

    return {"messages": result}


async def support_assistant(state: State, config: RunnableConfig, runnable=None) -> dict:
    if runnable is None:
        runnable = _get_support_runnable()

    thread_id = config["configurable"].get("thread_id")
    if thread_id:
        set_thread_id(thread_id)

    if hasattr(runnable, "ainvoke"):
        result = await runnable.ainvoke(state, config=config)
    else:
        result = runnable.invoke(state, config=config)

    return {"messages": result}
