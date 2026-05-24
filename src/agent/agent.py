import logging
import os
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8").strip()


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_llm(tools: list):
    provider = os.environ.get("MODEL_PROVIDER", "ollama")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "qwen3.5"),
            base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        ).bind_tools(tools)


def build_agent(tools: list, checkpointer=None):
    llm = _build_llm(tools)
    tool_node = ToolNode(tools)

    def call_model(state: State):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response: AIMessage = llm.invoke(messages)
        return {"messages": [response]}

    def call_tools(state: State):
        return tool_node.invoke(state)

    graph = StateGraph(State)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(debug=True, checkpointer=checkpointer)