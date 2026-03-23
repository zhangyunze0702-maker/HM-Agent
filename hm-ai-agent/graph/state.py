from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    past_steps: Annotated[list[str], operator.add]
    next_agent: str
    shared_payload: dict

