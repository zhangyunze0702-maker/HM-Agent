from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    past_steps: Annotated[list[str], operator.add]
    next_agent: str
    # 💡 新增：用于存放专家提取的结构化数据（如 shop_id），彻底告别在历史消息里捞 ID
    shared_payload: dict

