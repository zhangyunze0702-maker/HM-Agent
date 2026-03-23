from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import AgentState
from graph.nodes import (
    supervisor_node,
    guide_agent_node,
    transaction_agent_node,
    guide_tools_node,
    transaction_tools_node
)

def create_agent_app():
    """
    构建 P&E 层级式多智能体工作流 (Supervisor Architecture)
    """
    memory = MemorySaver()
    workflow = StateGraph(AgentState)

    # 注册
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("guide_agent", guide_agent_node)
    workflow.add_node("transaction_agent", transaction_agent_node)
    workflow.add_node("guide_tools", guide_tools_node)
    workflow.add_node("transaction_tools", transaction_tools_node)

    # 设定起点
    workflow.add_edge(START, "supervisor")

    # 路由逻辑
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_agent", "FINISH"), # 默认没任务就结束
        {
            "guide": "guide_agent",
            "transaction": "transaction_agent",
            "FINISH": END
        }
    )

    # 导购内部循环 (思考 <-> 搜店工具)
    workflow.add_conditional_edges(
        "guide_agent",
        # 如果导购请求了工具，走 guide_tools；如果只是说话，则任务完成，回传给主管
        lambda state: "guide_tools" if state["messages"][-1].tool_calls else "supervisor",
    )
    workflow.add_edge("guide_tools", "guide_agent")

    # 交易内部循环 (思考 <-> 查券/抢券工具)
    workflow.add_conditional_edges(
        "transaction_agent",
        # 如果交易专家请求了工具，走 transaction_tools；否则回传给主管
        lambda state: "transaction_tools" if state["messages"][-1].tool_calls else "supervisor",
    )
    workflow.add_edge("transaction_tools", "transaction_agent")

    app = workflow.compile(checkpointer=memory)
    return app
