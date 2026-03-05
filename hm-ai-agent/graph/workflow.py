from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import AgentState
# 💡 这里我们预定义了接下来要在 nodes.py 里写的 5 个新节点
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

    # 1. 注册所有部门 (1个主管 + 2个专家 + 2个专用的工具节点)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("guide_agent", guide_agent_node)
    workflow.add_node("transaction_agent", transaction_agent_node)
    workflow.add_node("guide_tools", guide_tools_node)
    workflow.add_node("transaction_tools", transaction_tools_node)

    # 2. 设定大门起点：用户进门，永远先找主管看计划
    workflow.add_edge(START, "supervisor")

    # 3. 🎯 主管路由逻辑：根据 state["next_agent"] 分发任务，或者结束
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_agent", "FINISH"), # 默认没任务就结束
        {
            "guide": "guide_agent",
            "transaction": "transaction_agent",
            "FINISH": END
        }
    )

    # 4. 🍽️ 导购专家的内部循环 (思考 <-> 搜店工具)
    workflow.add_conditional_edges(
        "guide_agent",
        # 如果导购请求了工具，走 guide_tools；如果只是说话，则任务完成，回传给主管
        lambda state: "guide_tools" if state["messages"][-1].tool_calls else "supervisor",
    )
    workflow.add_edge("guide_tools", "guide_agent")

    # 5. 💰 交易专家的内部循环 (思考 <-> 查券/抢券工具)
    workflow.add_conditional_edges(
        "transaction_agent",
        # 如果交易专家请求了工具，走 transaction_tools；否则回传给主管
        lambda state: "transaction_tools" if state["messages"][-1].tool_calls else "supervisor",
    )
    workflow.add_edge("transaction_tools", "transaction_agent")

    # 6. 编译启动！
    app = workflow.compile(checkpointer=memory)
    return app