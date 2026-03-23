import re
import json
import uuid
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode

# 假设你已经有了以下导入（根据你的实际项目路径调整）
from langchain_core.messages import AIMessage, trim_messages
from core.llm import get_llm
from core.logger import get_logger
from graph.tools import search_shops, get_shop_detail_tool, get_shop_vouchers_tool, claim_voucher_tool
from langchain_core.exceptions import OutputParserException

backend_log = get_logger("AGENT_BACKEND")

# 导购专用的工具箱（只能看，不能动钱）
guide_tools = [search_shops, get_shop_detail_tool]
# 交易专用的工具箱（只能查券抢券）
transaction_tools = [get_shop_vouchers_tool, claim_voucher_tool]


def qwen_token_counter(messages):
    text = "".join([m.content for m in messages])
    return int(len(text) / 1.5)



# 角色提示词定义 (Prompts)
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from graph.schemas import SupervisorOutput, AgentOutput

# 实例化解析器
supervisor_parser = JsonOutputParser(pydantic_object=SupervisorOutput)
agent_parser = JsonOutputParser(pydantic_object=AgentOutput)

SUPERVISOR_PROMPT = SystemMessage(content=(
    "你是“黑马点评”项目经理，负责精准路由。请按以下逻辑执行：\n\n"
    "1. **核心思考**：分析专家刚做了什么？数据（如 ID）拿到了吗？是否该停下来等用户说话？\n"
    "2. **路由规则**：\n"
    "   - 搜店、查详情 -> guide\n"
    "   - 查券、下单 -> transaction\n"
    "   - 目标达成或需要用户输入 -> FINISH\n\n"
    f"【绝对输出规范】:\n{supervisor_parser.get_format_instructions()}\n"
    "严禁输出任何 Markdown 标记 (如 ```json) 或额外说明文字，必须直接输出合法的 JSON 对象！"
))

GUIDE_PROMPT = SystemMessage(content=(
    "你是“黑马点评”资深导购。请遵循以下铁律：\n\n"
    "1. **搜索规范**：泛提问调用 `search_shops`。\n"
    "2. **详情深挖**：必须使用数字 ID 调用 `get_shop_detail_tool`。\n"
    "3. **ID 准则**：仅提取纯数字 ID，写入 extracted_shop_id 字段。\n\n"
    f"【绝对输出规范】:\n{agent_parser.get_format_instructions()}\n"
    "如果你决定不调用工具而是直接回复用户，必须严格按照上述 JSON 格式输出！"
))

TRANSACTION_PROMPT = SystemMessage(content=(
    "你是严谨的交易专家。请执行以下 SOP：\n\n"
    "1. **找 ID**：从共享状态(shared_payload)或历史记录中提取纯数字 shop_id。\n"
    "2. **查券（必做）**：拿到 ID 后，立即调用 `get_shop_vouchers_tool` 查询优惠券。\n"
    "3. **鉴权（禁令）**：\n"
    "   - 仅汇报：若需求只是“看看/查询”，只需展示结果。\n"
    "   - 强制抢购：仅当用户明确要求“抢/买/领”时，才调用 `claim_voucher_tool`。\n"
    "4. **禁止反问**：严禁询问“要不要帮您抢”，根据用户意图直接执行汇报或抢购，完成后交回话筒。\n\n"
    f"【绝对输出规范】:\n{agent_parser.get_format_instructions()}\n"
    "如果你决定不调用工具而是直接回复用户，必须严格按照上述 JSON 格式输出！"
))


# 通用执行器 (底层大模型驱动)
def _run_agent(state: dict, prompt: SystemMessage, tools: list, agent_name: str):
    llm = get_llm().bind_tools(tools)
    dynamic_prompt = SystemMessage(content=(prompt.content))

    trimmed_history = trim_messages(
        state["messages"], max_tokens=5500, strategy="last",
        token_counter=qwen_token_counter, include_system=False
    )

    # 初始化默认的 shared_payload
    shared_payload = state.get("shared_payload", {})
    format_reminder = SystemMessage(
        content="【系统最高指令】: 无论你是否已经查到了结果，你的输出必须且只能是一个合法的 JSON 对象！绝对禁止直接输出自然语言或越界回复！"
    )

    try:
        response = llm.invoke([dynamic_prompt] + trimmed_history + [format_reminder])
    except Exception as e:
        backend_log.error(f"[{agent_name}] LLM 调用失败: {e}")
        response = llm.invoke([prompt] + state["messages"][-1:] + [format_reminder])

    step_report = ""
    display_text = ""

    # 模型决定调用工具
    if response.tool_calls:
        backend_log.info(f"🛠️ [{agent_name}] 执行工具: {response.tool_calls[0]['name']}")
        step_report = f"{agent_name} 正在执行工具 {response.tool_calls[0]['name']}"
        response.content = ""  # 清空乱码

    # 模型输出文本（现在必须是 JSON 格式了！）
    else:
        raw_text = response.content
        backend_log.debug(f"[{agent_name}] 尝试解析 JSON: {raw_text}")

        try:
            # 核心：使用 Pydantic 解析器将字符串转为 Python 字典
            parsed_data = agent_parser.parse(raw_text)

            # 提取结构化数据
            backend_log.info(f"[{agent_name}] 推理逻辑: {parsed_data.get('thinking')}")
            display_text = parsed_data.get('reply_to_user', '')

            # 实体加锁：提取到了 ID，立刻保存到 shared_payload！
            extracted_id = parsed_data.get('extracted_shop_id')
            if extracted_id:
                shared_payload["shop_id"] = extracted_id
                backend_log.info(f"成功锁定核心实体 shop_id: {extracted_id}")

            step_report = f"{agent_name} 回复: {display_text[:20]}..."

            # 把干净的回复文本放回 content，防止前端收到一堆 JSON
            response.content = display_text

        except OutputParserException as e:
            # 自愈防线：如果模型输出了非 JSON 格式
            backend_log.warning(f"[{agent_name}] JSON 解析失败，触发自愈修复...")
            # 这里的简单自愈策略是提取其中的文本，但在高级场景下可以抛回给 LLM 重写
            display_text = raw_text.replace("```json", "").replace("```", "")
            response.content = display_text
            step_report = f"{agent_name} 进行了非结构化回复。"

    return {
        "messages": [response],
        "past_steps": [step_report],
        "shared_payload": shared_payload  # 将带有 shop_id 的 payload 传递下去
    }


# 节点定义 (LangGraph Nodes)
def supervisor_node(state: dict):
    """主管节点：负责任务路由分发及阶段性总结"""
    llm = get_llm()

    # 1. 熔断机制：连续3次 AI 回复无工具调用，强制拦截
    recent_msgs = state["messages"][-3:]
    if len(recent_msgs) == 3 and all(isinstance(m, AIMessage) and not m.tool_calls for m in recent_msgs):
        backend_log.warning("触发系统熔断，强制结束并汇总。")
        breaker_prompt = SystemMessage(
            content="作为整理专家，请基于对话历史直接汇总已查到的店铺或优惠券信息")
        response = llm.invoke([breaker_prompt] + state["messages"])
        return {"next_agent": "FINISH", "messages": [response]}

    # 2. 历史修剪：控制上下文长度
    trimmed_history = trim_messages(
        state["messages"], max_tokens=5500, strategy="last",
        token_counter=qwen_token_counter, include_system=False
    )

    # 3. 决策调用：组装 Prompt 并执行
    backend_log.info("主管正在进行意图分析与路由决策...")
    prompt_content = SUPERVISOR_PROMPT.content
    if past_steps := state.get("past_steps", []):
        prompt_content += f"\n\n【专家执行日志】:\n" + "\n".join(past_steps)

    supervisor_reminder = SystemMessage(
        content="【警告】：你只是幕后路由主管，绝不允许直接回答用户的问题！即使目标已达成，你也只能输出包含 next_agent 为 'FINISH' 的 JSON 对象！"
    )
    response = llm.invoke([SystemMessage(content=prompt_content)] + trimmed_history + [supervisor_reminder])
    raw_text = str(response.content)
    backend_log.info(f"[主管原始内容]:\n{raw_text[:200]}")


    # 提取日志与指令
    try:
        clean_text = raw_text.replace("<tool_call>", "").replace("</tool_call>", "").strip()
        clean_text = clean_text.replace("```json", "").replace("```", "").strip()
        # 使用我们在外层定义好的解析器强制将大模型输出转为 Python 字典
        parsed_data = supervisor_parser.parse(clean_text)

        # 安全地获取 JSON 中的字段
        thinking = parsed_data.get("thinking", "无推理过程")
        next_agent = parsed_data.get("next_agent", "FINISH")  # 默认安全兜底

        backend_log.info(f"[主管思考]:\n{thinking}")
        backend_log.info(f"主管派单: {next_agent}")

    except OutputParserException as e:
        # 容错防线
        backend_log.error(f"[Supervisor] JSON 解析失败: {e}。原始文本: {raw_text}")
        backend_log.warning("触发兜底机制：强制路由至 FINISH 以保护系统不崩溃")
        next_agent = "FINISH"
    except Exception as e:
        # 捕获其他意料之外的错误
        backend_log.error(f"[Supervisor] 发生未知解析错误: {e}")
        next_agent = "FINISH"

    # 5. 分支流转处理 (保持不变)
    if next_agent == "FINISH":
        backend_log.info("任务阶段结束，生成最终回复...")
        summary_prompt = SystemMessage(content=(
            "作为业务总结专家，请根据对话向用户输出最终答复：\n"
            "1. 简明、有温度地展示已找到的店铺或优惠券结果。\n"
            "2. 若专家此前已提出反问（如'您想看哪家'），请优雅地复述该问题，交出话语权。\n"
            "严禁暴露内部过程或说'无权限'，直接作为贴心助手回答。"
        ))
        # 这里改用 trimmed_history，防止对话轮数过多时引发 Token 超限崩溃
        final_summary = llm.invoke([summary_prompt] + trimmed_history)
        return {"next_agent": "FINISH", "messages": [final_summary]}

    return {"next_agent": next_agent}


def guide_agent_node(state: dict):
    return _run_agent(state, GUIDE_PROMPT, guide_tools, "导购专家")


def transaction_agent_node(state: dict):
    return _run_agent(state, TRANSACTION_PROMPT, transaction_tools, "交易专家")


# 工具节点定义
def guide_tools_node(state: dict, config):
    backend_log.info("[导购工具] 开始执行...")
    result = ToolNode(guide_tools).invoke(state, config=config)
    return result


def transaction_tools_node(state: dict, config):
    backend_log.info("[交易工具] 开始执行...")
    result = ToolNode(transaction_tools).invoke(state, config=config)
    return result

