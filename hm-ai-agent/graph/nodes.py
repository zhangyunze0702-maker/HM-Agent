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

# ==========================================
# 🧰 1. 工具箱拆分 (技能解耦)
# ==========================================
# 导购专用的工具箱（只能看，不能动钱）
guide_tools = [search_shops, get_shop_detail_tool]
# 交易专用的工具箱（只能查券抢券）
transaction_tools = [get_shop_vouchers_tool, claim_voucher_tool]


def qwen_token_counter(messages):
    text = "".join([m.content for m in messages])
    return int(len(text) / 1.5)


# ==========================================
# 📝 2. 角色提示词定义 (Prompts)
# ==========================================
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


# ==========================================
# 通用执行器 (底层大模型驱动)
# ==========================================
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
        backend_log.error(f"❌ [{agent_name}] LLM 调用失败: {e}")
        response = llm.invoke([prompt] + state["messages"][-1:] + [format_reminder])

    step_report = ""
    display_text = ""

    # 🔀 分支 1：模型决定调用工具
    if response.tool_calls:
        backend_log.info(f"🛠️ [{agent_name}] 执行工具: {response.tool_calls[0]['name']}")
        step_report = f"{agent_name} 正在执行工具 {response.tool_calls[0]['name']}"
        response.content = ""  # 清空乱码

    # 🔀 分支 2：模型输出文本（现在必须是 JSON 格式了！）
    else:
        raw_text = response.content
        backend_log.debug(f"🔍 [{agent_name}] 尝试解析 JSON: {raw_text}")

        try:
            # 核心：使用 Pydantic 解析器将字符串转为 Python 字典
            parsed_data = agent_parser.parse(raw_text)

            # 提取结构化数据
            backend_log.info(f"🧠 [{agent_name}] 推理逻辑: {parsed_data.get('thinking')}")
            display_text = parsed_data.get('reply_to_user', '')

            # 实体加锁：提取到了 ID，立刻保存到 shared_payload！
            extracted_id = parsed_data.get('extracted_shop_id')
            if extracted_id:
                shared_payload["shop_id"] = extracted_id
                backend_log.info(f"🔒 成功锁定核心实体 shop_id: {extracted_id}")

            step_report = f"{agent_name} 回复: {display_text[:20]}..."

            # 把干净的回复文本放回 content，防止前端收到一堆 JSON
            response.content = display_text

        except OutputParserException as e:
            # 自愈防线：如果模型发疯输出了非 JSON 格式
            backend_log.warning(f"⚠️ [{agent_name}] JSON 解析失败，触发自愈修复...")
            # 这里的简单自愈策略是提取其中的文本，但在高级场景下可以抛回给 LLM 重写
            display_text = raw_text.replace("```json", "").replace("```", "")
            response.content = display_text
            step_report = f"{agent_name} 进行了非结构化回复。"

    return {
        "messages": [response],
        "past_steps": [step_report],
        "shared_payload": shared_payload  # 💡 将带有 shop_id 的 payload 传递下去
    }


# def _run_agent(state: dict, prompt: SystemMessage, tools: list, agent_name: str):
#     """将你之前的 agent_node 封装成通用的专家引擎"""
#     llm = get_llm().bind_tools(tools)
#
#     dynamic_prompt = SystemMessage(content=(
#             prompt.content + "请根据上述计划中属于你的职责部分进行操作。"
#     ))
#
#     trimmed_history = trim_messages(
#         state["messages"], max_tokens=5500, strategy="last",
#         token_counter=qwen_token_counter, include_system=False
#     )
#
#     try:
#         response = llm.invoke([dynamic_prompt] + trimmed_history)
#     except Exception as e:
#         backend_log.error(f"❌ [{agent_name}] LLM 调用失败: {e}")
#         response = llm.invoke([prompt] + state["messages"][-1:])
#
#     raw_text = response.content if isinstance(response.content, str) else ""
#     backend_log.debug(f"🔍 [{agent_name}] 原始输出原文: {raw_text}")
#
#     display_text = raw_text
#
#     thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw_text, re.DOTALL)
#     if thinking_match:
#         backend_log.info(f"🧠 [{agent_name}] 内部推理逻辑:\n{thinking_match.group(1).strip()}")
#         display_text = re.sub(r'<thinking>.*?</thinking>', '', raw_text, flags=re.DOTALL).strip()
#
#     if not response.tool_calls and "<tool_call>" in display_text:
#         response = _salvage_tool_calls(response, display_text)
#         display_text = response.content
#
#     step_report = ""  # 准备发给主管的“简报”
#
#     if response.tool_calls:
#         # 情况 1: 专家决定动用工具
#         backend_log.info(f"🛠️ [{agent_name}] 拦截到工具调用...")
#         for tc in response.tool_calls:
#             backend_log.info(f"🛠️ 执行工具: {tc['name']} | 参数: {tc['args']}")
#
#         # 汇报日志：告诉主管我正在调用什么工具，参数是什么
#         step_report = f"{agent_name} 正在执行工具 {response.tool_calls[0]['name']}，参数为 {response.tool_calls[0]['args']}"
#
#         # 为了前端洁净，清空 content
#         response.content = ""
#     else:
#         # 情况 2: 专家完成了阶段性回复
#         backend_log.info(f"💬 [{agent_name}] 汇报工作结论: {display_text}")
#         # 汇报日志：把专家最后说的关键结论摘要给主管
#         step_report = f"{agent_name} 已完成任务，其结论摘要为: {display_text}..."
#         response.content = ""
#
#     # 💡 返回时增加 past_steps 字段，由于在 AgentState 中用了 operator.add，这里会自动追加
#     return {
#         "messages": [response],
#         "past_steps": [step_report]  # 这一步是让主管能“看见”进度的关键
#     }


# ==========================================
# 🏢 4. 部门节点定义 (LangGraph Nodes)
# ==========================================


def supervisor_node(state: dict):
    """主管节点：负责任务路由分发及阶段性总结"""
    llm = get_llm()

    # 1. 熔断机制：连续3次 AI 回复无工具调用，强制拦截
    recent_msgs = state["messages"][-3:]
    if len(recent_msgs) == 3 and all(isinstance(m, AIMessage) and not m.tool_calls for m in recent_msgs):
        backend_log.warning("⚠️ 触发系统熔断，强制结束并汇总。")
        breaker_prompt = SystemMessage(
            content="作为整理专家，请基于对话历史直接汇总已查到的店铺或优惠券信息")
        response = llm.invoke([breaker_prompt] + state["messages"])
        return {"next_agent": "FINISH", "messages": [response]}

    # 2. 历史修剪：控制上下文长度
    trimmed_history = trim_messages(
        state["messages"], max_tokens=5500, strategy="last",
        token_counter=qwen_token_counter, include_system=False
    )

    # 3. 👑 决策调用：组装 Prompt 并执行
    backend_log.info("👑 主管正在进行意图分析与路由决策...")
    prompt_content = SUPERVISOR_PROMPT.content
    if past_steps := state.get("past_steps", []):
        prompt_content += f"\n\n【专家执行日志】:\n" + "\n".join(past_steps)

    supervisor_reminder = SystemMessage(
        content="【警告】：你只是幕后路由主管，绝不允许直接回答用户的问题！即使目标已达成，你也只能输出包含 next_agent 为 'FINISH' 的 JSON 对象！"
    )
    response = llm.invoke([SystemMessage(content=prompt_content)] + trimmed_history + [supervisor_reminder])
    raw_text = str(response.content)
    backend_log.info(f"👑 [主管原始内容]:\n{raw_text[:200]}")


    # 提取日志与指令
    try:
        clean_text = raw_text.replace("<tool_call>", "").replace("</tool_call>", "").strip()
        clean_text = clean_text.replace("```json", "").replace("```", "").strip()
        # 使用我们在外层定义好的解析器强制将大模型输出转为 Python 字典
        parsed_data = supervisor_parser.parse(clean_text)

        # 安全地获取 JSON 中的字段
        thinking = parsed_data.get("thinking", "无推理过程")
        next_agent = parsed_data.get("next_agent", "FINISH")  # 默认安全兜底

        backend_log.info(f"👑 [主管思考]:\n{thinking}")
        backend_log.info(f"👑 主管派单: {next_agent}")

    except OutputParserException as e:
        # 🛡️ 容错防线：如果大模型突然犯傻，没有输出标准 JSON
        backend_log.error(f"❌ [Supervisor] JSON 解析失败: {e}。原始文本: {raw_text}")
        backend_log.warning("触发兜底机制：强制路由至 FINISH 以保护系统不崩溃")
        next_agent = "FINISH"
    except Exception as e:
        # 捕获其他意料之外的错误
        backend_log.error(f"❌ [Supervisor] 发生未知解析错误: {e}")
        next_agent = "FINISH"

    # 5. 🚦 分支流转处理 (保持不变)
    if next_agent == "FINISH":
        backend_log.info("👑 任务阶段结束，生成最终回复...")
        summary_prompt = SystemMessage(content=(
            "作为业务总结专家，请根据对话向用户输出最终答复：\n"
            "1. 简明、有温度地展示已找到的店铺或优惠券结果。\n"
            "2. 若专家此前已提出反问（如'您想看哪家'），请优雅地复述该问题，交出话语权。\n"
            "⚠️ 严禁暴露内部过程或说'无权限'，直接作为贴心助手回答。"
        ))
        # 这里改用 trimmed_history，防止对话轮数过多时引发 Token 超限崩溃
        final_summary = llm.invoke([summary_prompt] + trimmed_history)
        return {"next_agent": "FINISH", "messages": [final_summary]}

    return {"next_agent": next_agent}


def guide_agent_node(state: dict):
    return _run_agent(state, GUIDE_PROMPT, guide_tools, "导购专家")


def transaction_agent_node(state: dict):
    return _run_agent(state, TRANSACTION_PROMPT, transaction_tools, "交易专家")


# ==========================================
# 🔌 5. 工具节点定义
# ==========================================

def guide_tools_node(state: dict, config):
    backend_log.info("🔌 [导购工具] 开始执行...")
    result = ToolNode(guide_tools).invoke(state, config=config)
    return result


def transaction_tools_node(state: dict, config):
    backend_log.info("🔌 [交易工具] 开始执行...")
    result = ToolNode(transaction_tools).invoke(state, config=config)
    return result

# import json
# import re
# import uuid
# from langchain_core.messages import AIMessage, trim_messages
# from langchain_core.tools import tool
# from langchain_core.messages import SystemMessage
# from graph.state import AgentState
# from core.llm import get_llm
# from core.logger import get_logger
# from langchain_core.runnables import RunnableConfig
# from langgraph.prebuilt import ToolNode
#
# from tools.java_api import claim_voucher
#
# backend_log = get_logger("AGENT_BACKEND")
#
# SYSTEM_PROMPT = SystemMessage(content=(
#     "你是“黑马点评”的专属 AI 美食助手，致力于为用户提供精准的餐饮推荐和秒杀抢券服务。\n\n"
#     "<thinking>\n"
#     "1. 意图分析：用户的最终目标是什么？是否包含多个子任务或‘如果...则...’的备选逻辑？\n"
#     "2. 状态复核：我当前在哪一步？之前的工具调用结果（如：库存不足、查询失败）意味着什么？\n"
#     "3. 策略选择：根据当前结果，是否已达成用户目标？若未达成且存在备选方案，我必须立刻调用下一个工具，严禁直接向用户道歉或结束任务。\n"
#     "4. 格式检查：确保输出不含标签，ID 必须为数字。\n"
#     "</thinking>\n\n"
#     "【核心红线（必须绝对遵守）】\n"
#     "1. 绝不捏造：绝对禁止凭藉大模型的预训练记忆来编造餐厅类型、地址、评分或优惠券信息。\n"
#     "2. 禁吐标签：严禁在给用户的最终回复中包含 `<|im_start|>` 或 `<tool_call>` 等标签。\n\n"
#     "【工具使用标准流程 (SOP)】\n"
#     "步骤一（找店）：只要用户的意图涉及寻找餐厅，必须立刻调用 search_shops 工具进行检索。\n"
#     "检索返回的结果可能包含不相关的店铺（例如搜川菜返回了面馆）。你必须严格根据用户的需求（如口味、菜系）在结果中进行二次过滤！只推荐真正符合条件的店铺！\n"
#     "步骤二（查券）：查询某家店的优惠券时，必须从检索结果中提取数字形式的【ID】字段（如 '1'），传给 get_shop_vouchers_tool。严禁传入餐厅名称！\n"
#     "步骤三（抢券）：当用户明确要求抢券时，调用 claim_voucher_tool。\n"
#     "【搜索优先级（绝对禁令）】\n"
#     "1. 严禁盲目调用：如果用户提到一家店，但你当前的对话历史中没有该店的数字 ID，严禁直接调用 get_shop_detail_tool 或 get_shop_vouchers_tool！\n"
#     "2. 强制搜索：只要你手中没有数字 ID，必须先调用 search_shops 工具通过店名获取信息，从搜索结果中提取 ID 后，才能进行下一步操作。\n"
#     "3. 参数校验：工具中的 shop_id 字段必须是纯数字字符串（如 '1'），绝对禁止填入餐厅名称！"
# ))
#
# def qwen_token_counter(messages):
#     """粗略估算 Token 数量"""
#     text = "".join([m.content for m in messages])
#     return int(len(text) / 1.5)
#
#
# # ================= 制造工具 (Tools) =================
#
# @tool
# def search_shops(query: str) -> str:
#     """
#     当用户寻找餐厅、寻找推荐、询问附近有什么好吃的时，使用此工具。
#     根据用户的自然语言查询，从 Redis 向量库中检索相关店铺或探店笔记。
#     """
#     from rag.vectorstore import get_shop_retriever
#     try:
#         retriever = get_shop_retriever(k=15)
#         docs = retriever.invoke(query)
#         if not docs:
#             return "没有找到相关的店铺信息。"
#
#         # 格式化检索结果给 AI 看
#         res = []
#         for i, doc in enumerate(docs, 1):
#             doc_type = doc.metadata.get('type')
#             if doc_type == 'shop':
#                 res.append(
#                     f"[{i}] 类型: 店铺 | 名称: {doc.metadata.get('name')} | ID: {doc.metadata.get('id')}\n    匹配信息: {doc.page_content}")
#             else:
#                 res.append(
#                     f"[{i}] 类型: 笔记 | 标题: {doc.metadata.get('title')} | 关联店铺ID: {doc.metadata.get('shop_id')}\n    匹配信息: {doc.page_content}")
#         return "\n\n".join(res)
#     except Exception as e:
#         return f"检索系统异常: {e}"
#
#
# @tool
# def get_shop_detail_tool(shop_id: str) -> str:
#     """
#     获取餐厅的详细地址、评分、人均消费等深度信息。
#     注意：参数 shop_id 必须通过 search_shops 工具检索获得。
#     禁止直接传入餐厅名称！必须是一个数字形式的字符串。
#     """
#     from tools.java_api import get_shop_detail
#     return get_shop_detail(shop_id)
#
#
# @tool
# def get_shop_vouchers_tool(shop_id: str) -> str:
#     """
#     通过店铺的数据库 ID（通常是数字字符串，如 '1'）查询优惠券。
#     注意：严禁传入餐厅名称（如 '103茶餐厅'），必须传入 search_shops 结果中的 'ID' 字段。
#     """
#     from tools.java_api import get_shop_vouchers
#     return get_shop_vouchers(shop_id)
#
#
# @tool
# def claim_voucher_tool(voucher_id: str, config: RunnableConfig) -> str:
#     """
#     为用户抢购指定 ID 的优惠券。
#     注意：必须传入数字形式的 voucher_id，严禁传入优惠券名称。
#     """
#     # 🕵️ 从 config 的 configurable 中提取前端传来的 authorization
#     # 这样 Token 就实现了透传，而模型完全感知不到它的存在
#     auth_token = config.get("configurable", {}).get("authorization")
#
#     # 调用解耦后的 API 方法
#     return claim_voucher(voucher_id, auth_token)
#
#
# # 将所有工具打包
# tools_list = [search_shops, get_shop_detail_tool, get_shop_vouchers_tool, claim_voucher_tool]
#
#
# # ================= 2. 定义节点 (Nodes) =================
#
# def agent_node(state: AgentState):
#     """
#     大脑节点：负责思考、决策并决定是否调用工具
#     """
#     # A. 消息处理与修剪
#     llm = get_llm().bind_tools(tools_list)
#     trimmed_history = trim_messages(
#         state["messages"],
#         max_tokens=5500,
#         strategy="last",
#         token_counter=qwen_token_counter,
#         include_system=False
#     )
#
#     # B. 调用大模型
#     try:
#         response = llm.invoke([SYSTEM_PROMPT] + trimmed_history)
#         backend_log.info(f"📏 历史消息修剪: {len(state['messages'])} -> {len(trimmed_history)}")
#     except Exception as e:
#         backend_log.error(f"❌ LLM 调用失败: {e}")
#         response = llm.invoke([SYSTEM_PROMPT] + state["messages"][-1:])
#
#     # C. 提取并清理 <thinking> 过程
#     raw_text = response.content if isinstance(response.content, str) else ""
#     thinking_content = ""
#     display_text = raw_text
#
#     # 使用正则分离思考过程与展示内容
#     thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw_text, re.DOTALL)
#     if thinking_match:
#         thinking_content = thinking_match.group(1).strip()
#         backend_log.info(f"🧠 AI 内部思考过程:\n{thinking_content}")
#         # 清除文本中的思考块
#         display_text = re.sub(r'<thinking>.*?</thinking>', '', raw_text, flags=re.DOTALL).strip()
#
#     # D. 工具调用“抢救”机制 (针对本地模型可能不按套路出牌的情况)
#     # 如果没有识别到 native tool_calls 但文本里有标签，进行正则打捞
#     if not response.tool_calls and "<tool_call>" in display_text:
#         response = _salvage_tool_calls(response, display_text)
#         # 更新展示文本，防止工具标签泄露给用户
#         display_text = response.content
#
#     # E. 最终结果封装与日志记录
#     if response.tool_calls:
#         # 情况 1: 需要调用工具
#         backend_log.info(f"🛠️ 拦截到工具调用，前言思考: {display_text[:300]}...")
#         for tc in response.tool_calls:
#             backend_log.info(f"🛠️ 执行工具: {tc['name']} | 参数: {tc['args']}")
#         # 💡 重要：调用工具时，content 必须为空，防止前端显示思考过程
#         response.content = ""
#     else:
#         # 情况 2: 直接回复用户
#         backend_log.info(f"💬 AI 决定直接回复用户")
#         response.content = display_text
#
#     return {"messages": [response]}
#
#
# def _salvage_tool_calls(original_response, text):
#     """正则打捞工具调用逻辑"""
#     backend_log.warning("⚠️ 触发正则抢救机制")
#     match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
#     if not match:
#         return original_response
#
#     try:
#         clean_json = match.group(1).replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
#         tool_data = json.loads(clean_json)
#         backend_log.info(f"✅ 抢救成功: {tool_data.get('name')}")
#
#         return AIMessage(
#             content=text.split("<tool_call>")[0].strip(),  # 保留标签前的文字
#             tool_calls=[{
#                 "name": tool_data.get("name"),
#                 "args": tool_data.get("arguments", {}),
#                 "id": f"call_{uuid.uuid4().hex[:8]}"
#             }]
#         )
#     except Exception as e:
#         backend_log.error(f"❌ 抢救失败: {e}")
#         return original_response
#
#
# # 对于工具节点，我们也想看到原始返回
# def tool_node_with_log(state: AgentState, config: RunnableConfig):  # 💡 增加 config 参数
#     """
#     带日志记录的工具执行节点
#     """
#     t_node = ToolNode(tools_list)
#
#     backend_log.info("🔌 工具节点开始执行...")
#
#     # 💡 关键修改：在调用时必须传入 config 参数
#     # 这样 LangGraph 才能把 thread_id 等信息透传给工具
#     result = t_node.invoke(state, config=config)
#
#     # 记录日志逻辑保持不变
#     for msg in result.get("messages", []):
#         # 截取前 100 字符记录日志
#         backend_log.info(f"📥 工具执行返回: {msg.content[:100]}...")
#
#     return result
#
#
# # 工具节点：使用 LangGraph 官方的高级封装 ToolNode，它会自动解析大脑下达的指令并执行对应的 Python 函数
# tool_node = ToolNode(tools_list)
