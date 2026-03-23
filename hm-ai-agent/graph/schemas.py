from pydantic import BaseModel, Field

# 主管的路由输出规范
class SupervisorOutput(BaseModel):
    thinking: str = Field(description="对当前进度的思考和推理逻辑")
    next_agent: str = Field(description="下一个执行的节点，严格从以下选取：['guide', 'transaction', 'FINISH']")

# 专家的通用输出规范 (导购和交易专家共用)
class AgentOutput(BaseModel):
    thinking: str = Field(description="内部推理逻辑，说明你为什么这么做")
    reply_to_user: str = Field(description="最终展示给用户的回复话术。如果使用了工具，这里可以留空")
    extracted_shop_id: str = Field(default="", description="如果你在上下文中识别或检索到了具体的店铺纯数字ID，请填入这里；若无则留空")
