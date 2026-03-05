import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

JAVA_BACKEND_URL = os.getenv("JAVA_BACKEND_URL", "http://127.0.0.1:8081")


def get_shop_detail(shop_id: str) -> str:
    """
    通过店铺ID查询实时详情（如评分、营业时间、具体地址）。
    当向量检索到的信息不够详细时，调用此工具获取最新数据。
    """
    url = f"{JAVA_BACKEND_URL}/shop/{shop_id}"
    try:
        # 黑马点评原生接口通常返回 Result 对象 {success: true, data: {...}}
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        res_json = response.json()

        if res_json.get("success"):
            shop_data = res_json.get("data")
            # 简化数据，只返回 AI 感兴趣的部分，节省 Token
            refined_data = {
                "名称": shop_data.get("name"),
                "评分": shop_data.get("score", 0) / 10.0,
                "地址": shop_data.get("address"),
                "营业时间": shop_data.get("openHours"),
                "人均消费": f"{shop_data.get('avgPrice')}元"
            }
            return json.dumps(refined_data, ensure_ascii=False)
        return f"查询失败：未找到 ID 为 {shop_id} 的店铺。"
    except Exception as e:
        return f"接口调用异常: {str(e)}"


def get_shop_vouchers(shop_id: str) -> str:
    """
    查询指定店铺当前所有可用的优惠券和代金券信息。
    """
    url = f"{JAVA_BACKEND_URL}/voucher/list/{shop_id}"
    try:
        response = requests.get(url, timeout=5)
        res_json = response.json()

        if res_json.get("success"):
            vouchers = res_json.get("data")
            if not vouchers:
                return "该店铺当前没有可用的优惠券。"

            voucher_info = []
            for v in vouchers:
                # 提取真实的数据库 ID
                real_voucher_id = v.get('id')

                # 价格转换 (分 -> 元)
                pay_price = v.get('payValue', 0) / 100.0
                original_price = v.get('actualValue', 0) / 100.0

                # 清洗规则文本中的换行符，防止破坏格式
                rules = v.get('rules', '').replace('\n', ' ').replace('\\n', ' ')
                sub_title = v.get('subTitle', '')

                # 提取库存和类型信息 (对 AI 决策极其重要)
                stock = v.get('stock')
                stock_info = f" | 剩余库存：{stock}张" if stock is not None else ""
                voucher_type = "【秒杀券】" if v.get('type') == 1 else "【普通券】"

                # 💡 核心：把 ID 极其醒目地展示给 AI，并附带库存等决策信息
                info = (f"{voucher_type} {v.get('title')} ({sub_title})\n"
                        f"👉 [必须使用的券ID: {real_voucher_id}]\n"
                        f"   价值：原价{original_price}元，现价{pay_price}元{stock_info}\n"
                        f"   规则：{rules}\n")
                voucher_info.append(info)

            return "\n".join(voucher_info)
        return "暂无优惠信息。"
    except Exception as e:
        return f"获取优惠券失败: {str(e)}"


def claim_voucher(voucher_id: str, auth_token: str = None) -> str:
    """
    底层 Java API 调用逻辑
    抢优惠券的函数
    """
    if not auth_token:
        return "❌ 抢券失败：当前用户未登录，请先登录后再试。"

    url = f"{JAVA_BACKEND_URL}/voucher-order/seckill/{voucher_id}"
    headers = {
        "authorization": auth_token,
        "Content-Type": "application/json"
    }

    try:
        # 增加 timeout 防止请求一直挂起
        response = requests.post(url, headers=headers, timeout=10)

        # 🔥 关键修复：先判断状态码，再尝试解析 JSON
        if response.status_code == 401:
            return "❌ 抢券失败：登录凭证已过期，请重新登录。"

        # 如果返回的不是 JSON（比如报错 HTML），response.json() 会崩溃
        if "application/json" not in response.headers.get("Content-Type", ""):
            return f"❌ 系统响应异常：后端未返回有效的 JSON 数据（状态码：{response.status_code}）"

        res_json = response.json()

        if res_json.get("success"):
            order_id = res_json.get("data")
            return f"✅ 抢券成功！订单号为：{order_id}。请提醒用户尽快去查看。"
        else:
            reason = res_json.get("errorMsg") or "未知错误"
            return f"❌ 抢券失败，原因：{reason}"

    except requests.exceptions.Timeout:
        return "❌ 抢券失败：连接 Java 后端超时，请检查网络。"
    except Exception as e:
        return f"抢券系统异常: {str(e)}"


# ================= 阶段测试 (本地跑) =================
if __name__ == "__main__":
    print("🧪 开始测试 Java 接口连通性...")

    # 测试前请确保：
    # 1. 你的黑马点评 Java 后端已启动 (端口 8081)
    # 2. 数据库里确实有 ID 为 1 的店铺
    test_shop_id = "1"
    #
    # print(f"\n1. 测试查询店铺 [{test_shop_id}] 详情:")
    # print(get_shop_detail(test_shop_id))
    #
    print(f"\n2. 测试查询店铺 [{test_shop_id}] 优惠券:")
    print(get_shop_vouchers(test_shop_id))

    # target_voucher_id = "13"
    #
    # print(f"\n2. [动作阶段] 尝试抢购优惠券 (ID: {target_voucher_id})...")
    # # 注意：在黑马点评项目中，seckill 接口通常需要 User 登录态。
    # # 如果报错“未登录”，请在 claim_voucher 的 headers 中填入你前端登录后的 token
    # claim_result = claim_voucher(target_voucher_id)
    #
    # print(f"--- 抢券结果 ---\n{claim_result}\n----------------")