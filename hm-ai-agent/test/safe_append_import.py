import json
import pymysql

# 1. 数据库配置（请确认密码和库名是否正确）
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'hmdp',
    'charset': 'utf8mb4'
}


def safe_append_import(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    conn = pymysql.connect(**DB_CONFIG)
    id_map = {}  # 核心映射：{ 临时ID : 数据库真实自增ID }

    try:
        with conn.cursor() as cursor:
            # --- 1. 插入新商铺 ---
            for s in raw_data['shops']:
                temp_id = s.pop('temp_id')  # 暂存并移除临时ID
                if 'id' in s: s.pop('id')  # 确保不传 id，使用数据库自增

                keys = s.keys()
                values = tuple(s.values())
                sql = f"INSERT INTO tb_shop ({', '.join(keys)}) VALUES ({', '.join(['%s'] * len(keys))})"

                cursor.execute(sql, values)
                real_id = cursor.lastrowid  # 拿到数据库刚分配的 ID (比如15, 16...)
                id_map[temp_id] = real_id

            print(f"✅ 成功追加 {len(id_map)} 家商铺。")

            # --- 2. 插入新笔记 ---
            for b in raw_data['blogs']:
                if 'id' in b: b.pop('id')
                temp_shop_id = b['shop_id']

                if temp_shop_id in id_map:
                    # 将笔记的 shop_id 替换为刚才数据库生成的真实 ID
                    b['shop_id'] = id_map[temp_shop_id]

                    keys = b.keys()
                    values = tuple(b.values())
                    sql = f"INSERT INTO tb_blog ({', '.join(keys)}) VALUES ({', '.join(['%s'] * len(keys))})"
                    cursor.execute(sql, values)
                else:
                    print(f"⚠️ 跳过失效笔记：找不到 temp_id {temp_shop_id}")

            conn.commit()
            print(f"🚀 数据安全追加完成！1-14 号旧数据毫发无损。")

    except Exception as e:
        conn.rollback()
        print(f"❌ 错误: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    safe_append_import('data.json')