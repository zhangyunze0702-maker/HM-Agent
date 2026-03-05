import logging
import os

# 确保日志目录存在
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志格式：时间 - 节点名 - 消息
formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')

# 创建文件处理器，将后台逻辑写入 backend.log
file_handler = logging.FileHandler(
    os.path.join(log_dir, "backend.log"),
    mode='a',
    encoding='utf-8'
)
file_handler.setFormatter(formatter)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 切断与 Uvicorn 默认日志系统的联系，防止日志被框架“吃掉”
    logger.propagate = False
    # 关键：不要让日志往控制台流（StreamHandler），只往文件流
    if not logger.handlers:
        logger.addHandler(file_handler)
    return logger