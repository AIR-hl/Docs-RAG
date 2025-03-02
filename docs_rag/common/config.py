import os


QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
INFINI_BASE_URL = "https://cloud.infini-ai.com/maas/v1/"
OPENAI_BASE_URL = "https://api.openai.com/v1/"


CHAT_LLM_OPTIONS = {"Infini": ["deepseek-v3", "deepseek-r1"],
                    "Aliyun": ["qwen-max", "qwen-max-2025-01-25", "qwen-plus"],
                    "Zhipu": ["glm-4-flash"]}

API_KEYS_OPTIONS = {
    "Infini": "xxxxxxxxxx",
    "Aliyun": "xxxxxxxxxx",
    "Zhipu": "xxxxxxxxxx"
}

EMBEDDING_MODEL_CONFIG = {"model": "bge-m3",
                          "api_key": "sk-dasveas7kymv4vak",
                          "provider": "Infini"}

QDRANT_CONFIG = {"url": "xxxxxxxxxx",
                 "api_key": "xxxxxxxxxx"}

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../","data/data.pkl")