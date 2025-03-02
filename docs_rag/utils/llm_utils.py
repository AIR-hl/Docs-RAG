import os

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client.http import models
from docs_rag.common import config
from docs_rag.common.infini_embedding import InfiniEmbeddings


def get_embeddings(api_key, model_name, provider="infini"):
    """
    根据提供的API密钥、模型名称和供应商获取嵌入模型。

    Args:
        api_key (str): API密钥。
        model_name (str): 模型名称。
        provider (str, optional): 供应商名称，默认为"infini"。

    Returns:
        embeddings: 嵌入模型实例。
    """
    os.environ["OPENAI_API_KEY"] = api_key
    if provider.lower() == "openai":
        os.environ["OPENAI_BASE_URL"] = config.OPENAI_BASE_URL
        embeddings = OpenAIEmbeddings(model=model_name)
    elif provider.lower() == "infini":
        os.environ["OPENAI_BASE_URL"] = config.INFINI_BASE_URL
        embeddings = InfiniEmbeddings(model=model_name)
    elif provider.lower() == "aliyun":
        os.environ["OPENAI_BASE_URL"] = config.QWEN_BASE_URL
        embeddings = DashScopeEmbeddings(model=model_name, dashscope_api_key=api_key)
    else:
        assert False, "provider not support"
    return embeddings

def get_llm(model_name, api_key, provider="infini", temperature=0.6, stream=True):
    """
    根据提供的模型名称、API密钥、供应商、温度和流式设置获取语言模型。

    Args:
        model_name (str): 模型名称。
        api_key (str): API密钥。
        provider (str, optional): 供应商名称，默认为"infini"。
        temperature (float, optional): 温度参数，控制生成文本的随机性，默认为0.6。
        stream (bool, optional): 是否使用流式生成，默认为True。

    Returns:
        llm: 语言模型实例。
    """
    if provider.lower()=="infini":
        base_url=config.INFINI_BASE_URL
    elif provider.lower()=="aliyun":
        base_url=config.QWEN_BASE_URL
    elif provider.lower()=='zhipu':
        base_url=config.GLM_BASE_URL
    elif provider.lower()=='openai':
        base_url=config.OPENAI_BASE_URL
    else:
        assert False, "provider not support"
    llm = ChatOpenAI(model_name=model_name,
                     streaming=stream,
                     temperature=temperature,
                     openai_api_base=base_url,
                     openai_api_key=api_key)
    return llm

def generate(query, llm, stream=True):
    """
    根据查询和语言模型生成响应。

    Args:
        query (str): 查询文本。
        llm: 语言模型实例。
        stream (bool, optional): 是否使用流式生成，默认为True。

    Returns:
        生成器或直接响应结果。
    """
    if stream:
        return _generate_stream(query, llm)
    else:
        return _generate_directly(query, llm)


def _generate_stream(query, llm):
    """
    使用流式生成响应。
    """
    for chunk in llm.stream(query):
        yield chunk

def _generate_directly(query, llm):
    """
    直接生成响应。
    """
    return llm.invoke(query)