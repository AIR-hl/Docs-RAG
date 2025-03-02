import os
from typing import List

import requests
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from docs_rag.common.config import INFINI_BASE_URL


class Reranker:
    def __init__(self, api_key: str, model: str = "bge-reranker-v2-m3"):
        self.api_key = api_key
        self.model = model
    def rerank_documents(self, docs: List[Document], query: str, topn: int = 10) -> List[Document]:
        """
        使用reranker模型对检索到的文档进行重排序。

        Args:
            docs (List[Document]): 初始检索的文档列表。
            query (str): 用户查询。
            topn (int): 保留的最相关文档数量，默认为10。

        Returns:
            List[Document]: 重排序后的文档列表。
        """
        if not docs: return []

        documents = [doc.page_content for doc in docs]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": True
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(os.path.join(INFINI_BASE_URL, "rerank"), json=payload, headers=headers)
            response.raise_for_status()  # 检查请求是否成功
            result = response.json()

            # 根据相关性分数重新排序文档
            reranked_docs = []
            for item in result.get("results", [])[:topn]:
                index = item.get("index")
                if 0 <= index < len(docs):
                    reranked_docs.append(docs[index])

            return reranked_docs
        except Exception as e:
            print(f"Reranking failed: {e}")
            return docs[:min(topn, len(docs))]


def format_context(docs: List[Document], llm) -> str:
    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content

        context += f"**文档 {i + 1}** (来源: {source}):\n\n{content}\n\n---\n\n"
    return context


def build_generate_prompt(query: str, context: str):
    system_message_content = (
        "你是一个专业的技术文档助手。你的任务是回答用户的问题。"
        "请你根据自身知识和用户提供的**参考信息**来回答问题。"
        "如果你不知道答案，请如实说你不知道，不要编造答案。"
    )

    final_query = (
        "\n\n---\n# 参考信息:\n"
        f"{context}\n"
        f"# 用户问题:\n{query}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message_content),
        ("user", "{query_text}"),
    ])
    prompt = prompt_template.invoke({"query_text": final_query})

    return prompt
