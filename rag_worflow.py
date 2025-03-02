from langchain_core.prompts import ChatPromptTemplate

from docs_rag.common.config import API_KEYS_OPTIONS, EMBEDDING_MODEL_CONFIG
from docs_rag.rag import retrieve_n_generate
from docs_rag.utils.llm_utils import get_llm, get_embeddings, generate
from docs_rag.utils.rag_utils import Reranker
from docs_rag.utils.vectorstore_utils import get_vectorstore, get_client, retrieve_docs, retrieve_docs

if __name__ == "__main__":
    # 初始化对话 LLM
    llm = get_llm(model_name="deepseek-v3", api_key=API_KEYS_OPTIONS['Infini'])
    # 初始化reranker
    reranker = Reranker(api_key=API_KEYS_OPTIONS['Infini'], model="./models/reranker-v2-m3")

    # 获取向量库
    client = get_client(mode='remote')
    # 获取嵌入模型
    embeddings = get_embeddings(EMBEDDING_MODEL_CONFIG["api_key"], EMBEDDING_MODEL_CONFIG['model'])
    # 获取向量库实例
    vectorstore = get_vectorstore(
        client=client,
        collection_name="test",
        embeddings=embeddings,
        mode="hybrid"
    )

    query="How to use `trl` to run a training of DPO?"
    # 直接生成
    response=generate(query, llm)
    for chunk in response:
        if hasattr(chunk, 'content') and isinstance(chunk.content, str):  # 确保chunk.content存在且是字符串
            print(chunk.content, end="", flush=True)
    print("===========================================")

    # 检索
    docs=retrieve_docs(query, vectorstore, topk=5)

    # 生成
    # context="\n".join([doc.page_content for doc in docs])
    # system_message_content = (
    #     "You are an assistant for question-answering tasks. "
    #     "You can use the reference context provided to answer the question."
    #     "If you don't know the answer, say that you don't know."
    #     "\n\n---\nReference Context:**"
    #     "{context}"
    #     "\n\n---\n"
    # )
    # prompt_template = ChatPromptTemplate.from_messages([
    #     ("system", system_message_content),
    #     ("user", "{query}"),
    # ])
    # prompt=prompt_template.invoke({"context":context, "query":"你好"})
    # response=llm.invoke(prompt)

    response=retrieve_n_generate(query, llm, reranker, vectorstore, topk=5, topn=2)
    for chunk in response:
        if hasattr(chunk, 'content') and isinstance(chunk.content, str):  # 确保chunk.content存在且是字符串
            print(chunk.content, end="", flush=True)
