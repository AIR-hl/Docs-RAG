from docs_rag.utils.llm_utils import generate
from docs_rag.utils.vectorstore_utils import retrieve_docs
from docs_rag.utils.rag_utils import format_context, build_generate_prompt

def retrieve_n_generate(query, llm, reranker, vectorstore, topk=15, topn=5,doc_name=None, stream=True):
    """
    检索相关文档并生成回答。

    Args:
        query (str): 用户查询。
        llm: 语言模型实例。
        reranker:
        vectorstore: 向量库实例。
        topk: 初步向量库检索的文档数量。
        topn: 重排保留的文档数量
        doc_name (str, optional): 指定文档名称， None为collection下全部文档。
        stream (bool): 流式生成。

    Returns:
    """

    # 检索相关文档
    docs = retrieve_docs(query, vectorstore, topk, doc_name)

    # 重排序文档
    docs = reranker.rerank_documents(docs, query, topn=topn)

    # 格式化上下文
    context = format_context(docs, llm)

    # 构建RAG prompt
    prompt = build_generate_prompt(query, context)

    # 生成回答
    return generate(prompt, llm, stream=stream)