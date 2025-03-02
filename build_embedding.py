import os.path

from docs_rag.common.config import EMBEDDING_MODEL_CONFIG
from docs_rag.utils.llm_utils import get_embeddings
from docs_rag.utils.common_utils import load_n_split
from docs_rag.utils.vectorstore_utils import create_collection, get_vectorstore, get_client, add_documents

if __name__ == "__main__":
    collection_name="test"
    knowledge_lib_path="../knowledge_lib"

    # 获取潜入模型
    embeddings=get_embeddings(EMBEDDING_MODEL_CONFIG["api_key"],
                              EMBEDDING_MODEL_CONFIG["model"],
                              EMBEDDING_MODEL_CONFIG["provider"],)

    # 向量库代理
    client=get_client("remote")

    # 创建向量库中集合
    collection=create_collection(client, embed_dim=1024, collection_name=collection_name)

    # 获取向量库
    vectorstore=get_vectorstore(client, collection_name, embeddings)

    # 加载文档
    documents=load_n_split(os.path.join(knowledge_lib_path, collection_name))

    # 添加文档
    add_documents(vectorstore, documents, batch_size=10)