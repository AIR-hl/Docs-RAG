import os.path
import pickle
from uuid import uuid4

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import VectorParams, SparseVectorParams
from qdrant_client.http.models import Distance, Filter, FieldCondition, MatchValue

from docs_rag.common.config import QDRANT_CONFIG, DATA_PATH


def create_collection(client, embed_dim, collection_name, mode="hybrid"):
    """
    创建Qdrant集合
    :param client: Qdrant客户端实例
    :param embed_dim: 向量维度
    :param collection_name: 集合名称
    :param mode: 模式，可选值为"dense"、"sparse"或"hybrid"
    """
    if mode == "dense":
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embed_dim,
                                        distance=Distance.COSINE),
        )
    elif mode == "sparse":
        client.create_collection(
            collection_name=collection_name,
            sparse_vectors_config={
                "langchain-sparse": SparseVectorParams(),
            },
        )
    elif mode == "hybrid":
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embed_dim,
                                        distance=Distance.COSINE),
            sparse_vectors_config={
                "langchain-sparse": SparseVectorParams(),
            },
        )
    else:
        assert False, "mode error, should be in [dense, sparse, hybrid]"


def get_collections(client):
    """
    获取所有集合名称
    :param client: Qdrant客户端实例
    :return: 集合名称列表
    """
    cols = client.get_collections().collections
    return [c.name for c in cols]


def get_client(mode="memory"):
    """
    获取Qdrant客户端实例
    :param mode: 模式，可选值为"memory"或"remote"
    :return: Qdrant客户端实例
    """
    if mode == "memory":
        client = QdrantClient(":memory:")
    elif mode == "remote":
        client = QdrantClient(
            url=QDRANT_CONFIG["url"],
            api_key=QDRANT_CONFIG["api_key"],
            prefer_grpc=True, )
    else:
        assert False, "mode error, should be in [memory, remote]"
    return client


def get_vectorstore(client, collection_name, embeddings=None, mode="hybrid"):
    """
    根据检索模式创建向量库实例
    :param client: Qdrant客户端实例
    :param collection_name: 集合名称
    :param embeddings: 嵌入模型
    :param mode: 检索模式，可选值为"dense"、"sparse"或"hybrid"
    :return: 向量库实例
    """
    if mode == "hybrid":
        retrieve_mode = RetrievalMode.HYBRID
        sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
    elif mode == "dense":
        retrieve_mode = RetrievalMode.DENSE
        sparse_embedding = None
    elif mode == "sparse":
        embeddings = None
        retrieve_mode = RetrievalMode.SPARSE
        sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
    else:
        assert False, "mode error, should be in [dense, sparse, hybrid]"

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        retrieval_mode=retrieve_mode,
        sparse_embedding=sparse_embedding,
    )
    return vector_store


def add_documents(vectorstore, documents, batch_size):
    """
    在向量库中添加文档chunks
    :param vectorstore: 向量库实例
    :param documents: 文档chunks列表
    :param batch_size: 批处理大小
    """
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'rb') as f:
            col_2_doc = pickle.load(f)
    else:
        col_2_doc = {}
    collection_name = vectorstore.collection_name
    if collection_name in col_2_doc:
        doc_2_ids = col_2_doc[collection_name]
    else:
        doc_2_ids = {}
    with open(DATA_PATH, 'wb') as f:
        # for i in range(0, len(documents), batch_size):
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = [uuid4().hex for _ in range(batch_size)]
            for doc, id in zip(batch, ids):
                doc_name = doc.metadata["source"].split("/")[-1]
                if doc_name not in doc_2_ids:
                    doc_2_ids[doc_name] = [id]
                else:
                    doc_2_ids[doc_name].append(id)
            vectorstore.add_documents(batch, ids=ids)
        col_2_doc[collection_name] = doc_2_ids
        pickle.dump(col_2_doc, f)


def delete_documents(vectorstore, doc_name):
    """
    根据文档名-id映射，从向量库中删除文档名为doc_name的chunks
    :param vectorstore: 向量存储实例
    :param doc_name: 文档名称
    """
    project_root_path = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(project_root_path, "data", "data.pkl")
    with open(data_file_path, 'rb') as f:
        collection_name = vectorstore.collection_name
        col_2_doc = pickle.load(f)
        doc_2_ids = col_2_doc[collection_name]

    with open(data_file_path, 'wb') as f:
        vectorstore.delete(ids=list(doc_2_ids[doc_name].values()))
        del doc_2_ids[doc_name]
        if len(doc_2_ids) == 0:
            del col_2_doc[collection_name]
        else:
            col_2_doc[collection_name] = doc_2_ids
        pickle.dump(col_2_doc, f)


def retrieve_docs(query: str, vectorstore, topk=5, doc_name=None):
    """
    根据query从向量库中检索文档。

    :param query: 查询字符串。
    :param vectorstore: 向量库实例。
    :param topk: 返回的最相关文档chunk数量，默认为5。
    :param doc_name: 文档名称过滤条件，默认为None表示全部文档。
    :return: 匹配的文档chunks列表。元素类型为Document，包含metadata字典和page_content字符串
    """
    if doc_name is None:
        filter_condition = None
    else:
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=f"{doc_name}"),
                )
            ]
        )
    docs = vectorstore.similarity_search(query, k=topk, filter=filter_condition)
    return docs
