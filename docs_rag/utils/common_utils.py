import os
import pickle

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter

from docs_rag.common.config import DATA_PATH


def load_n_split(dir_path, file_name="*.md" ):
    """
    加载指定目录下的Markdown文件，并将其拆分为多个文本块。

    Args:
        dir_path (str): 包含Markdown文件的目录路径。
        file_name (str, optional): 文件名的匹配模式，默认为"*.md"。

    Returns:
        list: 包含所有文件拆分后的文本块列表。
    """
    loader = DirectoryLoader(dir_path, loader_cls=TextLoader, glob=file_name)
    documents = loader.load()

    # ========== 文本拆分 ==========
    # headers_to_split_on = [
    #     ("#", "Header 1"),
    #     ("##", "Header 2"),
    #     ("###", "Header 3"),
    # ]
    # markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter()

    chunks_of_all_files = list()
    for doc in documents:
        chunks = markdown_splitter.split_text(doc.page_content)
        for chunk in chunks:
            source=doc.metadata.get("source", "unknown").split("/")[-1]
            chunk.metadata["source"] = source
        chunks_of_all_files.extend(chunks)
    return chunks_of_all_files

def get_docs_name(collection_name: str):
    """
    根据集合名称获取文档名称列表。

    Args:
        collection_name (str): 集合名称。

    Returns:
        list: 文档名称列表。
    """
    # project_root_path = os.path.dirname(os.path.abspath(__file__))
    # data_file_path = os.path.join(project_root_path, "data", "data.pkl")
    with open(DATA_PATH, 'rb') as f:
        col_2_doc=pickle.load(f)
        doc_2_ids=col_2_doc[collection_name]
        docs_name=list(doc_2_ids.keys())
        return docs_name