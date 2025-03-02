# import os

import pandas as pd
import streamlit as st

from docs_rag.common.config import CHAT_LLM_OPTIONS, EMBEDDING_MODEL_CONFIG, API_KEYS_OPTIONS
from docs_rag.rag import retrieve_n_generate
from docs_rag.utils.common_utils import get_docs_name
from docs_rag.utils.llm_utils import get_llm, get_embeddings, generate
from docs_rag.utils.vectorstore_utils import get_vectorstore, get_client, get_collections


def setup_tabs():
    tab1, tab2 = st.tabs(["Q&A", "TXT-doc"])
    return tab1, tab2
from qdrant_client import models

def display_header(tab):
    with tab:
        st.header("Docs RAG - 技术文档助手")
        st.write("请参阅侧边栏以选择文档集合或直接问答")


def setup_doc_selector(client):
    with st.sidebar:
        collection_name = st.selectbox("选择文档集合", get_collections(client), index=None)
        document_name = st.selectbox("选择文档",["ALL"] + get_docs_name("test"), disabled=(collection_name is None))
    return collection_name, document_name


def \
        setup_llm_selector():
    with st.sidebar:
        provider = st.selectbox("选择对话模型提供商", list(CHAT_LLM_OPTIONS.keys()))
        llm = st.selectbox("选择模型型号", CHAT_LLM_OPTIONS[provider])
        return provider, llm

def setup_api_keys():
    with st.sidebar:
        api_keys = st.text_input(label="根据所选 LLM 输入 api-key",
                                 value=API_KEYS_OPTIONS['Infini'],
                                 placeholder="api-key不能为空")
        return api_keys


def setup_rag_param():
    with st.sidebar:
        with st.expander("⚙️ RAG Settings"):
            num_source = st.slider("Top K sources to view:", min_value=1, max_value=10, value=5, step=1)
    return num_source

def display_chat_history(tab, msgs):
    tmp_query = ""
    avatars = {"human": "user", "ai": "assistant"}
    with tab:
        for msg in msgs.messages:
            if msg.content.startswith("Query:"):
                tmp_query = msg.content.lstrip("Query: ")
            elif msg.content.startswith("# Retrieval"):
                with st.expander(
                        f"📖 **Context Retrieval:** {tmp_query}", expanded=False
                ):
                    st.write(msg.content, unsafe_allow_html=True)
            else:
                tmp_query = ""
                st.chat_message(avatars[msg.type]).write(msg.content)
# def display_markdown(tab, document_name, collection_name):
#     if document_name == "All":
#         with tab:
#             st.write(
#                 "选择文档时后会加载并展示Markdown文件。显示的文本，公式，表，图表和图像可能会包含一些渲染错误"
#             )
#     else:
#         md_file_path = md_path_creator(md_path, collection_name, document_name)
#         with tab:
#             md_container = st.container(height=700)
#             with md_container:
#                 md_txt = md_loader(md_file_path)
#                 st.write(md_txt, unsafe_allow_html=True)
def stream_response(response):
    chat_placeholder = st.empty()  # 创建一个占位符用于显示聊天内容
    full_message = ""
    for chunk in response:
        if hasattr(chunk, 'content') and isinstance(chunk.content, str):  # 确保chunk.content存在且是字符串
            full_message += chunk.content
            chat_placeholder.markdown(full_message, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": full_message})


def clear_chat_history(msgs):
    msgs.clear()
    msgs.add_ai_message("欢迎使用Docs RAG文档问答助手")


def convert_df(msgs):
    df = []
    for msg in msgs.messages:
        df.append({"type": msg.type, "content": msg.content})

    df = pd.DataFrame(df)
    return df.to_csv().encode("utf-8")


if __name__ == "__main__":
    st.set_page_config(page_title="Docs RAG", page_icon="📄", layout="wide") # 设置页面配置

    if "messages" not in st.session_state:
        st.session_state.messages = []

    provider, llm_name = setup_llm_selector()
    api_key = setup_api_keys()

    llm = get_llm(model_name=llm_name, api_key=api_key, provider=provider)

    client = get_client("remote")
    embeddings = get_embeddings(EMBEDDING_MODEL_CONFIG["api_key"],
                                EMBEDDING_MODEL_CONFIG["model"],
                                EMBEDDING_MODEL_CONFIG["provider"])
    tab1, tab2 = setup_tabs()
    display_header(tab1)

    user_query = st.chat_input(placeholder="What is your question on the selected collection/document?")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

    collection_name, document_name = setup_doc_selector(client)
    if not collection_name and user_query:
        response = generate(user_query, llm, stream=True)
        stream_response(response)
    elif collection_name and user_query:
        vectorstore=get_vectorstore(client, collection_name, embeddings, "hybrid")
        num_source = setup_rag_param()
        if document_name.lower()=="all":
            response=retrieve_n_generate(user_query, llm, vectorstore, num_source, stream=True)
        else:
            response=retrieve_n_generate(user_query, llm, vectorstore, num_source, doc_name=document_name, stream=True)
        stream_response(response)
