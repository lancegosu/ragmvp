import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import openai
import nltk
from pathlib import Path

# Set LLAMA_INDEX_CACHE_DIR environment variable if not already set
llama_index_cache_dir = os.getenv("LLAMA_INDEX_CACHE_DIR", "llama_index_cache")
os.environ["LLAMA_INDEX_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), llama_index_cache_dir)

# Create the cache directory if it doesn't exist
llama_index_cache_path = os.path.join(os.path.dirname(__file__), llama_index_cache_dir)
Path(llama_index_cache_path).mkdir(parents=True, exist_ok=True)

# Set the NLTK data path to the llama_index_cache_path
nltk.data.path.append(llama_index_cache_path)

# Download stopwords if not already downloaded
if not os.path.exists(os.path.join(llama_index_cache_path, "corpora", "stopwords")):
    nltk.download('stopwords', download_dir=llama_index_cache_path)
# from dotenv import load_dotenv
#
# load_dotenv()

st.set_page_config(
    page_title="RAG",
    page_icon="🤖",
)

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.write("# Welcome to RAG! 👋")

# Create data directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)


# File uploader widget
docs = st.file_uploader("Upload a file", type=["pdf", "csv", "txt"], accept_multiple_files=True)

# Handle the uploaded file
if docs is not None:
    for doc in docs:
        # Display the file content (optional)
        file_contents = doc.read()
        # st.write(f"File Content for {doc.name}:")
        # st.write(file_contents)

        # Save each file to the "data" directory
        file_path = os.path.join(data_dir, doc.name)
        with open(file_path, "wb") as f:
            f.write(file_contents)

        st.success(f"File {doc.name} saved successfully at: {file_path}")

# st.sidebar.success("Select a demo above.")
if not os.listdir(data_dir):
    st.warning("Please upload at least one file to interact with the chatbot.")
else:
    st.title("💬 Chat")

    openai.api_key = st.secrets.openai_key
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    chat_engine = index.as_chat_engine()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        # if not openai_api_key:
        #     st.info("Please add your OpenAI API key to continue.")
        #     st.stop()

        # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = chat_engine.chat(prompt)
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
        st.chat_message("assistant").write(str(response))
