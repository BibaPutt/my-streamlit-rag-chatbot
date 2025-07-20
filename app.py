# =====================================================================
#  FIX FOR SQLITE3 ON STREAMLIT CLOUD
#  This code snippet must be placed at the top of your app's script.
# =====================================================================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# =====================================================================

import streamlit as st
import os
import tempfile
import pandas as pd
from PIL import Image

# LangChain and AI components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

# Other libraries
from chromadb.config import Settings
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- UTILITY FUNCTIONS ---

def format_docs(docs):
    """Prepares the retrieved documents for insertion into the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(chat_history):
    """Formats chat history into a string."""
    return "\n".join(f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history)

# --- STATE MANAGEMENT AND CACHING ---

@st.cache_resource(ttl="2h")
def configure_retriever(uploaded_files):
    """
    Configures the retriever by loading, splitting, and embedding documents.
    """
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    
    img_dir = os.path.join(temp_dir.name, "images")
    os.makedirs(img_dir, exist_ok=True)

    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        
        try:
            if file.name.endswith('.pdf'):
                loader = PyMuPDFLoader(temp_filepath, extract_images=True)
            elif file.name.endswith('.docx'):
                loader = Docx2txtLoader(temp_filepath)
            elif file.name.endswith('.txt'):
                loader = TextLoader(temp_filepath)
            elif file.name.endswith('.html'):
                loader = UnstructuredHTMLLoader(temp_filepath)
            else:
                st.warning(f"Unsupported file type: {file.name}. Skipping.")
                continue
            
            loaded_docs = loader.load()
            
            if file.name.endswith('.pdf'):
                for doc in loaded_docs:
                    if 'image_paths' in doc.metadata:
                        doc.metadata['image_paths'] = [os.path.join(temp_dir.name, p) for p in doc.metadata['image_paths']]

            docs.extend(loaded_docs)
        except Exception as e:
            st.error(f"Error loading file {file.name}: {e}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    doc_chunks = text_splitter.split_documents(docs)

    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key not found in Streamlit secrets. Please add it.")
        st.stop()

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    chroma_settings = Settings(anonymized_telemetry=False)
    vectorstore = Chroma.from_documents(doc_chunks, embeddings_model, client_settings=chroma_settings)

    # OPTIMIZATION: Retrieve slightly fewer documents to reduce memory per request
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# --- MAIN APP ---

st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="🧠")
st.title("🧠 Advanced Multimodal RAG Assistant")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
if not google_api_key:
    st.info("Please add your Google API key to the Streamlit secrets to continue.")
    st.stop()

# --- SIDEBAR AND FILE UPLOAD ---
with st.sidebar:
    st.header("Upload Your Documents")
    uploaded_files = st.file_uploader(
        label="Supports PDF, DOCX, TXT, and HTML files.",
        type=["pdf", "docx", "txt", "html"],
        accept_multiple_files=True
    )
    # OPTIMIZATION: Add a button to clear chat history and free up memory
    if st.button("Clear Conversation History"):
        st.session_state.langchain_messages = []
        st.rerun()


if not uploaded_files:
    st.info("Please upload your documents in the sidebar to start chatting.")
    st.stop()

retriever = configure_retriever(uploaded_files)
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# --- LLM AND PROMPT SETUP ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.2,
    streaming=True,
    google_api_key=google_api_key,
    safety_settings=safety_settings,
)

conversational_qa_template = """You are an expert research assistant. Your goal is to provide clear, accurate, and well-formatted answers based on the provided context, which may include text and images.

**Instructions:**
1.  Analyze the user's question and the provided chat history.
2.  Carefully examine the context, including any text and images.
3.  Synthesize the information to construct a comprehensive answer.
4.  If the context contains relevant data for a table, format your answer as a Markdown table.
5.  Use lists, bolding, and italics to structure your answer for readability.
6.  Conclude your response with a "Key Highlights" section, summarizing the most important points in a bulleted list.
7.  If the context (including images) does not contain the answer, state that clearly. Do not make up information.

**Chat History:**
{chat_history}

**Context:**
{context}

**Question:**
{question}
"""
conversational_qa_prompt = ChatPromptTemplate.from_template(conversational_qa_template)

# --- CHATBOT UI AND LOGIC ---
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello! I'm your advanced RAG assistant. How can I help you analyze your documents?")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if user_prompt := st.chat_input("Ask a question about your documents..."):
    st.chat_message("user").write(user_prompt)

    with st.chat_message("ai"):
        retrieved_docs = retriever.invoke(user_prompt)
        
        context_text = format_docs(retrieved_docs)
        image_sources = []
        
        with st.expander("📚 View Sources"):
            for doc in retrieved_docs:
                source_info = {
                    "source": os.path.basename(doc.metadata.get('source', 'N/A')),
                    "page": doc.metadata.get('page', 'N/A') + 1 if isinstance(doc.metadata.get('page'), int) else 'N/A',
                }
                st.markdown(f"**Source:** `{source_info['source']}` | **Page:** `{source_info['page']}`")
                st.caption(f"Content: *{doc.page_content[:250]}...*")

                if 'image_paths' in doc.metadata:
                    for img_path in doc.metadata['image_paths']:
                        if os.path.exists(img_path):
                            st.image(img_path, width=200)
                            image_sources.append(img_path)

        prompt_content = [
            conversational_qa_prompt.format(
                context=context_text,
                chat_history=format_chat_history(msgs.messages),
                question=user_prompt
            )
        ]
        
        for img_path in image_sources:
            try:
                img = Image.open(img_path)
                prompt_content.append(img)
            except Exception as e:
                st.warning(f"Could not load image {img_path}: {e}")

        multimodal_message = HumanMessage(content=prompt_content)

        try:
            response_stream = llm.stream([multimodal_message])
            full_response = st.write_stream(response_stream)

            msgs.add_user_message(user_prompt)
            msgs.add_ai_message(full_response)
        except Exception as e:
            st.error("An error occurred while generating the response. This might be due to resource limits.")
            st.error(f"Details: {e}")

