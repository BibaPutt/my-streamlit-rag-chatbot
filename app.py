# =====================================================================
#  FIX FOR SQLITE3 ON STREAMLIT CLOUD
#  This code snippet must be placed at the top of your app's script.
# =====================================================================
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# =====================================================================

import streamlit as st
import os
import tempfile
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
from langchain_core.documents import Document
import base64
from io import BytesIO
import re

# LangChain and AI components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import (
    Docx2txtLoader, TextLoader, UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader, CSVLoader
)
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

def _get_doc_metadata(doc, key, default=None):
    """Safely retrieves a key from a document's metadata."""
    return doc.metadata.get(key, default)

# Custom function to handle PDF loading with robust image extraction
def load_pdf_with_images(file_path, temp_dir_path):
    """
    Manually loads a PDF, extracts text and images, and associates them.
    """
    documents = []
    image_save_dir = os.path.join(temp_dir_path, "images", os.path.splitext(os.path.basename(file_path))[0])
    os.makedirs(image_save_dir, exist_ok=True)
    
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        image_paths = []

        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(image_save_dir, image_filename)
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_paths.append(image_path)
            
        image_paths_str = ";".join(image_paths)
        
        documents.append(Document(
            page_content=text,
            metadata={
                'source': os.path.basename(file_path),
                'page': page_num,
                'image_paths': image_paths_str
            }
        ))
    return documents

# --- STATE MANAGEMENT AND CACHING ---

@st.cache_resource(ttl="2h")
def configure_retriever(uploaded_files, temp_dir_path):
    """
    Configures the retriever by loading, splitting, and embedding documents.
    """
    docs = []
    
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir_path, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        
        try:
            file_extension = os.path.splitext(file.name)[1].lower()
            if file_extension == '.pdf':
                loaded_docs = load_pdf_with_images(temp_filepath, temp_dir_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(temp_filepath)
                loaded_docs = loader.load()
            elif file_extension == '.txt':
                loader = TextLoader(temp_filepath)
                loaded_docs = loader.load()
            elif file_extension == '.html':
                loader = UnstructuredHTMLLoader(temp_filepath)
                loaded_docs = loader.load()
            elif file_extension == '.pptx':
                loader = UnstructuredPowerPointLoader(temp_filepath)
                loaded_docs = loader.load()
            elif file_extension == '.csv':
                loader = CSVLoader(temp_filepath)
                loaded_docs = loader.load()
            else:
                st.warning(f"Unsupported file type: {file.name}. Skipping.")
                continue
            
            docs.extend(loaded_docs)
        except Exception as e:
            st.error(f"Error loading file '{file.name}': {e}", icon="⚠️")
            continue

    if not docs:
        st.warning("No documents were successfully loaded. Please upload supported files.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=250)
    doc_chunks = text_splitter.split_documents(docs)

    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key not found in Streamlit secrets. Please add it.")
        st.stop()

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    chroma_settings = Settings(anonymized_telemetry=False)
    vectorstore = Chroma.from_documents(doc_chunks, embeddings_model, client_settings=chroma_settings)

    return vectorstore.as_retriever(search_kwargs={"k": 10})

# --- MAIN APP ---

st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="🧠")
st.title("🧠 Advanced Multimodal RAG Assistant")

if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

google_api_key = st.secrets.get("GOOGLE_API_KEY")
if not google_api_key:
    st.info("Please add your Google API key to the Streamlit secrets to continue.")
    st.stop()

# --- SIDEBAR AND FILE UPLOAD ---
with st.sidebar:
    st.header("Upload Your Documents")
    uploaded_files = st.file_uploader(
        label="Supports PDF, DOCX, TXT, HTML, PPTX, CSV",
        type=["pdf", "docx", "txt", "html", "pptx", "csv"],
        accept_multiple_files=True
    )
    if st.button("Clear Conversation & Files"):
        st.session_state.langchain_messages = []
        st.cache_resource.clear()
        st.rerun()
    st.markdown("---")
    st.info("Image extraction is best supported for PDF files.", icon="ℹ️")
    st.caption("Note: This app uses a free API tier. If you encounter errors, please wait a minute before trying again.")


if not uploaded_files:
    st.info("Please upload your documents in the sidebar to start chatting.")
    st.stop()

retriever = configure_retriever(uploaded_files, st.session_state.temp_dir)
if not retriever:
    st.stop()

msgs = StreamlitChatMessageHistory(key="langchain_messages")

# --- LLM AND PROMPT SETUP ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Use a non-streaming model for this approach
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.2,
    google_api_key=google_api_key,
    safety_settings=safety_settings,
)

# KEY CHANGE: Updated prompt with instructions for inline image citation
conversational_qa_template = """You are an expert research assistant. Your goal is to provide clear, accurate, and well-formatted answers based on the provided context, which may include text and images.

**Instructions:**
1.  Analyze the user's question and the provided chat history.
2.  Carefully examine the context, including any text and images. The context provided under 'Context:' is the most relevant information.
3.  Synthesize the information to construct a comprehensive answer.
4.  When you reference information from an image, you MUST embed it in your response using the placeholder format: [IMAGE: <path_to_image>] where <path_to_image> is the full path provided in the context.
5.  If the context contains relevant data for a table, format your answer as a Markdown table.
6.  Use lists, bolding, and italics to structure your answer for readability.
7.  Conclude your response with a "Key Highlights" section, summarizing the most important points in a bulleted list.
8.  If the context (including images) does not contain the answer, state that clearly. Do not make up information.

**Chat History:**
{chat_history}

**Context:**
{context}

**Available Image Paths:**
{image_paths}

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
        image_sources_to_pass = []
        all_image_paths = []
        
        with st.expander("📚 View Sources"):
            for doc in retrieved_docs:
                source_info = {
                    "source": os.path.basename(_get_doc_metadata(doc, 'source', 'N/A')),
                    "page": _get_doc_metadata(doc, 'page', 0) + 1,
                }
                st.markdown(f"**Source:** `{source_info['source']}` | **Page:** `{source_info['page']}`")
                st.caption(f"Content: *{doc.page_content[:250]}...*")

                image_paths_str = _get_doc_metadata(doc, 'image_paths', '')
                if image_paths_str:
                    image_paths_list = image_paths_str.split(';')
                    for img_path in image_paths_list:
                        if os.path.exists(img_path):
                            st.image(img_path, width=200)
                            image_sources_to_pass.append(img_path)
                            all_image_paths.append(img_path)

        prompt_content_text = conversational_qa_prompt.format(
            context=context_text,
            chat_history=format_chat_history(msgs.messages),
            image_paths="\n".join(all_image_paths),
            question=user_prompt
        )
        
        if not context_text.strip() and not image_sources_to_pass:
            st.error("Could not extract any relevant content from the documents. Please try a different question.", icon="🤷")
        else:
            prompt_content = [{"type": "text", "text": prompt_content_text}]
            for img_path in image_sources_to_pass:
                try:
                    img = Image.open(img_path)
                    buffered = BytesIO()
                    img_format = img.format if img.format else 'JPEG'
                    img.save(buffered, format=img_format)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    data_url = f"data:image/{img_format.lower()};base64,{img_str}"
                    prompt_content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception as e:
                    st.warning(f"Could not process image {img_path}: {e}")

            multimodal_message = HumanMessage(content=prompt_content)

            try:
                # KEY CHANGE: Invoke the model to get the full response at once
                response = llm.invoke([multimodal_message])
                full_response = response.content

                # KEY CHANGE: Parse the response and render text and images
                # Regex to find all [IMAGE: ...] placeholders
                image_pattern = r'\[IMAGE:\s*(.*?)\]'
                parts = re.split(image_pattern, full_response)

                for i, part in enumerate(parts):
                    if i % 2 == 1:  # This is an image path
                        image_path = part.strip()
                        if os.path.exists(image_path):
                            st.image(image_path)
                        else:
                            st.warning(f"AI referenced an image that could not be found: {image_path}")
                    else:  # This is a text part
                        if part.strip():
                            st.markdown(part)

                msgs.add_user_message(user_prompt)
                msgs.add_ai_message(full_response)
            except Exception as e:
                st.error("An error occurred while generating the response. This might be due to API rate limits or other issues.", icon="🚨")
                st.error(f"Details: {e}")
