# =====================================================================
#  FIX FOR SQLITE3 ON STREAMLIT CLOUD
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
from PIL import Image
import base64
from io import BytesIO
import re
from typing import List, Dict, Any
import time  # Added for rate limiting

# LangChain and AI components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from unstructured.partition.pdf import partition_pdf
from langchain_chroma import Chroma

# Other libraries
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- UTILITY FUNCTIONS ---

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 data URL."""
    buffered = BytesIO()
    img_format = image.format if image.format else 'JPEG'
    image.save(buffered, format=img_format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{img_format.lower()};base64,{img_str}"

def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """Formats chat history into a string for the LLM."""
    formatted_messages = []
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("original_content", msg.get("content", ""))
        formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages)

# --- CORE LOGIC: UPGRADED DATA PIPELINE ---

@st.cache_resource(ttl="2h")
def configure_retriever(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], temp_dir_path: str, api_key: str) -> Chroma | None:
    """
    Configures a robust retriever using `unstructured` for intelligent partitioning and OCR,
    with added batch processing to handle API rate limits.
    """
    if not uploaded_files:
        return None

    with st.status("Processing documents...", expanded=True) as status:
        all_elements = []
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir_path, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            try:
                status.update(label=f"Partitioning {os.path.basename(file.name)} with OCR...")
                elements = partition_pdf(
                    filename=temp_filepath,
                    extract_images_in_pdf=True,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                    max_characters=4000,
                    new_after_n_chars=3800,
                    combine_text_under_n_chars=2000,
                    image_output_dir_path=temp_dir_path,
                )
                all_elements.extend(elements)
            except Exception as e:
                st.error(f"Error processing file '{file.name}': {e}", icon="⚠️")

        if not all_elements:
            status.update(label="Processing failed: No content could be extracted.", state="error")
            return None

        texts_and_tables, images = [], []
        for el in all_elements:
            if 'unstructured.documents.elements.Image' in str(type(el)):
                images.append(el)
            else:
                texts_and_tables.append(el)

        docs_to_embed = []
        for el in texts_and_tables:
            page_num = el.metadata.page_number
            image_paths = [
                img.metadata.image_path for img in images
                if img.metadata.page_number == page_num and img.metadata.image_path and os.path.exists(img.metadata.image_path)
            ]
            docs_to_embed.append({
                "content": str(el),
                "metadata": {
                    "source": os.path.basename(el.metadata.filename),
                    "page": page_num,
                    "image_paths": ";".join(image_paths)
                }
            })
        
        doc_contents = [d['content'] for d in docs_to_embed]
        doc_metadatas = [d['metadata'] for d in docs_to_embed]

        try:
            # UPGRADE: Initialize vector store and add texts in batches to respect rate limits
            embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            vectorstore = Chroma(embedding_function=embedding_function)

            batch_size = 100
            num_batches = len(doc_contents) // batch_size + (1 if len(doc_contents) % batch_size > 0 else 0)

            for i in range(num_batches):
                start_index = i * batch_size
                end_index = start_index + batch_size
                
                status.update(label=f"Embedding batch {i+1}/{num_batches}...")
                vectorstore.add_texts(
                    texts=doc_contents[start_index:end_index],
                    metadatas=doc_metadatas[start_index:end_index]
                )
                
                if i < num_batches - 1:
                    status.update(label=f"Waiting 60s to respect API rate limit...")
                    time.sleep(60)

            status.update(label="Processing complete!", state="complete", expanded=False)
            return vectorstore.as_retriever(search_kwargs={'k': 8})
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            status.update(label="Embedding failed.", state="error")
            return None

# --- MAIN APP (Your code is preserved) ---

st.set_page_config(page_title="Intelligent RAG Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Intelligent RAG Assistant")

if 'temp_dir' not in st.session_state: st.session_state.temp_dir = tempfile.mkdtemp()
if "messages" not in st.session_state: st.session_state.messages = []
if 'conversation_stats' not in st.session_state:
    st.session_state.conversation_stats = {'queries': 0, 'sources': 0, 'images': 0, 'ocr': False}

google_api_key = st.secrets.get("GOOGLE_API_KEY")
if not google_api_key:
    st.info("Please add your Google API key to the Streamlit secrets to continue."); st.stop()

with st.sidebar:
    st.header("📁 Upload Your Documents")
    uploaded_files = st.file_uploader(
        label="Supports PDF files with text, tables, and images.",
        type=["pdf"],
        accept_multiple_files=True
    )
    if st.button("🗑️ Clear Conversation & Files"):
        st.session_state.clear(); st.cache_resource.clear(); st.rerun()
    st.markdown("---")
    st.subheader("💭 Conversation Memory")
    stats = st.session_state.conversation_stats
    col1, col2 = st.columns(2)
    with col1: st.metric("🔍 Queries", stats['queries']); st.metric("📄 Sources", stats['sources'])
    with col2: st.metric("🖼️ Images", stats['images']); st.metric("🔤 OCR", "✅" if stats['ocr'] else "❌")

if not uploaded_files:
    st.info("Please upload PDF documents to begin."); st.stop()

retriever = configure_retriever(uploaded_files, st.session_state.temp_dir, google_api_key)
if not retriever:
    st.warning("Retriever could not be configured. Please check your files."); st.stop()

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest", temperature=0.2, google_api_key=google_api_key, safety_settings=safety_settings
)
conversational_qa_template = """You are an expert research assistant. Your goal is to provide clear, accurate, and well-formatted answers based on the provided context, which may include text and images.

**Instructions:**
1.  Analyze the user's question and the provided chat history.
2.  Carefully examine the context. Synthesize the information to construct a comprehensive answer.
3.  When you reference information that is visually represented in an image, you MUST embed that image in your response using the placeholder format: [IMAGE: <path_to_image>] where <path_to_image> is the full path provided in the context.
4.  Format your answer in the most helpful way for the user. Use Markdown, such as tables, lists, bolding, and italics, as appropriate to enhance readability.
5.  If the context does not contain the answer, state that clearly. Do not make up information.

**Chat History:**
{chat_history}

**Context:**
{context}

**Question:**
{question}
"""
conversational_qa_prompt = ChatPromptTemplate.from_template(conversational_qa_template)

if not st.session_state.messages:
    st.session_state.messages.append({"role": "ai", "content": "Hello! Upload your documents and I'll help you analyze them."})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "rendered_content" in msg:
            for part in msg["rendered_content"]:
                if part["type"] == "text": st.markdown(part["content"])
                elif part["type"] == "image" and os.path.exists(part["path"]): st.image(part["path"])
        else:
            st.markdown(msg["content"])
        
        if msg["role"] == "ai" and "sources" in msg:
            with st.expander("View Sources for this message"):
                for source in msg["sources"]:
                    st.caption(f"📄 {source['source']} - Page {source['page']}")
                    if source.get('image_paths'):
                        for img_path in source['image_paths']:
                            if os.path.exists(img_path): st.image(img_path, width=150)

if user_prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    with st.chat_message("ai"):
        with st.spinner("🔍 Retrieving relevant context..."):
            retrieved_docs = retriever.invoke(user_prompt)
        
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        image_sources_to_pass = []
        source_data_for_display = []

        for doc in retrieved_docs:
            image_paths_str = doc.metadata.get('image_paths', '')
            image_paths_list = image_paths_str.split(';') if image_paths_str else []
            source_data_for_display.append({
                "source": os.path.basename(doc.metadata.get('source', 'N/A')),
                "page": doc.metadata.get('page', 'N/A'),
                "image_paths": image_paths_list
            })
            image_sources_to_pass.extend(image_paths_list)
        
        if not context_text.strip() and not image_sources_to_pass:
            st.error("Could not find any relevant context. Please try another question.", icon="🤷")
        else:
            prompt_content_text = conversational_qa_prompt.format(
                context=context_text, chat_history=format_chat_history(st.session_state.messages), question=user_prompt
            )
            prompt_content = [{"type": "text", "text": prompt_content_text}]
            for img_path in set(image_sources_to_pass):
                try:
                    img = Image.open(img_path)
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    data_url = image_to_base64(img)
                    prompt_content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception: continue

            try:
                with st.spinner("🧠 Generating response..."):
                    response = llm.invoke([HumanMessage(content=prompt_content)])
                full_response = response.content
                image_pattern = r'\[IMAGE:\s*(.*?)\]'
                parts = re.split(image_pattern, full_response)
                
                rendered_content = []
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        image_path = part.strip()
                        if os.path.exists(image_path):
                            st.image(image_path); rendered_content.append({"type": "image", "path": image_path})
                        else: st.warning(f"AI referenced an image that could not be found: {image_path}")
                    elif part.strip():
                        st.markdown(part); rendered_content.append({"type": "text", "content": part})
                
                stats = st.session_state.conversation_stats
                stats['queries'] += 1; stats['sources'] += len(source_data_for_display); stats['images'] += len(image_sources_to_pass)
                stats['ocr'] = True

                st.session_state.messages.append({
                    "role": "ai", "original_content": full_response, "rendered_content": rendered_content, "sources": source_data_for_display
                })
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}", icon="🚨")

