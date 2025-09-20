# =====================================================================
# FIX FOR SQLITE3 ON STREAMLIT CLOUD
# This code snippet must be placed at the top of your app's script.
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
import time
import random

# LangChain and AI components
from langchain_google_genai import ChatGoogleGenerativeAI
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
from langchain_community.embeddings import HuggingFaceEmbeddings

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

# OCR function for image-heavy PDFs
def simple_ocr_extraction(image_path):
    """Simple OCR extraction with error handling."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        # If OCR fails, return empty string
        return ""

# Enhanced PDF loading with OCR fallback
def load_pdf_with_images(file_path, temp_dir_path):
    """
    Manually loads a PDF, extracts text and images, with OCR fallback for image-heavy PDFs.
    """
    documents = []
    image_save_dir = os.path.join(temp_dir_path, "images", os.path.splitext(os.path.basename(file_path))[0])
    os.makedirs(image_save_dir, exist_ok=True)
    
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        image_paths = []
        ocr_texts = []

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
            
            # If no text found, try OCR on images
            if not text.strip():
                ocr_text = simple_ocr_extraction(image_path)
                if ocr_text:
                    ocr_texts.append(f"Image {img_index+1} text: {ocr_text}")
        
        # Combine extracted text with OCR text
        combined_text = text
        if ocr_texts:
            combined_text += "\n\n" + "\n".join(ocr_texts)
        
        # If still no content, add a placeholder
        if not combined_text.strip():
            combined_text = f"[Page {page_num+1} contains images - visual content available]"
            
        image_paths_str = ";".join(image_paths)
        
        documents.append(Document(
            page_content=combined_text,
            metadata={
                'source': os.path.basename(file_path),
                'page': page_num,
                'image_paths': image_paths_str,
                'has_ocr': bool(ocr_texts)
            }
        ))
    return documents

# Proactive rate limiting helper
def check_rate_limit():
    """Simple rate limiting to avoid API errors by enforcing a delay."""
    if 'last_request_time' not in st.session_state:
        st.session_state.last_request_time = 0
    
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_request_time

    # Wait at least 2 seconds between requests
    if time_since_last < 2:
        time.sleep(2 - time_since_last)

    st.session_state.last_request_time = time.time()


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

    # Filter out empty chunks that could cause embedding errors
    doc_chunks = [chunk for chunk in doc_chunks if chunk.page_content.strip()]

    if not doc_chunks:
        st.warning("No text content could be extracted. Please check if files contain readable text.")
        return None

    # Use Hugging Face embeddings (free and local)
    try:
        # Try multiple embedding models for better compatibility
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",  # Fast and efficient
            "sentence-transformers/all-mpnet-base-v2",  # Higher quality
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual support
        ]
        
        embeddings_model = None
        for model_name in embedding_models:
            try:
                st.info(f"🔄 Loading embedding model: {model_name}")
                embeddings_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},  # Use CPU to avoid GPU memory issues
                    encode_kwargs={'normalize_embeddings': True}
                )
                st.success(f"✅ Successfully loaded: {model_name}")
                break
            except Exception as e:
                st.warning(f"⚠️ Failed to load {model_name}: {e}")
                continue
        
        if not embeddings_model:
            st.error("❌ Could not load any embedding model. Please check your internet connection.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        st.stop()
    
    chroma_settings = Settings(anonymized_telemetry=False)

    try:
        vectorstore = Chroma.from_documents(doc_chunks, embeddings_model, client_settings=chroma_settings)
        return vectorstore.as_retriever(search_kwargs={"k": 8})
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.error("This might be due to empty content or embedding issues. Please try different documents.")
        return None

# --- MAIN APP ---
st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Advanced Multimodal RAG Assistant")

# Initialize session state variables
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'all_sources' not in st.session_state:
    st.session_state.all_sources = []
if 'all_images' not in st.session_state:
    st.session_state.all_images = []
if 'conversation_stats' not in st.session_state:
    st.session_state.conversation_stats = {
        'total_queries': 0,
        'total_sources': 0,
        'total_images': 0,
        'has_ocr': False
    }

# Check for Google API key (only needed for LLM, not embeddings)
google_api_key = st.secrets.get("GOOGLE_API_KEY")
if not google_api_key:
    st.warning("⚠️ Google API key not found. You can still use the app for document analysis, but AI responses will be limited.")
    st.info("💡 To enable full AI responses, add your Google API key to Streamlit secrets.")
    # Don't stop the app, just show warning

# --- LLM AND PROMPT SETUP ---
llm = None
if google_api_key:
    try:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.2,
            google_api_key=google_api_key,
            safety_settings=safety_settings,
        )
        st.success("✅ AI model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load AI model: {e}")
        llm = None
else:
    st.info("ℹ️ AI responses disabled - only document search available")

# --- SIDEBAR AND FILE UPLOAD ---
with st.sidebar:
    st.header("📁 Upload Your Documents")
    uploaded_files = st.file_uploader(
        label="Supports PDF, DOCX, TXT, HTML, PPTX, CSV",
        type=["pdf", "docx", "txt", "html", "pptx", "csv"],
        accept_multiple_files=True
    )
    
    if st.button("🗑️ Clear Conversation & Files"):
        st.session_state.langchain_messages = []
        st.session_state.all_sources = []
        st.session_state.all_images = []
        st.session_state.conversation_stats = {
            'total_queries': 0,
            'total_sources': 0,
            'total_images': 0,
            'has_ocr': False
        }
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    # Conversation Memory Panel
    st.subheader("💭 Conversation Memory")
    stats = st.session_state.conversation_stats

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔍 Queries", stats['total_queries'])
        st.metric("📄 Sources", stats['total_sources'])
    with col2:
        st.metric("🖼️ Images", stats['total_images'])
        st.metric("🔤 OCR", "✅" if stats['has_ocr'] else "❌")

    # Show accumulated sources
    if st.session_state.all_sources:
        with st.expander(f"📚 All Sources ({len(st.session_state.all_sources)})"):
            unique_sources = {}
            for source in st.session_state.all_sources:
                key = f"{source['source']}_p{source['page']}"
                if key not in unique_sources:
                    unique_sources[key] = source
            
            for source in unique_sources.values():
                st.caption(f"📄 {source['source']} - Page {source['page']}")

    # Show accumulated images
    if st.session_state.all_images:
        with st.expander(f"🖼️ All Images ({len(st.session_state.all_images)})"):
            num_images = len(st.session_state.all_images)
            cols_per_row = 2
            
            for i in range(0, num_images, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < num_images:
                        img_path = st.session_state.all_images[i + j]
                        if os.path.exists(img_path):
                            with col:
                                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

    st.markdown("---")
    st.info("💡 Images are embedded directly in responses where relevant", icon="ℹ️")
    st.caption("⚡ Using local Hugging Face embeddings (no API limits!)")
    if llm:
        st.caption("🤖 AI responses enabled with Google Gemini")
    else:
        st.caption("📄 Document search only (add Google API key for AI responses)")

if not uploaded_files:
    st.info("Please upload your documents in the sidebar to start chatting.")
    st.stop()

retriever = configure_retriever(uploaded_files, st.session_state.temp_dir)
if not retriever:
    st.stop()

msgs = StreamlitChatMessageHistory(key="langchain_messages")

conversational_qa_template = """You are an expert research assistant. Your goal is to provide clear, accurate, and well-formatted answers based on the provided context, which may include text and images.
Instructions:
- Analyze the user's question and the provided chat history.
- Carefully examine the context, including any text and images. The context provided under 'Context:' is the most relevant information.
- Synthesize the information to construct a comprehensive answer.
- SMART IMAGE EMBEDDING: When you reference information from an image, embed it directly in your response using: [IMAGE: <path_to_image>]
- Only embed images that are directly relevant to your answer (typically 1-3 max)
- Place images naturally in the flow of your text where they add most value
- Briefly explain what the image shows and why it's relevant
- Structure your response with appropriate markdown formatting (headers, bold, lists) for readability.
- If relevant data exists, present it as a clear Markdown table.
- End with a brief "Key Points" section summarizing the most important information.
- If the context doesn't contain the answer, state that clearly and don't make up information.

Chat History:
{chat_history}

Context:
{context}

Available Image Paths:
{image_paths}

Question:
{question}

Provide a comprehensive, well-structured response that directly addresses the user's question."""
conversational_qa_prompt = ChatPromptTemplate.from_template(conversational_qa_template)

# --- MAIN CHAT INTERFACE ---
if len(msgs.messages) == 0:
    if llm:
        msgs.add_ai_message("Hello! I'm your advanced RAG assistant with AI-powered responses. Upload documents and I'll analyze them. 🚀")
    else:
        msgs.add_ai_message("Hello! I'm your document search assistant. Upload documents and I'll help you find relevant content. (AI responses disabled - add Google API key for full features) 📄")

for msg in msgs.messages:
    with st.chat_message(msg.type):
        if msg.type == "ai" and "[IMAGE:" in msg.content:
            image_pattern = r'\[IMAGE:\s*(.*?)\]'
            parts = re.split(image_pattern, msg.content)
            
            for i, part in enumerate(parts):
                if i % 2 == 1:  # This is an image path
                    image_path = part.strip()
                    found_image = None
                    if image_path in st.session_state.all_images and os.path.exists(image_path):
                        found_image = image_path
                    else:
                        for stored_img in st.session_state.all_images:
                            if os.path.basename(stored_img) == os.path.basename(image_path) and os.path.exists(stored_img):
                                found_image = stored_img
                                break
                    if found_image:
                        st.image(found_image, width=500, caption=f"📷 {os.path.basename(found_image)}")
                    else:
                        st.warning(f"Referenced image not found: {os.path.basename(image_path)}")
                else:  # This is text
                    if part.strip():
                        st.markdown(part)
        else:
            st.markdown(msg.content)

if user_prompt := st.chat_input("Ask a question about your documents..."):
    st.chat_message("user").write(user_prompt)

    with st.chat_message("ai"):
        with st.spinner("🔍 Analyzing documents..."):
            retrieved_docs = retriever.invoke(user_prompt)
        
        context_text = format_docs(retrieved_docs)
        current_images = []
        current_sources = []
        all_image_paths = []
        
        for doc in retrieved_docs:
            source_info = {
                "source": os.path.basename(_get_doc_metadata(doc, 'source', 'N/A')),
                "page": _get_doc_metadata(doc, 'page', 0) + 1,
            }
            if source_info not in current_sources:
                 current_sources.append(source_info)

            image_paths_str = _get_doc_metadata(doc, 'image_paths', '')
            if image_paths_str:
                for img_path in image_paths_str.split(';'):
                    if os.path.exists(img_path) and img_path not in current_images:
                        current_images.append(img_path)
                        all_image_paths.append(img_path)
        
        # Update conversation memory
        for source in current_sources:
            if source not in st.session_state.all_sources:
                st.session_state.all_sources.append(source)
        for img in current_images:
            if img not in st.session_state.all_images:
                st.session_state.all_images.append(img)
        
        # Update stats
        st.session_state.conversation_stats['total_queries'] += 1
        st.session_state.conversation_stats['total_sources'] = len(st.session_state.all_sources)
        st.session_state.conversation_stats['total_images'] = len(st.session_state.all_images)
        st.session_state.conversation_stats['has_ocr'] = any(_get_doc_metadata(doc, 'has_ocr', False) for doc in retrieved_docs)
        
        with st.expander(f"📊 Retrieved: {len(current_sources)} sources, {len(current_images)} images"):
            if current_sources:
                for source in current_sources: st.caption(f"📄 {source['source']} - Page {source['page']}")
            if current_images:
                st.write("🖼️ Images found:")
                img_cols = st.columns(min(3, len(current_images)))
                for i, img_path in enumerate(current_images):
                    with img_cols[i % 3]: st.image(img_path, use_container_width=True)

        if not context_text.strip() and not current_images:
            st.error("Could not find relevant content. Please try rephrasing.", icon="🤷")
        else:
            # Check if LLM is available
            if llm is None:
                # Show document search results without AI processing
                st.info("📄 **Document Search Results** (AI responses disabled)")
                st.markdown("**Found relevant content:**")
                st.markdown(context_text)
                
                if current_images:
                    st.markdown("**Images found:**")
                    img_cols = st.columns(min(3, len(current_images)))
                    for i, img_path in enumerate(current_images):
                        with img_cols[i % 3]: 
                            st.image(img_path, use_container_width=True)
                
                # Add to chat history
                msgs.add_user_message(user_prompt)
                msgs.add_ai_message(f"Document search results:\n\n{context_text}")
                
            else:
                # Full AI processing with LLM
                prompt_text = conversational_qa_prompt.format(
                    context=context_text,
                    chat_history=format_chat_history(msgs.messages[-6:]),
                    image_paths="\n".join(all_image_paths),
                    question=user_prompt
                )
                
                prompt_content = [{"type": "text", "text": prompt_text}]
                for img_path in current_images[:5]:
                    try:
                        img = Image.open(img_path)
                        if img.size[0] > 1024 or img.size[1] > 1024:
                            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                        buffered = BytesIO()
                        img_format = img.format if img.format else 'JPEG'
                        img.save(buffered, format=img_format)
                        data_url = f"data:image/{img_format.lower()};base64,{base64.b64encode(buffered.getvalue()).decode()}"
                        prompt_content.append({"type": "image_url", "image_url": {"url": data_url}})
                    except Exception as e:
                        st.warning(f"Could not process image {os.path.basename(img_path)}: {e}")

                multimodal_message = HumanMessage(content=prompt_content)

                # --- LLM INVOCATION WITH EXPONENTIAL BACKOFF ---
                max_retries = 5
                initial_backoff = 2  # seconds
                full_response = None
                
                for attempt in range(max_retries):
                    try:
                        with st.spinner(f"🧠 Generating response... (Attempt {attempt + 1}/{max_retries})"):
                            check_rate_limit() # Proactive delay
                            response = llm.invoke([multimodal_message])
                            full_response = response.content
                            break  # Success, exit loop
                    except Exception as e:
                        error_str = str(e).lower()
                        if "429" in error_str or "rate limit" in error_str:
                            if attempt < max_retries - 1:
                                wait_time = initial_backoff * (2 ** attempt) + random.uniform(0, 1)
                                st.warning(f"Rate limit hit. Retrying in {wait_time:.1f}s...", icon="⏳")
                                time.sleep(wait_time)
                            else:
                                st.error("API rate limit exceeded after multiple retries.", icon="🚨")
                                st.error(f"Final error: {e}")
                                break
                        else:
                            st.error(f"An unexpected error occurred: {e}", icon="🚨")
                            break
                
                # --- DISPLAY RESPONSE IF SUCCESSFUL ---
                if full_response:
                    # Same display logic as before
                    if "[IMAGE:" in full_response:
                        image_pattern = r'\[IMAGE:\s*(.*?)\]'
                        parts = re.split(image_pattern, full_response)
                        for i, part in enumerate(parts):
                            if i % 2 == 1:
                                image_path = part.strip()
                                found_image = next((p for p in st.session_state.all_images if os.path.basename(p) == os.path.basename(image_path)), None)
                                if found_image and os.path.exists(found_image):
                                    st.image(found_image, width=500, caption=f"📷 {os.path.basename(found_image)}")
                                else:
                                    st.warning(f"Referenced image not found: {os.path.basename(image_path)}")
                            elif part.strip():
                                st.markdown(part)
                    else:
                        st.markdown(full_response)
                    
                    msgs.add_user_message(user_prompt)
                    msgs.add_ai_message(full_response)

