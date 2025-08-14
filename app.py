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
import pytesseract
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple, Optional
import json

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

# --- ENHANCED UTILITY FUNCTIONS ---

def format_docs(docs):
    """Prepares the retrieved documents for insertion into the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(chat_history):
    """Formats chat history into a string."""
    return "\n".join(f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history)

def _get_doc_metadata(doc, key, default=None):
    """Safely retrieves a key from a document's metadata."""
    return doc.metadata.get(key, default)

def calculate_term_frequencies(text: str) -> Counter:
    """Calculate term frequencies in the given text."""
    # Simple tokenization and frequency calculation
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return Counter(filtered_words)

def intelligent_chunk_size(text: str, base_chunk_size: int = 4000) -> int:
    """Dynamically adjust chunk size based on term frequency analysis."""
    term_freq = calculate_term_frequencies(text)
    
    if not term_freq:
        return base_chunk_size
    
    # Calculate diversity score (unique terms / total terms)
    total_words = sum(term_freq.values())
    unique_words = len(term_freq)
    diversity_score = unique_words / total_words if total_words > 0 else 0
    
    # Adjust chunk size based on content density
    if diversity_score > 0.7:  # High diversity - smaller chunks
        return max(int(base_chunk_size * 0.7), 2000)
    elif diversity_score < 0.3:  # Low diversity - larger chunks
        return min(int(base_chunk_size * 1.3), 6000)
    else:
        return base_chunk_size

def enhanced_ocr_extraction(image_path: str) -> str:
    """Enhanced OCR with multiple configurations for robust text extraction."""
    try:
        img = Image.open(image_path)
        
        # Multiple OCR configurations for better accuracy
        configs = [
            '--oem 3 --psm 6',  # Uniform block of text
            '--oem 3 --psm 3',  # Fully automatic page segmentation
            '--oem 3 --psm 11', # Sparse text
            '--oem 3 --psm 12', # Sparse text with OSD
        ]
        
        extracted_texts = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(img, config=config)
                if text.strip():
                    extracted_texts.append(text.strip())
            except:
                continue
        
        # Return the longest extracted text (likely most accurate)
        if extracted_texts:
            return max(extracted_texts, key=len)
        
        return ""
    except Exception as e:
        st.warning(f"OCR extraction failed for {image_path}: {e}")
        return ""

def calculate_image_relevance(image_text: str, surrounding_text: str, context_window: int = 500) -> float:
    """Calculate relevance score between image content and surrounding text."""
    if not image_text.strip() or not surrounding_text.strip():
        return 0.0
    
    # Extract key terms from both texts
    image_terms = set(calculate_term_frequencies(image_text).keys())
    text_terms = set(calculate_term_frequencies(surrounding_text).keys())
    
    # Calculate Jaccard similarity
    intersection = len(image_terms.intersection(text_terms))
    union = len(image_terms.union(text_terms))
    
    return intersection / union if union > 0 else 0.0

def load_pdf_with_enhanced_images(file_path, temp_dir_path):
    """Enhanced PDF loading with intelligent image processing and OCR."""
    documents = []
    image_save_dir = os.path.join(temp_dir_path, "images", os.path.splitext(os.path.basename(file_path))[0])
    os.makedirs(image_save_dir, exist_ok=True)
    
    pdf_document = fitz.open(file_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        image_data = []

        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(image_save_dir, image_filename)
            
            # Save image
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            # Enhanced OCR extraction
            ocr_text = enhanced_ocr_extraction(image_path)
            
            # Calculate relevance to surrounding text
            relevance_score = calculate_image_relevance(ocr_text, text)
            
            image_data.append({
                'path': image_path,
                'ocr_text': ocr_text,
                'relevance': relevance_score,
                'size': len(image_bytes)
            })
        
        # Sort images by relevance score (descending)
        image_data.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Prepare metadata
        image_paths = [img['path'] for img in image_data]
        image_texts = [img['ocr_text'] for img in image_data if img['ocr_text']]
        
        # Combine text with OCR text for better context
        combined_text = text
        if image_texts:
            combined_text += "\n\n[Image Content]:\n" + "\n".join(image_texts)
        
        documents.append(Document(
            page_content=combined_text,
            metadata={
                'source': os.path.basename(file_path),
                'page': page_num,
                'image_paths': ";".join(image_paths),
                'image_relevance_scores': ";".join([str(img['relevance']) for img in image_data]),
                'has_images': len(image_paths) > 0
            }
        ))
    
    return documents

def create_intelligent_text_splitter(documents):
    """Create a text splitter with dynamic chunk sizes based on content analysis."""
    # Analyze all documents to determine optimal chunking strategy
    all_text = " ".join([doc.page_content for doc in documents])
    optimal_chunk_size = intelligent_chunk_size(all_text)
    
    # Calculate overlap based on chunk size (10-15% of chunk size)
    overlap = max(int(optimal_chunk_size * 0.12), 200)
    
    return RecursiveCharacterTextSplitter(
        chunk_size=optimal_chunk_size, 
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def determine_answer_format(question: str, retrieved_docs: List[Document]) -> str:
    """Intelligently determine the best answer format based on question type and content."""
    question_lower = question.lower()
    
    # Analyze question patterns
    comparison_keywords = ['compare', 'versus', 'vs', 'difference', 'similarities', 'contrast']
    list_keywords = ['list', 'enumerate', 'steps', 'process', 'procedure', 'how to']
    analysis_keywords = ['analyze', 'explain', 'why', 'how', 'what causes', 'impact', 'effect']
    summary_keywords = ['summarize', 'summary', 'overview', 'brief', 'main points']
    
    # Check for numerical data that might warrant tables
    has_numbers = any(re.search(r'\d+', doc.page_content) for doc in retrieved_docs)
    has_structured_data = any('|' in doc.page_content or '\t' in doc.page_content for doc in retrieved_docs)
    
    if any(keyword in question_lower for keyword in comparison_keywords):
        return "comparison"
    elif any(keyword in question_lower for keyword in list_keywords):
        return "step_by_step"
    elif any(keyword in question_lower for keyword in summary_keywords):
        return "summary"
    elif has_structured_data or (has_numbers and ('data' in question_lower or 'table' in question_lower)):
        return "table"
    elif any(keyword in question_lower for keyword in analysis_keywords):
        return "analytical"
    else:
        return "conversational"

# --- STATE MANAGEMENT AND CACHING ---

@st.cache_resource(ttl="2h")
def configure_retriever(uploaded_files, temp_dir_path):
    """Enhanced retriever configuration with intelligent processing."""
    docs = []
    
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir_path, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        
        try:
            file_extension = os.path.splitext(file.name)[1].lower()
            if file_extension == '.pdf':
                loaded_docs = load_pdf_with_enhanced_images(temp_filepath, temp_dir_path)
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

    # Use intelligent text splitter
    text_splitter = create_intelligent_text_splitter(docs)
    doc_chunks = text_splitter.split_documents(docs)

    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key not found in Streamlit secrets. Please add it.")
        st.stop()

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    chroma_settings = Settings(anonymized_telemetry=False)
    vectorstore = Chroma.from_documents(doc_chunks, embeddings_model, client_settings=chroma_settings)

    return vectorstore.as_retriever(search_kwargs={"k": 12})

def create_adaptive_prompt(format_type: str) -> str:
    """Create prompts that adapt to the question type and expected answer format."""
    
    base_instructions = """You are an expert research assistant. Your goal is to provide clear, accurate, and well-formatted answers based on the provided context, which may include text and images.

**Core Instructions:**
1. Analyze the user's question and the provided chat history carefully.
2. Examine the context thoroughly, including any text and images.
3. When referencing images, embed them using: [IMAGE: <path_to_image>]
4. Base your answer strictly on the provided context - do not add external information.
5. If the context doesn't contain the answer, state this clearly.

**Chat History:**
{chat_history}

**Context:**
{context}

**Available Image Paths:**
{image_paths}

**Question:**
{question}
"""

    format_specific_instructions = {
        "comparison": """
**Response Format for Comparison:**
- Provide a clear comparison addressing the key differences and similarities
- Use structured sections if helpful (e.g., "Key Differences:", "Similarities:")
- Support points with specific evidence from the context
- Be balanced and objective in your analysis
""",
        "step_by_step": """
**Response Format for Process/Steps:**
- Present information in a logical, sequential order
- Use numbered steps or clear progression markers
- Include relevant details for each step
- Highlight important warnings or considerations
""",
        "summary": """
**Response Format for Summary:**
- Provide a concise yet comprehensive overview
- Prioritize the most important information
- Use clear, accessible language
- Structure with main points and supporting details
""",
        "table": """
**Response Format for Data/Tables:**
- Present structured data as Markdown tables when appropriate
- Include relevant columns and clear headers
- Add context or explanation below tables if needed
- Highlight key insights from the data
""",
        "analytical": """
**Response Format for Analysis:**
- Provide in-depth examination of the topic
- Explain relationships, causes, and effects
- Support arguments with evidence from the context
- Consider multiple perspectives where relevant
""",
        "conversational": """
**Response Format for General Questions:**
- Provide a natural, conversational response
- Structure information logically
- Use appropriate formatting (lists, emphasis) to enhance readability
- Maintain an informative yet accessible tone
"""
    }
    
    return base_instructions + format_specific_instructions.get(format_type, format_specific_instructions["conversational"])

# --- PERSISTENT CONTEXT MANAGEMENT ---

def initialize_session_state():
    """Initialize session state variables for persistent context."""
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    
    if 'persistent_sources' not in st.session_state:
        st.session_state.persistent_sources = []
    
    if 'persistent_images' not in st.session_state:
        st.session_state.persistent_images = []
    
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = []

def update_persistent_context(retrieved_docs, image_paths):
    """Update persistent context with new sources and images."""
    # Add new sources
    new_sources = []
    for doc in retrieved_docs:
        source_info = {
            "source": os.path.basename(_get_doc_metadata(doc, 'source', 'N/A')),
            "page": _get_doc_metadata(doc, 'page', 0) + 1,
            "content_preview": doc.page_content[:250] + "...",
            "has_images": _get_doc_metadata(doc, 'has_images', False)
        }
        if source_info not in st.session_state.persistent_sources:
            st.session_state.persistent_sources.append(source_info)
            new_sources.append(source_info)
    
    # Add new images
    for img_path in image_paths:
        if img_path not in st.session_state.persistent_images and os.path.exists(img_path):
            st.session_state.persistent_images.append(img_path)
    
    return new_sources

# --- MAIN APP ---

st.set_page_config(page_title="Enhanced Multimodal RAG Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Enhanced Multimodal RAG Assistant")
st.markdown("*Featuring intelligent chunking, persistent context, enhanced OCR, and adaptive responses*")

# Initialize session state
initialize_session_state()

google_api_key = st.secrets.get("GOOGLE_API_KEY")
if not google_api_key:
    st.info("Please add your Google API key to the Streamlit secrets to continue.")
    st.stop()

# --- SIDEBAR AND FILE UPLOAD ---
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_files = st.file_uploader(
        label="Supports PDF, DOCX, TXT, HTML, PPTX, CSV",
        type=["pdf", "docx", "txt", "html", "pptx", "csv"],
        accept_multiple_files=True
    )
    
    if st.button("🗑️ Clear Conversation & Files"):
        for key in ['persistent_sources', 'persistent_images', 'conversation_context', 'langchain_messages']:
            if key in st.session_state:
                st.session_state[key] = []
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Persistent Context Display
    if st.session_state.persistent_sources:
        st.subheader("📚 Session Sources")
        with st.expander(f"View {len(st.session_state.persistent_sources)} Sources"):
            for source in st.session_state.persistent_sources[-5:]:  # Show last 5
                st.markdown(f"**{source['source']}** (Page {source['page']})")
                st.caption(source['content_preview'])
    
    if st.session_state.persistent_images:
        st.subheader("🖼️ Session Images")
        with st.expander(f"View {len(st.session_state.persistent_images)} Images"):
            for img_path in st.session_state.persistent_images[-3:]:  # Show last 3
                if os.path.exists(img_path):
                    st.image(img_path, width=150)
    
    st.markdown("---")
    st.info("✨ **Enhanced Features:**\n- Intelligent chunking\n- Persistent context\n- Enhanced OCR\n- Adaptive responses", icon="ℹ️")
    st.caption("Note: This app uses a free API tier. If you encounter errors, please wait a minute before trying again.")

if not uploaded_files:
    st.info("Please upload your documents in the sidebar to start chatting.")
    st.stop()

retriever = configure_retriever(uploaded_files, st.session_state.temp_dir)
if not retriever:
    st.stop()

msgs = StreamlitChatMessageHistory(key="langchain_messages")

# --- LLM SETUP ---
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

# --- CHATBOT UI AND LOGIC ---
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello! I'm your enhanced RAG assistant with intelligent processing capabilities. How can I help you analyze your documents?")

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Display chat messages
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            # Parse and display text with inline images
            if msg.type == "ai" and "[IMAGE:" in msg.content:
                image_pattern = r'\[IMAGE:\s*(.*?)\]'
                parts = re.split(image_pattern, msg.content)
                
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # This is an image path
                        image_path = part.strip()
                        if os.path.exists(image_path):
                            st.image(image_path, width=300)
                        else:
                            st.warning(f"Referenced image not found: {os.path.basename(image_path)}")
                    else:  # This is text
                        if part.strip():
                            st.markdown(part)
            else:
                st.write(msg.content)

    # Chat input
    if user_prompt := st.chat_input("Ask a question about your documents..."):
        st.chat_message("user").write(user_prompt)

        with st.chat_message("ai"):
            retrieved_docs = retriever.invoke(user_prompt)
            
            # Determine optimal answer format
            answer_format = determine_answer_format(user_prompt, retrieved_docs)
            
            context_text = format_docs(retrieved_docs)
            image_sources_to_pass = []
            all_image_paths = []
            
            # Process retrieved documents and extract images
            for doc in retrieved_docs:
                image_paths_str = _get_doc_metadata(doc, 'image_paths', '')
                if image_paths_str:
                    image_paths_list = image_paths_str.split(';')
                    for img_path in image_paths_list:
                        if os.path.exists(img_path):
                            image_sources_to_pass.append(img_path)
                            all_image_paths.append(img_path)
            
            # Update persistent context
            new_sources = update_persistent_context(retrieved_docs, all_image_paths)
            
            # Create adaptive prompt
            adaptive_template = create_adaptive_prompt(answer_format)
            conversational_qa_prompt = ChatPromptTemplate.from_template(adaptive_template)
            
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
                
                # Add images to prompt
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
                        st.warning(f"Could not process image {os.path.basename(img_path)}: {e}")

                multimodal_message = HumanMessage(content=prompt_content)

                try:
                    with st.spinner("Generating enhanced response..."):
                        response = llm.invoke([multimodal_message])
                        full_response = response.content

                        # Parse and render response with inline images
                        image_pattern = r'\[IMAGE:\s*(.*?)\]'
                        parts = re.split(image_pattern, full_response)

                        for i, part in enumerate(parts):
                            if i % 2 == 1:  # This is an image path
                                image_path = part.strip()
                                if os.path.exists(image_path):
                                    st.image(image_path, width=400)
                                else:
                                    st.warning(f"AI referenced an image that could not be found: {os.path.basename(image_path)}")
                            else:  # This is text
                                if part.strip():
                                    st.markdown(part)

                        msgs.add_user_message(user_prompt)
                        msgs.add_ai_message(full_response)
                        
                        # Update conversation context
                        st.session_state.conversation_context.append({
                            'question': user_prompt,
                            'answer_format': answer_format,
                            'sources_used': len(retrieved_docs),
                            'images_used': len(image_sources_to_pass)
                        })

                except Exception as e:
                    st.error("An error occurred while generating the response. This might be due to API rate limits or other issues.", icon="🚨")
                    st.error(f"Details: {e}")

# Right column for additional context
with col2:
    if st.session_state.conversation_context:
        st.subheader("💬 Conversation Analytics")
        
        # Show recent question insights
        recent_context = st.session_state.conversation_context[-5:]
        
        with st.expander("Recent Questions", expanded=False):
            for ctx in reversed(recent_context):
                st.markdown(f"**Q:** {ctx['question'][:50]}...")
                st.caption(f"Format: {ctx['answer_format']} | Sources: {ctx['sources_used']} | Images: {ctx['images_used']}")
                st.markdown("---")
        
        # Analytics summary
        total_questions = len(st.session_state.conversation_context)
        total_sources = sum(ctx['sources_used'] for ctx in st.session_state.conversation_context)
        total_images = sum(ctx['images_used'] for ctx in st.session_state.conversation_context)
        
        st.metric("Questions Asked", total_questions)
        st.metric("Sources Referenced", total_sources)
        st.metric("Images Processed", total_images)
