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
import base64
from io import BytesIO
import re
import uuid

# LangChain and AI components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from unstructured.partition.pdf import partition_pdf
from langchain_chroma import Chroma

# Other libraries
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- UTILITY FUNCTIONS ---

def format_chat_history(chat_history):
    """Formats chat history into a string."""
    return "\n".join(f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history)

def image_to_base64(image_path):
    """Converts an image file to a base64 data URL."""
    try:
        img = Image.open(image_path)
        buffered = BytesIO()
        img_format = img.format if img.format else 'JPEG'
        img.save(buffered, format=img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{img_format.lower()};base64,{img_str}"
    except Exception:
        return None

# --- CORE LOGIC: MULTI-VECTOR RETRIEVER ---

@st.cache_resource(ttl="2h")
def configure_multi_vector_retriever(uploaded_files, temp_dir_path, api_key):
    """
    Creates a MultiVectorRetriever with summaries for text, tables, and images.
    """
    if not uploaded_files:
        return None

    processing_log = st.empty()

    # 1. Partition documents into raw elements
    processing_log.info("Step 1/4: Partitioning documents into text, tables, and images...")
    raw_pdf_elements = []
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir_path, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        
        try:
            # Use unstructured to partition the PDF, which includes OCR
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
            raw_pdf_elements.extend(elements)
        except Exception as e:
            st.error(f"Error partitioning file '{file.name}': {e}", icon="⚠️")
            continue
    
    if not raw_pdf_elements:
        st.warning("Could not partition any documents. Please check the file format.")
        return None
    
    processing_log.info(f"Step 2/4: Found {len(raw_pdf_elements)} elements. Generating summaries...")

    # 2. Create summaries and associate with original elements
    summary_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0)
    image_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0)

    element_summaries = []
    element_ids = []
    original_elements = {}
    
    for element in raw_pdf_elements:
        element_id = str(uuid.uuid4())
        summary = ""
        
        if 'unstructured.documents.elements.Table' in str(type(element)):
            summary = summary_llm.invoke("Summarize the following table content: \n\n" + str(element)).content
        elif 'unstructured.documents.elements.CompositeElement' in str(type(element)):
            summary = summary_llm.invoke("Summarize the following text chunk: \n\n" + str(element)).content
        elif 'unstructured.documents.elements.Image' in str(type(element)):
            img_path = element.metadata.image_path
            if img_path and os.path.exists(img_path):
                data_url = image_to_base64(img_path)
                if data_url:
                    desc = image_llm.invoke([
                        HumanMessage(content=[
                            {"type": "text", "text": "Describe the content and context of this image in detail."},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ])
                    ])
                    summary = desc.content
        
        if summary:
            element_summaries.append(summary)
            element_ids.append(element_id)
            original_elements[element_id] = element

    processing_log.info("Step 3/4: Storing summaries and raw elements...")

    # 3. Store raw elements and create the MultiVectorRetriever
    vectorstore = Chroma(
        collection_name="summaries",
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    )
    docstore = InMemoryStore()
    
    vectorstore.add_texts(texts=element_summaries, ids=element_ids)
    docstore.mset(list(original_elements.items()))

    processing_log.success("Step 4/4: Retriever is ready! You can now ask questions.")
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={'k': 5}
    )
    return retriever

# --- MAIN APP ---

st.set_page_config(page_title="True Multimodal RAG", page_icon="🧩")
st.title("🧩 True Multimodal RAG Assistant")

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
        label="Supports PDF files for multimodal analysis.",
        type=["pdf"],
        accept_multiple_files=True
    )
    if st.button("Clear Conversation & Relaunch"):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown("---")
    st.info("This app uses an advanced Multi-Vector RAG pipeline to understand text, tables, and images.", icon="ℹ️")

if not uploaded_files:
    st.info("Please upload PDF documents in the sidebar to start.")
    st.stop()

retriever = configure_multi_vector_retriever(uploaded_files, st.session_state.temp_dir, google_api_key)
if not retriever:
    st.warning("Retriever could not be configured. Please check your files and try again.")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hello! I'm ready to analyze your documents. How can I help?"}
    ]

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
    google_api_key=google_api_key,
    safety_settings=safety_settings,
)

conversational_qa_template = """You are an expert research assistant. Your goal is to provide clear, accurate, and well-formatted answers based on the provided context, which may include text, tables, and images.

**Instructions:**
1.  Analyze the user's question and the provided chat history.
2.  Carefully examine the context, which is composed of raw text, tables, and images.
3.  When you reference information from an image, you MUST embed it in your response using the placeholder format: [IMAGE: <path_to_image>] where <path_to_image> is the full path provided in the context.
4.  Synthesize all information to construct a comprehensive answer. Format it for clarity using Markdown (tables, lists, bolding).
5.  Conclude your response with a "Key Highlights" section.
6.  If the context does not contain the answer, state that clearly.

**Chat History:**
{chat_history}

**Context:**
{context}

**Question:**
{question}
"""
conversational_qa_prompt = ChatPromptTemplate.from_template(conversational_qa_template)

# --- CHATBOT UI AND LOGIC ---

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # This part is for displaying the final rendered content
        if "rendered_content" in msg:
            for part in msg["rendered_content"]:
                if part["type"] == "text":
                    st.markdown(part["content"])
                elif part["type"] == "image":
                    if os.path.exists(part["path"]):
                        st.image(part["path"])
        else:
            st.markdown(msg["content"])


if user_prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    with st.chat_message("ai"):
        retrieved_elements = retriever.invoke(user_prompt)
        
        context_text = "\n\n".join([str(el) for el in retrieved_elements if 'Image' not in str(type(el))])
        context_images = [el.metadata['image_path'] for el in retrieved_elements if 'Image' in str(type(el))]
        
        with st.expander("📚 View Retrieved Context"):
            st.markdown("#### Text & Tables:")
            st.text(context_text)
            st.markdown("#### Images:")
            for img_path in context_images:
                if os.path.exists(img_path):
                    st.image(img_path, width=200)

        prompt_content_text = conversational_qa_template.format(
            context=context_text,
            chat_history=format_chat_history(st.session_state.messages),
            question=user_prompt
        )
        
        if not context_text.strip() and not context_images:
            st.error("Could not find any relevant context. Please try another question.", icon="🤷")
        else:
            prompt_content = [{"type": "text", "text": prompt_content_text}]
            for img_path in context_images:
                data_url = image_to_base64(img_path)
                if data_url:
                    prompt_content.append({"type": "image_url", "image_url": {"url": data_url}})

            multimodal_message = HumanMessage(content=prompt_content)

            try:
                response = llm.invoke([multimodal_message])
                full_response = response.content

                # Parse the response and render text and images
                image_pattern = r'\[IMAGE:\s*(.*?)\]'
                parts = re.split(image_pattern, full_response)
                
                rendered_content = []
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        image_path = part.strip()
                        if os.path.exists(image_path):
                            st.image(image_path)
                            rendered_content.append({"type": "image", "path": image_path})
                        else:
                            st.warning(f"AI referenced an image that could not be found: {image_path}")
                    else:
                        if part.strip():
                            st.markdown(part)
                            rendered_content.append({"type": "text", "content": part})
                
                # Store the original response and the rendered parts for persistent display
                st.session_state.messages.append({
                    "role": "ai", 
                    "content": full_response, 
                    "rendered_content": rendered_content
                })

            except Exception as e:
                st.error("An error occurred while generating the response.", icon="🚨")
                st.error(f"Details: {e}")
