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
import uuid
from typing import List, Dict, Any

# LangChain and AI components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from unstructured.partition.pdf import partition_pdf
from langchain_chroma import Chroma

# Other libraries
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- UTILITY FUNCTIONS ---

def image_to_base64(image_path: str) -> str | None:
    """Converts an image file to a base64 data URL."""
    try:
        img = Image.open(image_path)
        buffered = BytesIO()
        img_format = img.format if img.format else 'JPEG'
        img.save(buffered, format=img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{img_format.lower()};base64,{img_str}"
    except Exception as e:
        st.warning(f"Could not process image {os.path.basename(image_path)}: {e}")
        return None

def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """Formats chat history into a string for the LLM."""
    formatted_messages = []
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("original_content", msg.get("content", ""))
        formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages)

# --- CORE LOGIC: RE-ENGINEERED PIPELINE ---

def partition_documents(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], temp_dir_path: str) -> List[Any]:
    """Partitions uploaded PDF files into a list of unstructured elements."""
    raw_elements = []
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir_path, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        try:
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
            raw_elements.extend(elements)
        except Exception as e:
            st.error(f"Error partitioning file '{file.name}': {e}", icon="⚠️")
    return raw_elements

def summarize_elements(elements: List[Any], api_key: str) -> Dict[str, Any]:
    """Generates summaries for text, tables, and images using efficient batch processing."""
    summary_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0)
    image_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0)

    texts_to_summarize, tables_to_summarize, images_to_describe = [], [], []
    original_elements_map = {}

    for el in elements:
        element_id = str(uuid.uuid4())
        original_elements_map[element_id] = el
        if 'unstructured.documents.elements.Table' in str(type(el)):
            tables_to_summarize.append({"id": element_id, "content": str(el)})
        elif 'unstructured.documents.elements.CompositeElement' in str(type(el)):
            texts_to_summarize.append({"id": element_id, "content": str(el)})
        elif 'unstructured.documents.elements.Image' in str(type(el)):
            img_path = el.metadata.image_path
            if img_path and os.path.exists(img_path):
                images_to_describe.append({"id": element_id, "path": img_path})

    # Batch process summaries
    text_contents = [item['content'] for item in texts_to_summarize]
    table_contents = [item['content'] for item in tables_to_summarize]
    
    text_summaries_res = summary_llm.batch(text_contents) if text_contents else []
    table_summaries_res = summary_llm.batch(table_contents) if table_contents else []

    image_desc_prompts = [
        HumanMessage(content=[
            {"type": "text", "text": "Describe the content and context of this image in detail."},
            {"type": "image_url", "image_url": {"url": image_to_base64(item['path'])}}
        ]) for item in images_to_describe if image_to_base64(item['path'])
    ]
    image_descriptions_res = image_llm.batch(image_desc_prompts) if image_desc_prompts else []

    # Map summaries back to their original IDs
    summaries = {}
    for i, summary in enumerate(text_summaries_res):
        summaries[texts_to_summarize[i]['id']] = summary.content
    for i, summary in enumerate(table_summaries_res):
        summaries[tables_to_summarize[i]['id']] = summary.content
    for i, desc in enumerate(image_descriptions_res):
        summaries[images_to_describe[i]['id']] = desc.content
        
    return summaries, original_elements_map


@st.cache_resource(ttl="2h")
def configure_multi_vector_retriever(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], temp_dir_path: str, api_key: str) -> MultiVectorRetriever | None:
    """Creates a MultiVectorRetriever with summaries for text, tables, and images."""
    if not uploaded_files:
        return None

    with st.status("Processing documents...", expanded=True) as status:
        st.write("Step 1/4: Partitioning documents with OCR...")
        raw_elements = partition_documents(uploaded_files, temp_dir_path)
        if not raw_elements:
            status.update(label="Processing failed: No content found.", state="error")
            return None
        
        st.write(f"Step 2/4: Found {len(raw_elements)} elements. Generating summaries...")
        summaries, original_elements = summarize_elements(raw_elements, api_key)
        if not summaries:
            status.update(label="Processing failed: Could not generate summaries.", state="error")
            return None

        st.write("Step 3/4: Storing summaries and raw elements...")
        element_ids = list(summaries.keys())
        summary_texts = list(summaries.values())
        
        vectorstore = Chroma(
            collection_name="summaries",
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        )
        docstore = InMemoryStore()
        
        vectorstore.add_texts(texts=summary_texts, ids=element_ids)
        docstore.mset([(k, original_elements[k]) for k in element_ids])

        status.update(label="Step 4/4: Retriever is ready!", state="complete", expanded=False)
    
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={'k': 5}
    )

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

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "ai", "content": "Hello! I'm ready to analyze your documents. How can I help?"}]

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
5.  If the context does not contain the answer, state that clearly.

**Chat History:**
{chat_history}

**Context:**
{context}

**Question:**
{question}
"""
conversational_qa_prompt = ChatPromptTemplate.from_template(conversational_qa_template)

# --- CHATBOT UI AND LOGIC ---

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
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
                
                st.session_state.messages.append({
                    "role": "ai", 
                    "content": full_response, 
                    "rendered_content": rendered_content
                })

            except Exception as e:
                st.error("An error occurred while generating the response.", icon="🚨")
                st.error(f"Details: {e}")
