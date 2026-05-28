import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import pypdf
import tempfile
import os
import hashlib

load_dotenv()

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #ffffff;
    color: #0a0a0a;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 780px;
}

.title-block {
    border-top: 3px solid #0a0a0a;
    border-bottom: 1px solid #0a0a0a;
    padding: 1.2rem 0 1rem 0;
    margin-bottom: 0.5rem;
}
.title-block h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0;
    color: #0a0a0a;
}
.title-block p {
    font-size: 0.82rem;
    color: #555;
    margin: 0.3rem 0 0 0;
    font-weight: 300;
    letter-spacing: 0.01em;
}

.disclaimer {
    background: #0a0a0a;
    color: #ffffff;
    padding: 0.7rem 1rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 1.5rem;
    letter-spacing: 0.01em;
    line-height: 1.5;
}
.disclaimer span {
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem;
    display: block;
    margin-bottom: 0.2rem;
    color: #ccc;
}

/* Upload box */
.upload-box {
    border: 1px dashed #0a0a0a;
    padding: 1rem;
    margin-bottom: 1.2rem;
    background: #fafafa;
}
.upload-box-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #0a0a0a;
    margin-bottom: 0.5rem;
}

/* Active source badge */
.source-badge {
    display: inline-block;
    background: #0a0a0a;
    color: #fff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.5rem;
    margin-bottom: 1rem;
}
.source-badge-default {
    display: inline-block;
    border: 1px solid #0a0a0a;
    color: #0a0a0a;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.5rem;
    margin-bottom: 1rem;
}

.chat-wrapper {
    border: 1px solid #d0d0d0;
    border-radius: 0px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    min-height: 200px;
    background: #fafafa;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.9rem;
}
.msg-user .bubble {
    background: #0a0a0a;
    color: #ffffff;
    padding: 0.6rem 0.9rem;
    max-width: 75%;
    font-size: 0.875rem;
    line-height: 1.55;
    font-family: 'IBM Plex Sans', sans-serif;
}

.msg-ai {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 0.9rem;
}
.msg-ai .bubble {
    background: #ffffff;
    color: #0a0a0a;
    border: 1px solid #0a0a0a;
    padding: 0.6rem 0.9rem;
    max-width: 75%;
    font-size: 0.875rem;
    line-height: 1.55;
    font-family: 'IBM Plex Sans', sans-serif;
}

.msg-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
    color: #888;
}

.stTextInput > div > div > input {
    border: 1px solid #0a0a0a !important;
    border-radius: 0px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 0.8rem !important;
    background: #fff !important;
    color: #0a0a0a !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input:focus {
    border: 1.5px solid #0a0a0a !important;
    outline: none !important;
}

.stButton > button {
    background: #0a0a0a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 0px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    padding: 0.55rem 1.2rem !important;
    text-transform: uppercase !important;
    transition: background 0.15s ease !important;
}
.stButton > button:hover {
    background: #333 !important;
}

.clear-btn > button {
    background: #fff !important;
    color: #0a0a0a !important;
    border: 1px solid #0a0a0a !important;
}
.clear-btn > button:hover {
    background: #f0f0f0 !important;
}

hr {
    border: none;
    border-top: 1px solid #d0d0d0;
    margin: 1rem 0;
}

.stSpinner > div {
    border-top-color: #0a0a0a !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #f0f0f0; }
::-webkit-scrollbar-thumb { background: #aaa; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> list[Document]:
    """Extract text from PDF bytes and return LangChain Documents."""
    reader = pypdf.PdfReader(file_bytes)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"page": i + 1}
            ))
    return docs


def build_vectorstore_from_docs(docs: list[Document]) -> Chroma:
    """Chunk documents and build an in-memory Chroma vectorstore."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    embedding_model = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
        # no persist_directory → stays in-memory for this session
    )
    return vectorstore


def get_pdf_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


# ─── Default RAG (persisted Chroma DB) ──────────────────────────────────────

@st.cache_resource
def load_default_rag():
    embedding_model = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="Chroma_db",
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 15, "lambda_mult": 0.4}
    )
    return retriever


def get_llm_and_prompt():
    llm = ChatMistralAI(model="mistral-small-2506")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a medical education assistant designed to help medical students 
and healthcare learners. Use ONLY the provided context from the medical PDF to answer questions.

Rules:
- If the answer is not in the context, say: 'I could not find this in the provided medical document.'
- Always recommend consulting a licensed medical professional for real clinical decisions.
- Do NOT provide personal medical diagnoses or treatment advice.
- Use clear medical terminology but explain terms when needed.
- If a question involves an emergency, always say: 'Please contact emergency services immediately.'"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    return llm, prompt


# ─── Session State ───────────────────────────────────────────────────────────

for key, default in [
    ("chat_history", []),
    ("messages", []),
    ("uploaded_vectorstore", None),
    ("uploaded_pdf_hash", None),
    ("uploaded_pdf_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
    <h1>Medical RAG Assistant</h1>
    <p>Upload your own PDF or use the default medical document — ask questions about anatomy, physiology, and more.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <span>Disclaimer</span>
    <strong>This tool is for educational purposes only and is not a substitute for professional medical advice.</strong>
</div>
""", unsafe_allow_html=True)


# ─── PDF Upload Box ───────────────────────────────────────────────────────────

st.markdown('<div class="upload-box">', unsafe_allow_html=True)
st.markdown('<div class="upload-box-title">📄 Upload a PDF Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Upload PDF",
    type=["pdf"],
    label_visibility="collapsed",
    help="Upload any medical PDF — the assistant will answer questions from it."
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_hash = get_pdf_hash(file_bytes)

    # Only reprocess if a new/different PDF is uploaded
    if file_hash != st.session_state.uploaded_pdf_hash:
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            import io
            docs = extract_text_from_pdf(io.BytesIO(file_bytes))
            if docs:
                vs = build_vectorstore_from_docs(docs)
                st.session_state.uploaded_vectorstore = vs
                st.session_state.uploaded_pdf_hash = file_hash
                st.session_state.uploaded_pdf_name = uploaded_file.name
                # Clear chat when a new PDF is loaded
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.success(f"✓ '{uploaded_file.name}' loaded — {len(docs)} pages indexed.")
            else:
                st.error("Could not extract text from this PDF. It may be scanned or image-based.")

    # Button to go back to default
    if st.session_state.uploaded_vectorstore is not None:
        if st.button("✕ Remove uploaded PDF & use default", key="remove_pdf"):
            st.session_state.uploaded_vectorstore = None
            st.session_state.uploaded_pdf_hash = None
            st.session_state.uploaded_pdf_name = None
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Active source indicator
if st.session_state.uploaded_pdf_name:
    st.markdown(
        f'<div class="source-badge">Active Source: {st.session_state.uploaded_pdf_name}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="source-badge-default">Active Source: Default Medical Document</div>',
        unsafe_allow_html=True
    )


# ─── Chat Display ─────────────────────────────────────────────────────────────

chat_html = '<div class="chat-wrapper">'

if not st.session_state.messages:
    chat_html += '<p style="color:#aaa; font-size:0.82rem; font-family:\'IBM Plex Mono\',monospace; text-align:center; margin-top:3rem;">Ask a question about the medical document.</p>'
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f"""
            <div class="msg-user">
                <div>
                    <div class="msg-label" style="text-align:right;">You</div>
                    <div class="bubble">{msg["content"]}</div>
                </div>
            </div>"""
        else:
            chat_html += f"""
            <div class="msg-ai">
                <div>
                    <div class="msg-label">Assistant</div>
                    <div class="bubble">{msg["content"]}</div>
                </div>
            </div>"""

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)


# ─── Input Row ────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns([7, 1.4, 1.4])

with col1:
    user_input = st.text_input(
        label="query",
        placeholder="Type your medical question...",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send = st.button("Send", use_container_width=True)

with col3:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    clear = st.button("Clear", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Handle Send ─────────────────────────────────────────────────────────────

if send and user_input.strip():
    try:
        llm, prompt = get_llm_and_prompt()

        # Pick retriever: uploaded PDF or default Chroma DB
        if st.session_state.uploaded_vectorstore is not None:
            retriever = st.session_state.uploaded_vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 15, "lambda_mult": 0.4}
            )
        else:
            retriever = load_default_rag()

        with st.spinner("Searching the document..."):
            docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = prompt.invoke({
                "context": context,
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })

            response = llm.invoke(final_prompt)
            answer = response.content

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()

    except Exception as e:
        st.error(f"Error: {str(e)}")


# ─── Handle Clear ─────────────────────────────────────────────────────────────

if clear:
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.rerun()


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="font-size:0.72rem; color:#aaa; font-family:\'IBM Plex Mono\',monospace; text-align:center;">'
    'Medical RAG Assistant — Educational use only'
    '</p>',
    unsafe_allow_html=True
)