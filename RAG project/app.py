import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import pypdf
import io
import hashlib

load_dotenv()

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="centered"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');

/* Base Styles & Theme overrides */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F2F4F3 !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #1A221E;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E3E7E5;
}

/* Header & Typography Styling */
.title-container {
    padding: 1.5rem 0 1rem 0;
    margin-bottom: 1rem;
}
.title-container h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: #1A221E;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
}
.title-container p {
    font-size: 0.95rem;
    color: #606C64;
    font-weight: 400;
}

/* Beautiful custom disclaimer callout */
.aesthetic-disclaimer {
    background-color: #1A221E;
    color: #EAF0EC;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    font-size: 0.85rem;
    line-height: 1.5;
    margin-bottom: 2rem;
    border-left: 4px solid #70877F;
}
.aesthetic-disclaimer strong {
    color: #FFFFFF;
    display: block;
    font-family: 'Space Grotesk', sans-serif;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

/* Active status badge styling */
.status-pill {
    display: inline-flex;
    align-items: center;
    background-color: #E2E8E4;
    color: #384A41;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    border: 1px solid #CFD7D2;
}

/* Form element aesthetic patches */
div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Customizing chat inputs and secondary buttons */
.stTextInput input {
    border-radius: 10px !important;
    border: 1px solid #CFD7D2 !important;
    background-color: #FFFFFF !important;
}
.stTextInput input:focus {
    border-color: #70877F !important;
    box-shadow: 0 0 0 1px #70877F !important;
}

/* Hide unneeded generic Streamlit components */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def extract_text_from_pdf(file_bytes: bytes) -> list[Document]:
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
    )
    return vectorstore


def get_pdf_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


# Default RAG 

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


#  Session State 
for key, default in [
    ("chat_history", []),
    ("messages", []),
    ("uploaded_vectorstore", None),
    ("uploaded_pdf_hash", None),
    ("uploaded_pdf_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


#  Sidebar (Clean Document Handling Context) 

with st.sidebar:
    st.markdown("<h2 style='font-family: Space Grotesk; font-size: 1.25rem; margin-bottom: 1rem;'>Upload Here</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        label="Upload Custom Medical PDF",
        type=["pdf"],
        help="Upload an analytical medical document/textbook excerpt."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_hash = get_pdf_hash(file_bytes)

        if file_hash != st.session_state.uploaded_pdf_hash:
            with st.spinner("Processing architectural chunks..."):
                docs = extract_text_from_pdf(io.BytesIO(file_bytes))
                if docs:
                    vs = build_vectorstore_from_docs(docs)
                    st.session_state.uploaded_vectorstore = vs
                    st.session_state.uploaded_pdf_hash = file_hash
                    st.session_state.uploaded_pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.session_state.messages = []
                    st.success("Resource indexed effectively.")
                    st.rerun()
                else:
                    st.error("Unable to parse text from target document.")

    if st.session_state.uploaded_vectorstore is not None:
        st.markdown("---")
        if st.button("Reset to Default Context", use_container_width=True):
            st.session_state.uploaded_vectorstore = None
            st.session_state.uploaded_pdf_hash = None
            st.session_state.uploaded_pdf_name = None
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()
            
    if st.session_state.messages:
        if st.button("Clear Chat Threads", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()


# Main Screen Display

st.markdown("""
<div class="title-container">
    <h1>Medical RAG Assistant</h1>
    <p>Synthesize and extract complex context from medical documentation effortlessly.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="aesthetic-disclaimer">
    <strong>Educational Safeguard</strong>
    This AI chatbot provides general educational information only. It cannot diagnose illnesses, prescribe medications, or replace a real doctor. If you are experiencing a medical emergency, please call emergency services immediately.
</div>
""", unsafe_allow_html=True)

# Context Status Pill Indicator
if st.session_state.uploaded_pdf_name:
    st.markdown(f'<div class="status-pill">🟢 Source: {st.session_state.uploaded_pdf_name}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-pill"> Source: Default Reference Document</div>', unsafe_allow_html=True)


#Modern Chat Feed Interface 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])



if user_input := st.chat_input("Inquire about structural anatomy, pathways or physiology..."):
  
    with st.chat_message("user"):
        st.write(user_input)
        
    try:
        llm, prompt = get_llm_and_prompt()

        if st.session_state.uploaded_vectorstore is not None:
            retriever = st.session_state.uploaded_vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 15, "lambda_mult": 0.4}
            )
        else:
            retriever = load_default_rag()

        with st.spinner("Analyzing document cross-references..."):
            docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = prompt.invoke({
                "context": context,
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })

            response = llm.invoke(final_prompt)
            answer = response.content

        with st.chat_message("assistant"):
            st.write(answer)

        # Session Memory Arrays
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    except Exception as e:
        st.error(f"Inference Fault Encountered: {str(e)}")
