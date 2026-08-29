import streamlit as st
import os
import re
import sqlite3
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings
import chromadb
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================
# SIMPLE CONFIG
# =====================
TOP_K        = 5
CHUNK_SIZE   = 900
CHUNK_OVER   = 120
MODEL_NAME   = "gpt-oss:20b"
OLLAMA_URL   = "Local IP"
EMBED_MODEL  = "intfloat/multilingual-e5-small"

os.environ["OLLAMA_HOST"] = OLLAMA_URL

def get_llm():
    return OllamaLLM(
        base_url = OLLAMA_URL,
        model=MODEL_NAME,
        temperature=0,
        top_p=0.95,
        model_kwargs={
            "num_ctx": 12288,
            "num_predict": 512,
            "num_batch": 512,
            "keep_alive": "30 mins",
        },
    )

# =====================
# STREAMLIT LAYOUT (minimal)
# =====================
st.set_page_config(page_title="Pharm-I", layout="wide")
APP_NAME = "🔬 Pharm-I"
logo_path = "C:/Users/logo.jpg"

with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)


c1, c2 = st.columns([0.85, 0.15])
with c1:
    st.markdown(f"<h1 style='margin-bottom:0'>{APP_NAME}</h1>", unsafe_allow_html=True)
    st.caption("AI Chatbot System by Pharma Integration")
with c2:
    if os.path.exists(logo_path):
        st.image(logo_path)

# =====================
# SQLite chat history (simple)
# =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_generated_answer" not in st.session_state:
    st.session_state.latest_generated_answer = None

db_path = "local_chat_history.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_query TEXT,
        bot_response TEXT
    )
    """
)
conn.commit()

def save_chat_history(user_query: str, bot_response: str):
    cursor.execute("INSERT INTO chat_history (user_query, bot_response) VALUES (?, ?)", (user_query, bot_response))
    conn.commit()

# =====================
# Cleaning (NO THINKING)
# =====================
THINK_BLOCK   = re.compile(r"(<think>.*?</think>|\[analysis\].*?\[/analysis\]|<\|.*?internal.*?\|>.*?<\|/.*?\|>)",
                           re.IGNORECASE | re.DOTALL)
GEN_TAGS      = re.compile(r"</?(?:analysis|final|scratchpad)>|<\|.*?\|>", re.IGNORECASE)
CITE_BRACKETS = re.compile(r"\s*\[\d+\]\s*")
ANS_PREFIX    = re.compile(r"^(?:answer|assistant)[:\-\s]+", re.IGNORECASE)

def clean_answer(text: str) -> str:
    if not text:
        return text
    text = THINK_BLOCK.sub("", text)
    text = GEN_TAGS.sub("", text)
    text = CITE_BRACKETS.sub(" ", text)
    text = ANS_PREFIX.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# =====================
# Helpers (PDF-only RAG)
# =====================
def _slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

def build_retriever_from_pdf(pdf_path: str,
                             embeddings: HuggingFaceEmbeddings,
                             collection_name: str,
                             client: chromadb.Client,
                             chunk_size: int = CHUNK_SIZE,
                             chunk_overlap: int = CHUNK_OVER):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = PyPDFLoader(pdf_path).load()
    for i, p in enumerate(pages):
        p.metadata["source"] = os.path.basename(pdf_path)
        p.metadata["page"] = i + 1
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.split_documents(pages)
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        client=client,
    )
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 40, "lambda_mult": 0.3},
    )
    return retriever

def make_context(docs: List[Document]) -> str:
    blocks, seen = [], set()
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        blocks.append(d.page_content)
        if len(blocks) >= TOP_K:
            break
    return "\n\n".join(blocks)

def build_refs_from_docs(docs: List[Document]) -> List[str]:
    seen = set()
    refs = []
    for d in docs:
        src = d.metadata.get("source", "Manual")
        page = d.metadata.get("page", "N/A")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        refs.append(f"- {src} — page {page}")
        if len(refs) >= TOP_K:
            break
    return refs

# =====================
# Upload PDFs → per-PDF retrievers
# =====================
uploaded_files = st.file_uploader("Upload Manuals (PDF only)", type=["pdf"], accept_multiple_files=True)

client_settings = Settings(is_persistent=False, anonymized_telemetry=False)
client = chromadb.Client(client_settings)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})

manual_retrievers = []
os.makedirs("temp", exist_ok=True)
if uploaded_files:
    suffix = EMBED_MODEL.split("/")[-1].replace("-", "_").replace(".", "_")
    for uf in uploaded_files:
        path = os.path.join("temp", uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        try:
            coll = f"manual_{_slug(uf.name)}_{suffix}"
            r = build_retriever_from_pdf(path, embeddings, coll, client, CHUNK_SIZE, CHUNK_OVER)
            manual_retrievers.append(r)
        except Exception as e:
            st.error(f"Failed to index {uf.name}: {e}")

# =====================
# LLM + prompt
# =====================
llm = get_llm()
SYSTEM_TMPL = """You are a cautious pharma QA assistant. Use ONLY the provided Context to answer.
If the answer is not fully supported by the Context, say: "I don't know based on the provided documents."
Return concise answers (no private notes or thinking). Do not include inline [1]-style citations.
Provide references at the end under "References:" listing file names and page numbers.

Context:
{context}

User: {input}
Assistant:"""

# =====================
# Ask / Answer UI
# =====================
st.markdown("### Ask a question")
with st.form("qa_form", clear_on_submit=False):
    user_query = st.text_area("Your question", height=120, placeholder="Type your question here…")
    submitted = st.form_submit_button("Ask")

answer_text = ""
refs_list: List[str] = []

if submitted and user_query:
    try:
        if not manual_retrievers:
            answer_text = "Please upload at least one manual PDF."
            refs_list = []
        else:
            # Aggregate top docs across PDFs
            docs: List[Document] = []
            for r in manual_retrievers:
                try:
                    docs.extend(r.get_relevant_documents(user_query))
                except Exception:
                    pass

            # Dedup by (source,page) and cap
            seen, uniq_docs = set(), []
            for d in docs:
                key = (d.metadata.get("source"), d.metadata.get("page"))
                if key in seen:
                    continue
                seen.add(key)
                uniq_docs.append(d)
                if len(uniq_docs) >= TOP_K:
                    break

            if not uniq_docs:
                answer_text = "I don't know based on the provided documents."
                refs_list = []
            else:
                context_str = make_context(uniq_docs)
                prompt = SYSTEM_TMPL.format(context=context_str, input=user_query)
                raw = llm.invoke(prompt)
                answer_text = clean_answer(str(raw))
                refs_list = build_refs_from_docs(uniq_docs)

        # Save + show
        st.session_state.chat_history.append((user_query, answer_text))
        st.session_state.latest_generated_answer = answer_text
        save_chat_history(user_query, answer_text)

    except Exception as e:
        answer_text = f"Error: {str(e)}"
        refs_list = []

# =====================
# Answer area
# =====================
if answer_text:
    st.markdown("### Answer")
    st.write(answer_text)
    if refs_list:
        st.markdown("**References:**")
        for r in refs_list:
            st.markdown(r)
    st.markdown("---")

# =====================
# Recent chat history (simple)
# =====================
st.markdown("### 📝 Recent Chat History")
for q, a in st.session_state.chat_history[-10:][::-1]:
    with st.expander(q):
        st.write(a)
