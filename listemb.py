import os, sqlite3, argparse, time, re
from typing import List, Dict, Any, Optional
from xml.parsers.expat import model

# LangChain / RAG bits
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Metrics
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ===================== Config =====================
DB_PATH       = "qa.db"
PDF1_PATH     = r"C:\Users\manual.pdf"
PDF2_PATH     = r"C:\Users\manual2.pdf"
SPLIT_QNO     = 193
TOP_K         = 5
CHUNK_SIZE    = 900
CHUNK_OVER    = 120
MODEL_NAME    = "deepseek-r1:32b"
OLLAMA_URL    = "LOCAL IP"

PROMPT_TEMPLATE = """Using ONLY the Context, write a single concise answer.
Output only the answer text. Do not include citations, bracketed numbers, page numbers, or any 'Source:' text.

Context:
{context}

Question:
{question}
"""

# ===================== DB utils =====================
def connect_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=60.0)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=60000;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def ensure_model_present(con: sqlite3.Connection, name: str) -> Dict[str, Any]:
    con.execute("CREATE TABLE IF NOT EXISTS model_list (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL)")
    con.execute("INSERT OR IGNORE INTO model_list(name) VALUES (?)", (name,))
    row = con.execute("SELECT id, name FROM model_list WHERE name=?", (name,)).fetchone()
    con.commit()
    return {"id": row[0], "name": row[1]}

def load_qas(con: sqlite3.Connection, limit: int) -> List[Dict[str, Any]]:
    sql = "SELECT id, question, answer FROM qas ORDER BY id"
    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"
    rows = con.execute(sql).fetchall()
    if not rows:
        raise SystemExit("No rows in qas.")
    return [{"id": i, "question": q, "answer": a} for (i, q, a) in rows]

def list_embedders(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    con.execute("""CREATE TABLE IF NOT EXISTS embedder_list(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        dims INTEGER, notes TEXT)""")
    rows = con.execute("SELECT id, name FROM embedder_list ORDER BY id").fetchall()
    return [{"id": r[0], "name": r[1]} for r in rows]

# ===================== RAG helpers =====================
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

# ===================== LLM and cleaners =====================
def get_llm():
    return OllamaLLM(
        base_url=OLLAMA_URL,
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

CITE_BRACKETS = re.compile(r"\s*\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]\s*")
SOURCE_LINES  = re.compile(r"(?mi)^\s*source\s*:\s*.*$")
THINK_BLOCK   = re.compile(r"(?is)<think>.*?</think>")
GEN_TAGS      = re.compile(r"(?is)</?(?:think|analysis|reasoning)>\s*")
ANS_PREFIX    = re.compile(r"(?im)^\s*(?:final\s*answer|your\s*answer|assistant\s*answer|answer)\s*[:\-]\s*")

def clean_answer(text: str) -> str:
    if not text:
        return text
    text = THINK_BLOCK.sub("", text)
    text = GEN_TAGS.sub("", text)
    text = CITE_BRACKETS.sub(" ", text)
    text = SOURCE_LINES.sub("", text)
    text = ANS_PREFIX.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ===================== Metrics =====================
def _tok(text: str):
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())

def compute_bleu(reference: str, prediction: str) -> Optional[float]:
    if not reference or not prediction:
        return None
    ref_tokens = _tok(reference)
    hyp_tokens = _tok(prediction)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    n = min(4, max(1, len(hyp_tokens)))
    weights = tuple([1.0 / n] * n)
    try:
        smooth = SmoothingFunction().method4
    except Exception:
        smooth = None
    try:
        return float(sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smooth))
    except Exception:
        return None

try:
    _rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
except Exception:
    _rouge = None

def compute_rouge(reference: str, prediction: str) -> Dict[str, Optional[float]]:
    if not reference or not prediction or _rouge is None:
        return {"rouge1": None, "rouge2": None, "rougeL": None}
    try:
        scores = _rouge.score(reference, prediction)
        return {
            "rouge1": float(scores["rouge1"].fmeasure),
            "rouge2": float(scores["rouge2"].fmeasure),
            "rougeL": float(scores["rougeL"].fmeasure),
        }
    except Exception:
        return {"rouge1": None, "rouge2": None, "rougeL": None}

def compute_semantic(embeddings: HuggingFaceEmbeddings, reference: str, prediction: str) -> Optional[float]:
    if not reference or not prediction:
        return None
    try:
        v_ref  = np.array(embeddings.embed_query(reference), dtype=float)
        v_pred = np.array(embeddings.embed_query(prediction), dtype=float)
        denom = (np.linalg.norm(v_ref) * np.linalg.norm(v_pred)) + 1e-12
        return float(np.dot(v_ref, v_pred) / denom)
    except Exception:
        return None

def compute_all_metrics(embeddings, reference: str, prediction: str, latency_ms: Optional[float]) -> Dict[str, Optional[float]]:
    bleu  = compute_bleu(reference, prediction)
    rouge = compute_rouge(reference, prediction)
    sem   = compute_semantic(embeddings, reference, prediction)
    return {
        "semantic": sem,
        "bleu": bleu,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "latency_ms": float(latency_ms) if latency_ms is not None else None,
    }

# ===================== Inserts =====================
def upsert_model_answer(con: sqlite3.Connection,
                        model_id: int,
                        qa_id: int,
                        embedder_id: int,
                        answer_text: str,
                        overwrite: bool) -> int:
    if overwrite:
        con.execute("""
            INSERT INTO model_answers(model_id, qa_id, embedder_id, model_answer)
            VALUES (?,?,?,?)
            ON CONFLICT(model_id, qa_id, embedder_id)
            DO UPDATE SET model_answer=excluded.model_answer
        """, (model_id, qa_id, embedder_id, answer_text))
    else:
        con.execute("""
            INSERT OR IGNORE INTO model_answers(model_id, qa_id, embedder_id, model_answer)
            VALUES (?,?,?,?)
        """, (model_id, qa_id, embedder_id, answer_text))
    con.commit()
    row = con.execute("""
        SELECT id FROM model_answers
        WHERE model_id=? AND qa_id=? AND embedder_id=?
    """, (model_id, qa_id, embedder_id)).fetchone()
    return int(row[0])

def upsert_eval(con: sqlite3.Connection,
                qa_id: int,
                model_id: int,
                model_answer_id: int,
                embedder_id: int,
                metrics: Dict[str, Optional[float]],
                overwrite: bool) -> None:
    params = (qa_id, model_id, model_answer_id, embedder_id,
              metrics["semantic"], metrics["bleu"], metrics["rouge1"],
              metrics["rouge2"], metrics["rougeL"], metrics["latency_ms"])
    if overwrite:
        con.execute("""
            INSERT INTO model_eval(qa_id, model_id, model_answer_id, embedder_id,
                                   semantic, bleu, rouge1, rouge2, rougeL, latency_ms)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(qa_id, model_id, embedder_id) DO UPDATE SET
              model_answer_id=excluded.model_answer_id,
              semantic=excluded.semantic,
              bleu=excluded.bleu,
              rouge1=excluded.rouge1,
              rouge2=excluded.rouge2,
              rougeL=excluded.rougeL,
              latency_ms=excluded.latency_ms
        """, params)
    else:
        con.execute("""
            INSERT OR IGNORE INTO model_eval
            (qa_id, model_id, model_answer_id, embedder_id,
             semantic, bleu, rouge1, rouge2, rougeL, latency_ms)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, params)
    con.commit()


# --- helpers -------------------------------------------------
def get_model_name(con, model_id: int) -> str:
    row = con.execute("SELECT name FROM model_list WHERE id=?", (model_id,)).fetchone()
    return row[0] if row else str(model_id)

def get_embedder_name(con, embedder_id: int) -> str:
    row = con.execute("SELECT name FROM embedder_list WHERE id=?", (embedder_id,)).fetchone()
    return row[0] if row else str(embedder_id)

def get_eval_row_id(con, qa_id: int, model_id: int, embedder_id: int):
    row = con.execute("""
        SELECT id FROM model_eval
        WHERE qa_id=? AND model_id=? AND embedder_id=?
    """, (qa_id, model_id, embedder_id)).fetchone()
    return row[0] if row else None

def get_answer_row_id(con, model_id: int, qa_id: int, embedder_id: int):
    row = con.execute("""
        SELECT id FROM model_answers
        WHERE model_id=? AND qa_id=? AND embedder_id=?
    """, (model_id, qa_id, embedder_id)).fetchone()
    return row[0] if row else None

def get_existing_answer_ids(con, model_id: int, embedder_id: int) -> set[int]:
    rows = con.execute("""
        SELECT qa_id FROM model_answers
        WHERE model_id=? AND embedder_id=?
    """, (model_id, embedder_id)).fetchall()
    return {r[0] for r in rows}

def has_eval(con, qa_id: int, model_id: int, embedder_id: int) -> bool:
    return bool(con.execute("""
        SELECT 1 FROM model_eval
        WHERE qa_id=? AND model_id=? AND embedder_id=?
        LIMIT 1
    """, (qa_id, model_id, embedder_id)).fetchone())

# ===================== Main =====================
def parse_indexes(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out

def main():
    ap = argparse.ArgumentParser(description="Generate model answers with a fixed LLM across multiple embedders.")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--pdf1", default=PDF1_PATH)
    ap.add_argument("--pdf2", default=PDF2_PATH)
    ap.add_argument("--split", type=int, default=SPLIT_QNO, help="Max question number for PDF1")
    ap.add_argument("--limit", type=int, default=0, help="Limit QAs (0=all)")
    ap.add_argument("--embedder-indexes", default="5", help="1-based row indexes from embedder_list, comma-separated (default: 5)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing (model_id, qa_id, embedder_id)")
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    args = ap.parse_args()

    # point Ollama
    os.environ["OLLAMA_HOST"] = args.ollama_url

    con = connect_db(args.db)
    model = ensure_model_present(con, MODEL_NAME)
    qas = load_qas(con, args.limit)
    emb_rows = list_embedders(con)
    if not emb_rows:
        raise SystemExit("embedder_list is empty.")

    # select which embedders to run
    wanted_indexes = parse_indexes(args.embedder_indexes)
    emb_map = {i+1: row for i, row in enumerate(emb_rows)}  # 1-based
    chosen = []
    for idx in wanted_indexes:
        if idx not in emb_map:
            raise SystemExit(f"embedder_list has fewer than {idx} rows (requested index {idx}).")
        chosen.append(emb_map[idx])

    print(f"Using model: {model['name']}")
    print(f"QAs: {len(qas)}")
    print("Embedders to run:", ", ".join(f"{e['id']}:{e['name']}" for e in chosen))

    llm = get_llm()
    TOTAL_QAS = len(qas)

    for emb in chosen:
        print(f"\n=== Embedder {emb['id']} :: {emb['name']} ===")

        # Build retrievers once per embedder
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={
                "device": "cpu",    "trust_remote_code": True },
            encode_kwargs={"normalize_embeddings": True}
        )

        client = chromadb.Client(Settings(is_persistent=False, anonymized_telemetry=False))
        col1 = f"pdf1_{_slug(emb['name'])}"
        col2 = f"pdf2_{_slug(emb['name'])}"
        print("Indexing PDF 1…"); retriever1 = build_retriever_from_pdf(args.pdf1, embeddings, col1, client)
        print("Indexing PDF 2…"); retriever2 = build_retriever_from_pdf(args.pdf2, embeddings, col2, client)

        # Resume state
        existing_answers = get_existing_answer_ids(con, model["id"], emb["id"])
        print(f"Resume: found {len(existing_answers)} answers for model={model['id']} embedder={emb['id']}.")

        model_name = get_model_name(con, model["id"])
        embedder_name = get_embedder_name(con, emb["id"])

        t0_all = time.perf_counter()
        for row_idx, qa in enumerate(qas, start=1):
            qa_id    = qa["id"]
            question = qa["question"]
            reference = qa["answer"]

            # Choose retriever by qa_id (stable even when resuming)
            retriever = retriever1 if qa_id <= args.split else retriever2

            # Skip if already answered and not overwriting; backfill eval if missing
            if (not args.overwrite) and (qa_id in existing_answers):
                ans_row_id = get_answer_row_id(con, model["id"], qa_id, emb["id"])
                if not has_eval(con, qa_id, model["id"], emb["id"]):
                    stored = con.execute("""
                        SELECT model_answer FROM model_answers
                        WHERE model_id=? AND qa_id=? AND embedder_id=?
                    """, (model["id"], qa_id, emb["id"])).fetchone()
                    stored_answer = stored[0] if stored else ""
                    metrics = compute_all_metrics(embeddings, reference, stored_answer, None)
                    if ans_row_id is not None:
                        upsert_eval(
                            con, qa_id=qa_id, model_id=model["id"], model_answer_id=ans_row_id,
                            embedder_id=emb["id"], metrics=metrics, overwrite=False
                        )
                eval_row_id = get_eval_row_id(con, qa_id, model["id"], emb["id"])
                print(
                    f"[{row_idx}/{TOTAL_QAS}] QA {qa_id} • SKIP (already answered) "
                    f"→ model_answers.id={ans_row_id} • model_eval.id={eval_row_id}",
                    flush=True
                )
                continue

            # Call model
            try:
                t0 = time.perf_counter()
                raw = llm.invoke(PROMPT_TEMPLATE.format(
                    context=make_context(retriever.invoke(question)) or "(no context)",
                    question=question
                ))
                latency_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                raw = f"[ERROR calling {model['name']}: {e}]"
                latency_ms = None

            answer_text = clean_answer(raw)

            # Upsert model_answer
            ma_id = upsert_model_answer(
                con, model_id=model["id"], qa_id=qa_id, embedder_id=emb["id"],
                answer_text=answer_text, overwrite=args.overwrite
            )

            # Resolve actual ids (works if insert was ignored/frozen)
            ans_row_id = get_answer_row_id(con, model["id"], qa_id, emb["id"])

            # Metrics + eval
            metrics = compute_all_metrics(embeddings, reference, answer_text, latency_ms)
            if ans_row_id is not None:
                upsert_eval(
                    con, qa_id=qa_id, model_id=model["id"], model_answer_id=ans_row_id,
                    embedder_id=emb["id"], metrics=metrics, overwrite=args.overwrite
                )
            eval_row_id = get_eval_row_id(con, qa_id, model["id"], emb["id"])

            # Progress line for THIS QA only
            print(
                f"[{row_idx}/{TOTAL_QAS}] "
                f"QA {qa_id} • model='{model_name}' (id={model['id']}) "
                f"• embedder='{embedder_name}' (id={emb['id']}) "
                f"→ model_answers.id={ans_row_id} • model_eval.id={eval_row_id}",
                flush=True
            )

        total_elapsed = (time.perf_counter() - t0_all) / 60.0
        print(f"Done embedder {emb['id']} :: processed {len(qas)} QAs in {total_elapsed:.1f} min.")

if __name__ == "__main__": main()