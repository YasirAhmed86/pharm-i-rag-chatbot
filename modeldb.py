import os, sqlite3, argparse, time, re
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ===================== Config =====================
DB_PATH      = "database.db"
PDF1_PATH    = r"C:\Users\manual.pdf"
PDF2_PATH    = r"C:\Users\manual2.pdf"
SPLIT_QNO    = 193
TOP_K        = 5
CHUNK_SIZE   = 900
CHUNK_OVER   = 120
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"

# Special PC Ollama endpoint
OLLAMA_BASE_URL = "LOCAL IP"

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

def ensure_model_list_table(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_list (
          id   INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE NOT NULL
        )
    """)
    con.commit()

def ensure_embedder_list_table(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS embedder_list (
          id    INTEGER PRIMARY KEY AUTOINCREMENT,
          name  TEXT UNIQUE NOT NULL,
          dims  INTEGER,
          notes TEXT
        )
    """)
    con.commit()

def ensure_embedder_present(con: sqlite3.Connection, name: str) -> dict:
    con.execute("CREATE TABLE IF NOT EXISTS embedder_list (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, dims INTEGER, notes TEXT)")
    con.execute("INSERT OR IGNORE INTO embedder_list(name) VALUES (?)", (name,))
    row = con.execute("SELECT id, name FROM embedder_list WHERE name=?", (name,)).fetchone()
    con.commit()
    return {"id": row[0], "name": row[1]}

def migrate_model_answers_add_embedder(con: sqlite3.Connection, default_embedder_id: int) -> None:
    """
    Add embedder_id to model_answers and make (model_id, qa_id, embedder_id) unique.
    Keeps the same ids, backfills embedder_id from model_eval when possible,
    otherwise uses default_embedder_id.
    Safe to run multiple times.
    """
    # If table doesn't exist, create with the new schema and return
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_answers'"
    ).fetchone()
    if not row:
        con.execute("""
            CREATE TABLE model_answers (
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              model_id      INTEGER NOT NULL,
              qa_id         INTEGER NOT NULL,
              embedder_id   INTEGER NOT NULL,
              model_answer  TEXT NOT NULL,
              UNIQUE(model_id, qa_id, embedder_id),
              FOREIGN KEY(model_id)    REFERENCES model_list(id),
              FOREIGN KEY(qa_id)       REFERENCES qas(id),
              FOREIGN KEY(embedder_id) REFERENCES embedder_list(id)
            )
        """)
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_model_answers_mqe ON model_answers(model_id, qa_id, embedder_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_model_answers_model ON model_answers(model_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_model_answers_qa    ON model_answers(qa_id)")
        con.commit()
        return

    # If already has embedder_id, just ensure indexes and backfill any NULLs
    cols = [c[1] for c in con.execute("PRAGMA table_info(model_answers)").fetchall()]
    if "embedder_id" in cols:
        con.execute("""
            UPDATE model_answers
               SET embedder_id = COALESCE(
                   (SELECT me.embedder_id
                      FROM model_eval me
                     WHERE me.model_answer_id = model_answers.id
                     LIMIT 1),
                   ?
               )
             WHERE embedder_id IS NULL OR embedder_id=0
        """, (default_embedder_id,))
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_model_answers_mqe ON model_answers(model_id, qa_id, embedder_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_model_answers_model ON model_answers(model_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_model_answers_qa    ON model_answers(qa_id)")
        con.commit()
        return

    # Otherwise migrate with FKs OFF
    # Drop any views that mention model_answers to avoid dependency errors
    dep_views = con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='view' AND sql LIKE '%model_answers%'"
    ).fetchall()

    fk_was_on = int(con.execute("PRAGMA foreign_keys").fetchone()[0]) == 1
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute("BEGIN")
    try:
        con.execute("ALTER TABLE model_answers RENAME TO model_answers_old")
        con.execute("""
            CREATE TABLE model_answers (
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              model_id      INTEGER NOT NULL,
              qa_id         INTEGER NOT NULL,
              embedder_id   INTEGER NOT NULL,
              model_answer  TEXT NOT NULL,
              UNIQUE(model_id, qa_id, embedder_id),
              FOREIGN KEY(model_id)    REFERENCES model_list(id),
              FOREIGN KEY(qa_id)       REFERENCES qas(id),
              FOREIGN KEY(embedder_id) REFERENCES embedder_list(id)
            )
        """)
        con.execute("""
            INSERT INTO model_answers (id, model_id, qa_id, embedder_id, model_answer)
            SELECT
              ma.id,
              ma.model_id,
              ma.qa_id,
              COALESCE(
                 (SELECT me.embedder_id
                    FROM model_eval me
                   WHERE me.model_answer_id = ma.id
                   LIMIT 1),
                 ?
              ) AS embedder_id,
              ma.model_answer
            FROM model_answers_old ma
        """, (default_embedder_id,))
        con.execute("DROP TABLE model_answers_old")

        # indexes after new schema exists
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_model_answers_mqe ON model_answers(model_id, qa_id, embedder_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_model_answers_model ON model_answers(model_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_model_answers_qa    ON model_answers(qa_id)")

        # recreate views that referenced the table
        for name, vsql in dep_views:
            con.execute(vsql.replace("model_answers_old", "model_answers"))

        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        if fk_was_on:
            con.execute("PRAGMA foreign_keys=ON")
    con.commit()


    # indexes (safe to re-run)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_model_answers_mqe ON model_answers(model_id, qa_id, embedder_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_answers_model ON model_answers(model_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_answers_qa    ON model_answers(qa_id)")
    con.commit()


def ensure_model_eval_table(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_eval (
          id               INTEGER PRIMARY KEY AUTOINCREMENT,
          qa_id            INTEGER NOT NULL,
          model_id         INTEGER NOT NULL,
          model_answer_id  INTEGER NOT NULL, 
          embedder_id      INTEGER NOT NULL,
          semantic         REAL,
          bleu             REAL,
          rouge1           REAL,
          rouge2           REAL,
          rougeL           REAL,
          latency_ms       REAL,
          FOREIGN KEY(qa_id)           REFERENCES qas(id),
          FOREIGN KEY(model_id)        REFERENCES model_list(id),
          FOREIGN KEY(model_answer_id) REFERENCES model_answers(id),
          FOREIGN KEY(embedder_id)     REFERENCES embedder_list(id),
          UNIQUE(qa_id, model_id, embedder_id)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_qa        ON model_eval(qa_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_model     ON model_eval(model_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_embedder  ON model_eval(embedder_id)")
    con.commit()


    # Indexes (safe to re-run)
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_qa        ON model_eval(qa_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_model     ON model_eval(model_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_embedder  ON model_eval(embedder_id)")
    con.commit()


def load_qas(con: sqlite3.Connection, limit: int) -> List[Dict[str, Any]]:
    sql = "SELECT id, question, answer FROM qas ORDER BY id"
    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"
    rows = con.execute(sql).fetchall()
    if not rows:
        raise SystemExit("No rows in qas.")
    return [{"id": i, "question": q, "answer": a} for (i, q, a) in rows]

def ensure_models_present(con: sqlite3.Connection, names_csv: Optional[str]) -> List[Dict[str, Any]]:
    if names_csv and names_csv.strip():
        names = [m.strip() for m in names_csv.split(",") if m.strip()]
        out = []
        for name in names:
            con.execute("INSERT OR IGNORE INTO model_list(name) VALUES (?)", (name,))
            row = con.execute("SELECT id, name FROM model_list WHERE name=?", (name,)).fetchone()
            out.append({"id": row[0], "name": row[1]})
        con.commit()
        return out
    rows = con.execute("SELECT id, name FROM model_list ORDER BY id").fetchall()
    if not rows:
        raise SystemExit("No models found in model_list. Pass --models to create them.")
    return [{"id": r[0], "name": r[1]} for r in rows]

def count_questions_completed(con: sqlite3.Connection, model_ids: List[int]) -> int:
    placeholders = ",".join("?" * len(model_ids))
    sql = f"""
    SELECT COUNT(*) FROM (
      SELECT qa_id, COUNT(*) AS c
      FROM model_answers
      WHERE model_id IN ({placeholders})
      GROUP BY qa_id
      HAVING c = ?
    )
    """
    return int(con.execute(sql, (*model_ids, len(model_ids))).fetchone()[0])



def migrate_model_answers_add_embedder(con: sqlite3.Connection, default_embedder_id: int) -> None:
    """
    Rebuilds model_answers to include embedder_id and UNIQUE(model_id, qa_id, embedder_id)
    and backfills embedder_id for all existing rows.
    Safe to run multiple times.
    """
    # Ensure source table exists
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_answers'").fetchone():
        raise RuntimeError("model_answers table does not exist")

    cols = [r[1] for r in con.execute("PRAGMA table_info(model_answers)").fetchall()]
    if "embedder_id" in cols:
        # Just backfill missing values
        con.execute("""
            UPDATE model_answers
            SET embedder_id = COALESCE(
                (SELECT me.embedder_id FROM model_eval me WHERE me.model_answer_id = model_answers.id LIMIT 1),
                ?
            )
            WHERE embedder_id IS NULL OR embedder_id=0
        """, (default_embedder_id,))
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_model_answers_mqe ON model_answers(model_id, qa_id, embedder_id)")
        con.commit()
        return

    # Rebuild with embedder_id
    con.execute("ALTER TABLE model_answers RENAME TO model_answers_old")
    con.execute("""
        CREATE TABLE model_answers (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          model_id      INTEGER NOT NULL,
          qa_id         INTEGER NOT NULL,
          embedder_id   INTEGER NOT NULL,
          model_answer  TEXT NOT NULL,
          UNIQUE(model_id, qa_id, embedder_id),
          FOREIGN KEY(model_id)      REFERENCES model_list(id),
          FOREIGN KEY(qa_id)         REFERENCES qas(id),
          FOREIGN KEY(embedder_id)   REFERENCES embedder_list(id)
        )
    """)
    con.execute("""
        INSERT INTO model_answers (id, model_id, qa_id, embedder_id, model_answer)
        SELECT
          ma.id,
          ma.model_id,
          ma.qa_id,
          COALESCE( (SELECT me.embedder_id FROM model_eval me WHERE me.model_answer_id = ma.id LIMIT 1),
                    ? ),
          ma.model_answer
        FROM model_answers_old ma
    """, (default_embedder_id,))
    #con.execute("DROP TABLE model_answers_old")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_model_answers_mqe ON model_answers(model_id, qa_id, embedder_id)")
    con.commit()

def drop_unique_on_model_eval_if_present(con: sqlite3.Connection) -> None:
    """
    If model_eval was created with model_answer_id UNIQUE, rebuild it without that UNIQUE.
    Safe to run repeatedly.
    """
    row = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='model_eval'").fetchone()
    if not row:
        return
    ddl = row[0] or ""
    if "model_answer_id INTEGER NOT NULL UNIQUE" not in ddl:
        return  # already good

    con.execute("ALTER TABLE model_eval RENAME TO model_eval_old")
    con.execute("""
        CREATE TABLE model_eval (
          id               INTEGER PRIMARY KEY AUTOINCREMENT,
          qa_id            INTEGER NOT NULL,
          model_id         INTEGER NOT NULL,
          model_answer_id  INTEGER NOT NULL,
          embedder_id      INTEGER NOT NULL,
          semantic         REAL,
          bleu             REAL,
          rouge1           REAL,
          rouge2           REAL,
          rougeL           REAL,
          latency_ms       REAL,
          FOREIGN KEY(qa_id)           REFERENCES qas(id),
          FOREIGN KEY(model_id)        REFERENCES model_list(id),
          FOREIGN KEY(model_answer_id) REFERENCES model_answers(id),
          FOREIGN KEY(embedder_id)     REFERENCES embedder_list(id),
          UNIQUE(qa_id, model_id, embedder_id)
        )
    """)
    con.execute("""
        INSERT INTO model_eval
        (id, qa_id, model_id, model_answer_id, embedder_id, semantic, bleu, rouge1, rouge2, rougeL, latency_ms)
        SELECT id, qa_id, model_id, model_answer_id, embedder_id, semantic, bleu, rouge1, rouge2, rougeL, latency_ms
        FROM model_eval_old
    """)
    con.execute("DROP TABLE model_eval_old")
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_qa       ON model_eval(qa_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_model    ON model_eval(model_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_embedder ON model_eval(embedder_id)")
    con.commit()


def count_evals_for_embedder(con, embedder_id, model_ids):
    placeholders = ",".join("?" * len(model_ids))
    sql = f"""
      SELECT COUNT(DISTINCT qa_id)
      FROM model_eval
      WHERE embedder_id=? AND model_id IN ({placeholders})
    """
    return int(con.execute(sql, (embedder_id, *model_ids)).fetchone()[0])


def migrate_model_eval_drop_unique(con: sqlite3.Connection) -> None:
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_eval'"
    ).fetchone()
    if not row:
        return

    ddl = row[0] or ""
    print("[debug] current model_eval DDL:", ddl)

    # Only migrate if UNIQUE(model_answer_id) exists (robust regex)
    if re.search(r"model_answer_id\s+INTEGER\s+NOT\s+NULL\s+UNIQUE", ddl, re.I) is None:
        print("[migrate] model_eval already OK (no UNIQUE on model_answer_id).")
        return

    # Capture and drop dependent views
    dep_views = con.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='view' AND (sql LIKE '%model_eval%' OR sql LIKE '%model_eval_old%')"
    ).fetchall()

    print("[migrate] Rebuilding model_eval to drop UNIQUE(model_answer_id)…")
    con.execute("BEGIN")
    try:
        # Drop the views to avoid dependency errors during ALTER/CREATE
        for name, _sql in dep_views:
            print(f"[migrate] Dropping view {name}")
            con.execute(f'DROP VIEW IF EXISTS "{name}"')

        # Rename + rebuild table
        con.execute("ALTER TABLE model_eval RENAME TO model_eval_old")
        con.execute("""
            CREATE TABLE model_eval (
              id               INTEGER PRIMARY KEY AUTOINCREMENT,
              qa_id            INTEGER NOT NULL,
              model_id         INTEGER NOT NULL,
              model_answer_id  INTEGER NOT NULL,
              embedder_id      INTEGER NOT NULL,
              semantic         REAL,
              bleu             REAL,
              rouge1           REAL,
              rouge2           REAL,
              rougeL           REAL,
              latency_ms       REAL,
              FOREIGN KEY(qa_id)           REFERENCES qas(id),
              FOREIGN KEY(model_id)        REFERENCES model_list(id),
              FOREIGN KEY(model_answer_id) REFERENCES model_answers(id),
              FOREIGN KEY(embedder_id)     REFERENCES embedder_list(id),
              UNIQUE(qa_id, model_id, embedder_id)
            )
        """)
        con.execute("""
            INSERT INTO model_eval
            (id, qa_id, model_id, model_answer_id, embedder_id,
             semantic, bleu, rouge1, rouge2, rougeL, latency_ms)
            SELECT id, qa_id, model_id, model_answer_id, embedder_id,
                   semantic, bleu, rouge1, rouge2, rougeL, latency_ms
            FROM model_eval_old
        """)
        con.execute("DROP TABLE model_eval_old")
        con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_qa       ON model_eval(qa_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_model    ON model_eval(model_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_model_eval_embedder ON model_eval(embedder_id)")

        # Recreate the views (ensure they point to model_eval, not model_eval_old)
        for name, vsql in dep_views:
            fixed = vsql.replace("model_eval_old", "model_eval")
            print(f"[migrate] Recreating view {name}")
            con.execute(fixed)

        con.execute("COMMIT")
        print("[migrate] model_eval rebuilt successfully.")
    except Exception:
        con.execute("ROLLBACK")
        raise





# ===================== Retrieval helpers =====================
def extract_error_tokens(text: str) -> set:
    toks = set()
    toks |= set(re.findall(r"\b0?\d{3,5}\b", text))
    toks |= set(re.findall(r"\b[A-Z]\.\d{2}[A-Z]\.\d{2}\.\d{3}\b", text))
    toks |= set(re.findall(r"\bW?\d{3}[A-Z]?\d*\.\d{2}\b", text))
    toks |= set(re.findall(r"\bWIA\S{0,30}\b", text))
    return toks

def infer_ui_tags(text: str) -> list:
    hints = [
        "Messages","Filtering","Filtered Records","Export","Reset filters",
        "Settings","Groups and privilege","Switch Language","Shutdown HMI","Service","Disable horn",
        "Service Selection","Line Handler","Time on","Disk space",
        "PROCESS AND SEQUENCE","CONNECTIONS","FAULHABER","GRIPPERS","ACTIVE EQUIPMENTS",
    ]
    tags = []
    for k in hints:
        if re.search(r"\b" + re.escape(k) + r"\b", text, flags=re.IGNORECASE):
            tags.append(k.lower())
    return sorted(set(tags))

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

# ===================== Output cleaner =====================
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

def upsert_eval(con: sqlite3.Connection,
                qa_id: int,
                model_id: int,
                model_answer_id: int,
                embedder_id: int,
                m: Dict[str, Optional[float]],
                overwrite: bool):
    params = (qa_id, model_id, model_answer_id, embedder_id,
              m["semantic"], m["bleu"], m["rouge1"], m["rouge2"], m["rougeL"], m["latency_ms"])
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




# ===================== Main =====================
def _slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

def main():

    ap = argparse.ArgumentParser(description="Generate model answers using your reference + RAG from two PDFs (LangChain stack)")
    ap.add_argument("--db", default=DB_PATH, help="Path to qa.db")
    ap.add_argument("--models", default="deepseek-r1:8b", help="Comma-separated model names (default: deepseek-r1:8b)")
    ap.add_argument("--embedder", default=EMBED_MODEL, help="HF embedding model (default: BAAI/bge-small-en-v1.5)")
    ap.add_argument("--pdf1", default=PDF1_PATH, help="PDF for questions <= --split")
    ap.add_argument("--pdf2", default=PDF2_PATH, help="PDF for questions > --split")
    ap.add_argument("--split", type=int, default=SPLIT_QNO, help="Max question number for PDF1")
    ap.add_argument("--k", type=int, default=TOP_K, help="Top-k chunks to retrieve")
    ap.add_argument("--limit", type=int, default=0, help="Limit QAs (0=all)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing answers for (model_id, qa_id)")
    args = ap.parse_args()

    os.environ["OLLAMA_HOST"] = OLLAMA_BASE_URL

    con = connect_db(args.db)

    # 1) Ensure embedders (old and current) exist
    ensure_embedder_list_table(con)
    old_embedder_name = "sentence-transformers/all-MiniLM-L6-v2"  # your old embedder
    old_emb = ensure_embedder_present(con, old_embedder_name)
    embedder = ensure_embedder_present(con, args.embedder)  # current embedder

    # 2) Ensure model list table
    ensure_model_list_table(con)

    # 3) Upgrade model_answers to include embedder_id (FK-safe, keeps ids)
    migrate_model_answers_add_embedder(con, default_embedder_id=old_emb["id"])

    # 4) Ensure model_eval schema (and drop unique(model_answer_id) if it ever existed)
    migrate_model_eval_drop_unique(con)
    ensure_model_eval_table(con)

    # 5) Load models and QAs
    models = ensure_models_present(con, args.models)
    qas = load_qas(con, args.limit)
    print("Models:", ", ".join(m["name"] for m in models))
    print(f"Questions: {len(qas)}")
    model_ids = [m["id"] for m in models]
    t_start = time.perf_counter()

    # ---- Embeddings (dynamic) ----
    embeddings = HuggingFaceEmbeddings(model_name=args.embedder, encode_kwargs={"normalize_embeddings": True})
    client_settings = Settings(is_persistent=False, anonymized_telemetry=False)
    client = chromadb.Client(client_settings)

    col1 = f"pdf1_{_slug(args.embedder)}"
    col2 = f"pdf2_{_slug(args.embedder)}"
    print("Indexing PDF 1…"); retriever1 = build_retriever_from_pdf(args.pdf1, embeddings, col1, client)
    print("Indexing PDF 2…"); retriever2 = build_retriever_from_pdf(args.pdf2, embeddings, col2, client)

    def get_llm(model_name: str):
        return OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=model_name,
            temperature=0,
            top_p=0.9,
            model_kwargs={
                "num_ctx": 1024,     # cap input tokens
                "num_predict": 192,  # cap output tokens
                "num_batch": 32,     # GPU/VRAM lever
                "keep_alive": "1h",
            },
        )

    total = len(qas) * len(models)
    done = 0

    for row_idx, qa in enumerate(qas, start=1):
        qa_id = qa["id"]
        question = qa["question"]
        reference = qa["answer"]

        retriever = retriever1 if row_idx <= args.split else retriever2

        tags  = infer_ui_tags(question + " " + reference)
        codes = extract_error_tokens(question + " " + reference)
        augmented_query = (
            f"{question}\n\n"
            f"Likely sections: {', '.join(tags) if tags else 'n/a'}\n"
            f"Error tokens: {', '.join(sorted(codes)) if codes else 'n/a'}\n\n"
            f"Reference gist: {reference[:300]}"
        )

        # use invoke() (new API) rather than deprecated get_relevant_documents()
        docs = retriever.invoke(augmented_query)
        context = make_context(docs) if docs else "(no context)"
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        for m in models:
            # 0) If this eval already exists for (qa, model, embedder) and not overwriting, skip.
            if not args.overwrite:
                already = con.execute(
                    "SELECT 1 FROM model_eval WHERE qa_id=? AND model_id=? AND embedder_id=?",
                    (qa_id, m["id"], embedder["id"])
                ).fetchone()
                if already:
                    # progress bookkeeping if you want
                    done += 1
                    if done % 25 == 0:
                        print(f"Progress {done}/{total} (skipped existing eval for embedder)")
                    continue

            # 1) Answer for THIS (model_id, qa_id, embedder_id)
            row = con.execute(
                "SELECT id, model_answer FROM model_answers "
                "WHERE model_id=? AND qa_id=? AND embedder_id=?",
                (m["id"], qa_id, embedder["id"])
            ).fetchone()

            if row and not args.overwrite:
                # reuse the already generated answer for THIS embedder
                ma_id, answer_text = row[0], row[1]
                latency_ms = None
            else:
                # (re)generate the answer text
                llm = get_llm(m["name"])
                try:
                    t0 = time.perf_counter()
                    answer_text = llm.invoke(prompt)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    answer_text = f"[ERROR calling {m['name']}: {e}]"
                    latency_ms = None

                answer_text = clean_answer(answer_text)

                # write/overwrite answer for THIS embedder
                if args.overwrite:
                    con.execute(
                        "INSERT INTO model_answers(model_id, qa_id, embedder_id, model_answer) "
                        "VALUES (?,?,?,?) "
                        "ON CONFLICT(model_id, qa_id, embedder_id) "
                        "DO UPDATE SET model_answer=excluded.model_answer",
                        (m["id"], qa_id, embedder["id"], answer_text)
                    )
                else:
                    con.execute(
                        "INSERT OR IGNORE INTO model_answers(model_id, qa_id, embedder_id, model_answer) "
                        "VALUES (?,?,?,?)",
                        (m["id"], qa_id, embedder["id"], answer_text)
                    )
                con.commit()

                # refresh id
                ma_id = con.execute(
                    "SELECT id FROM model_answers WHERE model_id=? AND qa_id=? AND embedder_id=?",
                    (m["id"], qa_id, embedder["id"])
                ).fetchone()[0]

            # 2) compute metrics for THIS embedder and upsert into model_eval
            metrics = compute_all_metrics(embeddings, reference, answer_text, latency_ms)
            upsert_eval(con, qa_id, m["id"], ma_id, embedder["id"], metrics, overwrite=args.overwrite)


            done_evals = count_evals_for_embedder(con, embedder["id"], model_ids)
            elapsed = time.perf_counter() - t_start
            pct = 100.0 * done_evals / max(1, len(qas))
            print(f"Final for embedder {embedder['id']} ({embedder['name']}): "
                f"{done_evals}/{len(qas)} ({pct:.1f}%) | elapsed {elapsed/60:.1f} min")

        
    print(f"Done. Generated answers for {len(qas)} questions × {len(models)} models.")

if __name__ == "__main__":
    main()
