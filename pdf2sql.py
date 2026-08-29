import sys, re, sqlite3, pathlib
from typing import Optional, List, Dict, Any, Tuple
import pdfplumber

# -------- Defaults --------
DEFAULT_DIR = pathlib.Path(r"C:\Users")
PREFERRED_FILENAME = "pdf_file.pdf"

# -------- Patterns --------
# "Question 123:" or "Question No 123:" (question may continue on next lines)
Q_HEADER_RE = re.compile(r"^\s*Question\s*(?:No\.?\s*)?(\d+)\s*[:\-\.\)]\s*(.*)$", re.IGNORECASE)
# "Answer:" either on its own line or inline
ANS_HEAD_RE   = re.compile(r"^\s*Answer\s*[:\-\.]?\s*$", re.IGNORECASE)
ANS_INLINE_RE = re.compile(r"^\s*Answer\s*[:\-\.]\s*(.+)$", re.IGNORECASE)

# -------- Path resolver --------
def resolve_pdf_path(arg_path: Optional[str]) -> pathlib.Path:
    """
    None/"" -> DEFAULT_DIR, prefer PREFERRED_FILENAME else newest .pdf
    Directory -> prefer PREFERRED_FILENAME else newest .pdf
    File path -> use it
    """
    base = DEFAULT_DIR if not arg_path or arg_path.strip() == "" else pathlib.Path(arg_path)

    if base.is_file():
        return base

    if base.is_dir():
        # try preferred
        for p in base.glob("*.pdf"):
            if p.name.lower() == PREFERRED_FILENAME.lower():
                return p
        # newest
        pdfs = sorted(base.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pdfs:
            return pdfs[0]
        raise FileNotFoundError(f"No PDF files found in: {base}")

    p = pathlib.Path(arg_path if arg_path else DEFAULT_DIR / PREFERRED_FILENAME)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    return p

# -------- PDF -> lines --------
def extract_lines(pdf_path: pathlib.Path) -> List[Tuple[int, str]]:
    lines: List[Tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            for raw in text.splitlines():
                line = raw.strip()
                if line:
                    lines.append((pageno, line))
    return lines

# -------- Parse Q/A --------
def group_qas(lines: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    """
    Collect multi-line Question text until 'Answer:' appears.
    Then collect multi-line Answer text until the next 'Question'.
    """
    qas: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    phase: Optional[str] = None

    for _, line in lines:
        m_q = Q_HEADER_RE.match(line)

        # New question
        if m_q:
            if cur:
                finalize(cur); qas.append(cur)
            first_q = (m_q.group(2) or "").strip()
            cur = {
                "question_lines": [first_q] if first_q else [],
                "answer_lines": [],
            }
            phase = "question"
            continue

        if not cur:
            continue

        # Answer label (inline or on its own)
        m_inline = ANS_INLINE_RE.match(line)
        if m_inline:
            phase = "answer"
            rem = m_inline.group(1).strip()
            if rem:
                cur["answer_lines"].append(rem)
            continue

        if ANS_HEAD_RE.match(line):
            phase = "answer"
            continue

        # Regular content
        if phase == "question":
            cur["question_lines"].append(line)
        else:
            cur["answer_lines"].append(line)

    if cur:
        finalize(cur); qas.append(cur)
    return qas

def finalize(cur: Dict[str, Any]) -> None:
    cur["question"] = " ".join(s.strip() for s in cur.get("question_lines", [])).strip()
    cur["answer"]   = "\n".join(cur["answer_lines"]).strip()

# -------- SQLite --------
def init_db(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS qas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer   TEXT NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_qas_question ON qas(question)")
    return con

def insert_qas(con: sqlite3.Connection, qas: List[Dict[str, Any]]) -> None:
    con.executemany(
        "INSERT INTO qas (question, answer) VALUES (?, ?)",
        [(qa["question"], qa["answer"]) for qa in qas]
    )
    con.commit()

# -------- CLI --------
def main() -> None:
    arg_path = sys.argv[1] if len(sys.argv) >= 2 else None
    pdf_path = resolve_pdf_path(arg_path)
    db_path  = sys.argv[2] if len(sys.argv) >= 3 else "qa.db"

    print(f"Using PDF: {pdf_path}")
    lines = extract_lines(pdf_path)
    qas = group_qas(lines)
    if not qas:
        print("No questions found. Adjust regex or extraction tolerances (x/y_tolerance).")
        sys.exit(2)

    con = init_db(db_path)
    insert_qas(con, qas)
    print(f"Inserted {len(qas)} Q/A into {db_path}")

if __name__ == "__main__":
    main()
