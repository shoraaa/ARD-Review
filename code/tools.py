"""Shared file access and lazy calibration retrieval for SDK and CLI backends.

The calibration corpus is a directory of human-review markdown files (one per
paper, each carrying a `- Scores:` line). It is NOT shipped with this release —
build it with `code/download_calibration_corpus.py`, which writes
`datasets/deepreview_13k_calibration/` from the public DeepReview-13K dataset.
The embedding matrix and the per-file avg-score index are downloaded from
HuggingFace on first use (see paths.ensure_hf_file).
"""
import os
import pickle
import re
import sys
from pathlib import Path

import dotenv
dotenv.load_dotenv()

from paths import DATASETS_DIR, ensure_hf_file

CALIBRATION_REVIEW_DIR = str((DATASETS_DIR / "deepreview_13k_calibration").resolve())
API_BASE_URL = os.environ.get("REVIEW_API_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("REVIEW_API_KEY", "")
ALLOWED_PATHS = [CALIBRATION_REVIEW_DIR]

database = {}
filenames: list[str] = []
vectors = None
score_index: dict[str, float] = {}
embed_client = None


def embedding_config() -> tuple[str, str, str]:
    """Embedding credentials are independent of reviewer API/CLI authentication."""
    base_url = os.environ.get("REVIEW_EMBEDDING_BASE_URL") or API_BASE_URL
    key = os.environ.get("REVIEW_EMBEDDING_API_KEY", "")
    # Never send a reviewer key to a different provider as an implicit fallback.
    if not key and base_url.rstrip("/") == API_BASE_URL.rstrip("/"):
        key = API_KEY
    if not key:
        raise ValueError(
            "Vector calibration requires REVIEW_EMBEDDING_API_KEY "
            "(or REVIEW_API_KEY when both endpoints match); use --calibration_mode bm25 or --no_cal otherwise."
        )
    default_model = ("gemini-embedding-001" if
                     base_url.rstrip("/") == "https://generativelanguage.googleapis.com/v1beta/openai"
                     else "google/gemini-embedding-001")
    model = os.environ.get("REVIEW_EMBEDDING_MODEL") or default_model
    return base_url, key, model


def initialize_calibration(mode: str = "vector"):
    """Load only the requested index. BM25 needs neither credentials nor embeddings."""
    global filenames, vectors, score_index, embed_client
    if mode not in {"bm25", "vector"}:
        raise ValueError("Calibration mode must be bm25 or vector")
    if not os.path.isdir(CALIBRATION_REVIEW_DIR):
        raise FileNotFoundError(
            f"Calibration corpus not found at {CALIBRATION_REVIEW_DIR}. "
            "Run `python code/download_calibration_corpus.py` first, or pass --no_cal."
        )
    if mode == "bm25" and not database:
        from rank_bm25 import BM25Okapi
        files, documents = [], []
        for path in sorted(Path(CALIBRATION_REVIEW_DIR).rglob("*")):
            if path.suffix not in {".md", ".txt"} or not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            scores = re.search(r"^- Scores:\s*(.+)$", content, re.MULTILINE)
            avg = re.search(r"^- Avg Score:\s*([\d.]+)", content, re.MULTILINE)
            if scores:
                values = [float(value.strip()) for value in scores[1].split(",")]
                score_index[path.name] = sum(values) / len(values)
            elif avg:
                score_index[path.name] = float(avg[1])
            else:
                continue
            tokens = re.findall(r"\w+", content.lower())
            if tokens:
                files.append(str(path))
                documents.append(tokens)
        if not documents:
            raise RuntimeError(f"No scored calibration reviews found under {CALIBRATION_REVIEW_DIR}")
        database[CALIBRATION_REVIEW_DIR] = {"files": files, "bm25": BM25Okapi(documents)}
    elif mode == "vector" and vectors is None:
        base_url, key, _ = embedding_config()
        import numpy as np
        from openai import OpenAI
        with open(ensure_hf_file("human_reviews_embeddings_deepreview.pkl"), "rb") as f:
            db = pickle.load(f)
        filenames = list(db)
        vectors = np.array(list(db.values()))
        with open(ensure_hf_file("human_review_score_index_deepreview.pkl"), "rb") as f:
            score_index = pickle.load(f)
        embed_client = OpenAI(base_url=base_url, api_key=key)


def path_allowed(path: str) -> bool:
    resolved = Path(path).resolve()
    return any(resolved == Path(root).resolve() or
               resolved.is_relative_to(Path(root).resolve()) for root in ALLOWED_PATHS)


# ── Tools ────────────────────────────────────────────────────────────
def allow_path(path: str):
    """Extend ALLOWED_PATHS at runtime (e.g. to grant the merger access to the paper_dir)."""
    resolved = str(Path(path).resolve())
    if resolved not in ALLOWED_PATHS:
        ALLOWED_PATHS.append(resolved)


def read_file(abs_path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read lines from a file. Returns lines numbered start_line to end_line (inclusive, 1-based).
    By default (start_line=1, end_line=0), reads the entire file. Only pass start_line/end_line
    when you specifically need a partial slice; the default is to read the whole file.
    If access is denied, only re-check whether the requested path is misspelled; do not explore
    directories or try nearby paths."""
    resolved = os.path.abspath(abs_path)
    print(f"  [read_file] Request to read '{resolved}' lines {start_line} to {end_line if end_line > 0 else 'EOF'}", file=sys.stderr)
    if not path_allowed(resolved):
        print(f"  [read_file] BLOCKED: '{resolved}' is not under any allowed directory.", file=sys.stderr)
        return f"ERROR: Access denied. Path '{resolved}' is not under any allowed directory. The agent may access only the specific allowed paths. Do not explore directories or nearby paths; only double-check whether the requested path was misspelled."
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    selected = lines[max(0, start_line - 1):end_line if end_line > 0 else len(lines)]
    return "".join(f"{start_line + i}: {line}" for i, line in enumerate(selected))


def grep_file(pattern: str, abs_path: str) -> str:
    """Search a single file for a pattern. Returns matching lines with line numbers.
    If access is denied, only re-check whether the requested path is misspelled; do not explore
    directories or try nearby paths."""
    import re
    resolved = os.path.abspath(abs_path)
    print(f"  [grep_file] Request to grep for pattern '{pattern}' in '{resolved}'", file=sys.stderr)
    if not path_allowed(resolved):
        print(f"  [grep_file] BLOCKED: '{resolved}' is not under any allowed directory.", file=sys.stderr)
        return f"ERROR: Access denied. Path '{resolved}' is not under any allowed directory. The agent may access only the specific allowed paths. Do not explore directories or nearby paths; only double-check whether the requested path was misspelled."
    if not os.path.isfile(resolved):
        return f"ERROR: '{resolved}' is not a file. Do not explore directories or nearby paths; only double-check whether the requested path was misspelled."
    matches = []
    with open(resolved, "r", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            if re.search(pattern, line):
                matches.append(f"{i}: {line.rstrip()}")
    return "\n".join(matches) if matches else "No matches found."


def search_file_impl(query: str, n: int, mode: str, low_score: float = -1.0, high_score: float = 11.0) -> str:
    """Search the calibration corpus, filtered by the reviewer avg-score range.

    Args:
        query: search query.
        n: number of top results.
        mode: 'vector' for semantic similarity, 'bm25' for keyword matching.
        low_score: include only papers with avg score > low_score (default -1.0).
        high_score: include only papers with avg score < high_score (default 11.0).

    Filtering is applied FIRST by score range, THEN ranking (BM25/vector) runs
    over the filtered subset. Use this to anchor calibration to a specific
    score band (e.g. low_score=7, high_score=10 for strong papers).
    """
    print(f"  [search_file] query='{query}' mode='{mode}' n={n} score=({low_score}, {high_score})", file=sys.stderr)
    initialize_calibration(mode)
    if n < 1:
        raise ValueError("n must be positive")
    if mode == "bm25":
        bm25 = list(database.values())[0]["bm25"]
        files = list(database.values())[0]["files"]
        allowed_idx = [
            i for i, p in enumerate(files)
            if low_score < score_index[os.path.basename(p)] < high_score
        ]
        if not allowed_idx:
            return "No files in that score range."
        tokenized_query = re.findall(r"\w+", query.lower())
        doc_scores = bm25.get_scores(tokenized_query)
        allowed_sorted = sorted(allowed_idx, key=lambda i: doc_scores[i], reverse=True)[:n]
        results = []
        for idx in allowed_sorted:
            file_path = os.path.abspath(files[idx])
            rel = doc_scores[idx]
            avg = score_index[os.path.basename(file_path)]
            with open(file_path, 'r', errors='replace') as f:
                content = f.read()
            results.append(f"{file_path}\navg_score: {avg:.2f}  bm25: {rel:.2f}\n first 1000 chars:\n{content[:1000]}\n")
        return "\n---\n".join(results) if results else "No relevant files found."
    elif mode == "vector":
        import numpy as np
        allowed_mask = np.array([
            low_score < score_index[fn] < high_score for fn in filenames
        ])
        if not allowed_mask.any():
            return "No files in that score range."
        query_embedding = embed_client.embeddings.create(
            model=embedding_config()[2],
            input=query,
            encoding_format="float"
        )
        query_vector = np.array(query_embedding.data[0].embedding)
        if query_vector.ndim != 1 or query_vector.shape[0] != vectors.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: query shape {query_vector.shape}, "
                f"corpus dimension {vectors.shape[1]}. Use the same embedding model "
                "and output dimensions as the corpus; the published vectors use gemini-embedding-001."
            )
        if not np.isfinite(query_vector).all():
            raise ValueError("Embedding provider returned non-finite values.")
        similarities = vectors @ query_vector.T
        masked = np.where(allowed_mask, similarities, -np.inf)
        top_indices = masked.argsort()[-n:][::-1]
        results = []
        for idx in top_indices:
            if not np.isfinite(masked[idx]):
                break
            fn = filenames[idx]
            file_path = os.path.abspath(os.path.join(CALIBRATION_REVIEW_DIR, fn))
            rel = similarities[idx]
            avg = score_index[fn]
            with open(file_path, "r", errors="replace") as file_handle:
                content = file_handle.read()
            results.append(f"{file_path}\navg_score: {avg:.2f}  sim: {rel:.2f}\n first 1000 chars:\n{content[:1000]}\n")
        return "\n---\n".join(results) if results else "No relevant files found."
    else:
        return "ERROR: Invalid search mode. Use 'bm25' or 'vector'."
