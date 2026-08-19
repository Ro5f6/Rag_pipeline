"""
pipeline/documents.py
======================
FLOWCHART BLOCK: "Documents"

Responsibility: represent and load your raw knowledge base BEFORE any
processing happens. This is intentionally the dumbest module in the
pipeline -- it does not chunk, embed, or clean text. That separation matters:
in production you'll swap this module constantly (S3 bucket, Confluence API,
a database dump, a CDC stream from Postgres) while every downstream block
(parse_chunk, embed, ...) stays untouched, because they only depend on the
`Document` shape below, not on where it came from.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Dict, Any
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single raw document before chunking."""
    id: str
    text: str
    source: str                      # e.g. file path, URL, DB row id
    metadata: Dict[str, Any] = field(default_factory=dict)


def _stable_id(source: str) -> str:
    """
    Deterministic id derived from the document's location.

    Stability is what lets ingestion be idempotent: because the same file
    always produces the same id, the indexes can recognise a document they
    already hold and replace it rather than storing a second copy. A random
    uuid would make that impossible, and would also break citations, since a
    chunk id returned to a user yesterday must still resolve today.

    Note this identifies *where* a document lives, not what it says -- an
    edited file keeps its id, which is exactly what makes replacement work.
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    """
    Extract text from a PDF, page by page.

    pypdf is imported lazily so the dependency is only needed by corpora that
    actually contain PDFs -- a .txt-only project never pays for it. Extraction
    reads the PDF's text layer; a scanned or image-only PDF yields no text and
    is skipped by the loader (OCR is a separate, heavier problem this does not
    pretend to solve).
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# One reader per extension. Adding a format (.md, .html, .docx) is one entry
# here plus its reader function -- nothing downstream of `Document` changes.
_READERS: Dict[str, Callable[[Path], str]] = {
    ".txt": _read_txt,
    ".pdf": _read_pdf,
}

SUPPORTED_SUFFIXES = tuple(_READERS)


def _iter_corpus_files(directory: str) -> List[Path]:
    """Every supported document file in the directory, sorted for determinism."""
    return sorted(
        p
        for p in Path(directory).glob("*")
        if p.is_file() and p.suffix.lower() in _READERS
    )


def load_documents_from_dir(directory: str) -> List[Document]:
    """
    Load every supported file in a directory into Document objects.

    File type is dispatched by extension (see _READERS): .txt and .pdf today.
    A file that cannot be read, or a PDF with no extractable text layer, is
    skipped with a warning rather than aborting the whole ingest -- one bad
    file in a large corpus should not take the rest down with it.

    In an enterprise iteration you would replace this single function with a
    pluggable `DocumentLoader` interface (LoaderFromS3, LoaderFromConfluence,
    LoaderFromPostgres, ...) all returning List[Document], so orchestrator.py
    never has to know the difference.
    """
    docs: List[Document] = []
    for path in _iter_corpus_files(directory):
        try:
            text = _READERS[path.suffix.lower()](path)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not abort ingest
            logger.warning("Skipping %s: could not read (%s)", path.name, exc)
            continue

        if not text.strip():
            logger.warning("Skipping %s: no extractable text (scanned/image PDF?)", path.name)
            continue

        docs.append(
            Document(
                id=_stable_id(str(path)),
                text=text,
                source=str(path),
                metadata={"filename": path.name},
            )
        )
    return docs


def corpus_fingerprint(directory: str) -> str:
    """
    A cheap identity for "the corpus as it currently exists on disk".

    A persisted index is only valid for the corpus it was built from. Without
    a way to detect that the corpus changed, adding a document leaves the
    service quietly answering from a stale index -- the failure is invisible,
    because everything still returns confident, well-cited answers drawn from
    documents that no longer represent the knowledge base.

    Filename, size, and modification time are hashed rather than file contents:
    reading every document to decide whether to read every document defeats the
    purpose on a large corpus. This misses an edit that preserves both size and
    mtime, which is rare enough to accept and is why `--rebuild` exists.

    Uses the same file enumeration as loading, so a PDF added to the corpus
    invalidates the index exactly as a .txt would.
    """
    parts = []
    for path in _iter_corpus_files(directory):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
