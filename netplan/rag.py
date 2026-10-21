"""Hybrid RAG over 3GPP specifications.

Ingestion: every PDF in a folder, chunked with overlap, page metadata kept for citations.
Retrieval: dense (FAISS + embeddings) and sparse (BM25) results fused with
Reciprocal Rank Fusion. BM25 matters for 3GPP text, where exact tokens like
"T310", "n78" or "SSB" are what the engineer is looking for.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

try:  # FAISS still lives in langchain-community
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover
    FAISS = None  # type: ignore[assignment]

_TOKEN = re.compile(r"[A-Za-z0-9_.\-]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def load_pdfs(folder: str | Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    docs: list[Document] = []
    for pdf in sorted(Path(folder).glob("**/*.pdf")):
        for page in PyPDFLoader(str(pdf)).load():
            page.metadata = {"source": pdf.name, "page": page.metadata.get("page", 0) + 1}
            docs.append(page)
    return docs


def split(docs: list[Document], chunk_size: int = 1000, overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def _fingerprint(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(folder.glob("**/*.pdf")):
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]


class HybridRetriever:
    def __init__(self, chunks: list[Document], embeddings: Embeddings, vector_store=None):
        if not chunks:
            raise ValueError("No chunks to index. Add PDFs to the docs folder.")
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
        self.vs = vector_store or FAISS.from_documents(chunks, embeddings)

    @classmethod
    def from_folder(cls, folder: str | Path, embeddings: Embeddings, index_dir: str | Path) -> HybridRetriever:
        """Build the index once and reuse it while the PDFs do not change."""
        folder, index_dir = Path(folder), Path(index_dir)
        fp = _fingerprint(folder)
        meta = index_dir / "meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("fingerprint") == fp:
            vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            chunks = pickle.loads((index_dir / "chunks.pkl").read_bytes())
            return cls(chunks, embeddings, vs)

        chunks = split(load_pdfs(folder))
        r = cls(chunks, embeddings)
        index_dir.mkdir(parents=True, exist_ok=True)
        r.vs.save_local(str(index_dir))
        (index_dir / "chunks.pkl").write_bytes(pickle.dumps(chunks))
        meta.write_text(json.dumps({"fingerprint": fp, "chunks": len(chunks)}))
        return r

    def search(self, query: str, k: int = 4, fetch_k: int = 20, rrf_k: int = 60) -> list[Document]:
        """Reciprocal Rank Fusion: score = sum over rankers of 1 / (rrf_k + rank)."""
        scores: dict[int, float] = {}

        dense = self.vs.similarity_search(query, k=fetch_k)
        for rank, d in enumerate(dense):
            cid = d.metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        bm = self.bm25.get_scores(tokenize(query))
        top_sparse = sorted(range(len(bm)), key=lambda i: bm[i], reverse=True)[:fetch_k]
        for rank, i in enumerate(top_sparse):
            if bm[i] <= 0:
                break
            cid = self.chunks[i].metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        by_id = {c.metadata["chunk_id"]: c for c in self.chunks}
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [by_id[i] for i in best]


def format_hits(docs: list[Document], max_chars: int = 900) -> str:
    """Context block the agent sees. Each hit carries source and page so it can cite."""
    out = []
    for d in docs:
        text = d.page_content.strip().replace("\n", " ")[:max_chars]
        out.append(f"[source={d.metadata.get('source')} page={d.metadata.get('page')}]\n{text}")
    return "\n\n".join(out) or "No relevant passage found."
