import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from netplan.rag import HybridRetriever

reportlab = pytest.importorskip("reportlab")
from reportlab.pdfgen import canvas  # noqa: E402


def make_pdf(path, pages):
    c = canvas.Canvas(str(path))
    for text in pages:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()


def test_ingest_builds_and_reuses_index(tmp_path):
    docs, idx = tmp_path / "specs", tmp_path / "index"
    docs.mkdir()
    make_pdf(
        docs / "38331.pdf",
        ["Timer T310 starts on physical layer problems.", "N310 counts out-of-sync.", "SSB periodicity."],
    )
    emb = DeterministicFakeEmbedding(size=16)

    r1 = HybridRetriever.from_folder(docs, emb, idx)
    assert len(r1.chunks) == 3
    assert (idx / "meta.json").exists()

    hit = r1.search("T310", k=1)[0]
    assert hit.metadata == {"source": "38331.pdf", "page": 1, "chunk_id": 0}

    r2 = HybridRetriever.from_folder(docs, emb, idx)  # loads from disk, same chunks
    assert [c.page_content for c in r2.chunks] == [c.page_content for c in r1.chunks]
