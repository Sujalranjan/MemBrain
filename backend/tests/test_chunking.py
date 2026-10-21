from app.services.chunking import chunk_text


def test_empty_text_has_no_chunks():
    assert chunk_text("   ") == []


def test_short_text_is_single_chunk():
    assert chunk_text("hello world", size=100) == ["hello world"]


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_oversized_paragraph_without_breaks_is_split():
    chunks = chunk_text("x" * 2500, size=1000, overlap=0)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_consecutive_chunks_overlap():
    text = "\n\n".join(f"Sentence number {i} talks about topic {i}." for i in range(40))
    chunks = chunk_text(text, size=200, overlap=60)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert any(word in nxt for word in prev.split()[-3:])


def test_all_content_is_preserved():
    paras = [f"unique{i} " + "filler " * 30 for i in range(15)]
    joined = " ".join(chunk_text("\n\n".join(paras), size=250, overlap=40))
    assert all(f"unique{i}" in joined for i in range(15))
