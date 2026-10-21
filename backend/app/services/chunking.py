import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, size: int) -> list[str]:
    """Break an oversized paragraph on sentence boundaries, then hard-wrap on words."""
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > size:
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Pack paragraphs into chunks of at most `size` chars, carrying `overlap` chars of context forward."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            units.extend(_split_long(para, size) if len(para) > size else [para])

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        tail = tail[tail.find(" ") + 1 :] if " " in tail else tail  # start the overlap on a word boundary
        current = f"{tail} {unit}".strip() if tail and len(tail) + 1 + len(unit) <= size else unit
    if current:
        chunks.append(current)
    return chunks
