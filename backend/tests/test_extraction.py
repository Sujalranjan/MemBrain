import pytest

from app.services.extraction import ExtractionError, UnsupportedDocument, detect_format, extract_text, normalize
from tests.documents import make_docx, make_pdf


def test_pdf_text_is_extracted_page_by_page():
    text = extract_text("notes.pdf", make_pdf("Postmortem (draft) for the outage", "Action items follow"))
    assert text == "Postmortem (draft) for the outage\n\nAction items follow"


def test_docx_keeps_tables_in_document_order():
    data = make_docx(["Release owners"], table=[["Service", "Owner"], ["billing", "Priya"]])
    assert extract_text("plan.docx", data) == "Release owners\n\nService | Owner\n\nbilling | Priya\n\nAfter the table."


def test_text_files_fall_back_from_utf8():
    assert extract_text("a.md", "# Café menu".encode()) == "# Café menu"
    assert extract_text("a.txt", "Café menu".encode("cp1252")) == "Café menu"


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedDocument):
        detect_format("slides.pptx", b"PK\x03\x04")


@pytest.mark.parametrize(
    "filename, data",
    [("report.pdf", b"just text"), ("legacy.docx", b"\xd0\xcf\x11\xe0 old binary doc")],
)
def test_extension_must_match_file_signature(filename, data):
    with pytest.raises(ExtractionError):
        detect_format(filename, data)


def test_pdf_without_text_layer_explains_ocr():
    with pytest.raises(ExtractionError, match="OCR"):
        extract_text("scan.pdf", make_pdf(""))


def test_corrupt_pdf_raises_extraction_error():
    with pytest.raises(ExtractionError):
        extract_text("broken.pdf", b"%PDF-1.4\nthis is not really a pdf")


def test_normalize_collapses_layout_whitespace():
    assert normalize("a\t\tb  \r\n\r\n\r\n\r\n  c\x00") == "a b\n\nc"
