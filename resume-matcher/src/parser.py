"""
Text extraction and document parser for resumes and job descriptions.
Engineered by M. Abdullah.
Supports TXT, PDF, DOCX, and in-memory file buffers.
"""
import io
from pathlib import Path


def extract_text_from_file(file_path_or_buffer) -> str:
    """Extract raw text from file path or file-like stream."""
    if hasattr(file_path_or_buffer, "read"):
        # File buffer (e.g. Streamlit UploadedFile)
        content = file_path_or_buffer.read()
        if isinstance(content, bytes):
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    import fitz
                    doc = fitz.open(stream=content, filetype="pdf")
                    return "\n".join([page.get_text() for page in doc])
                except Exception:
                    return content.decode("latin-1", errors="ignore")
        return str(content)

    path = Path(file_path_or_buffer)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif suffix == ".pdf":
        try:
            import fitz
            doc = fitz.open(path)
            return "\n".join([page.get_text() for page in doc])
        except ImportError:
            try:
                import pypdf
                reader = pypdf.PdfReader(path)
                return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            except ImportError:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()

    elif suffix == ".docx":
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def extract_text_from_pdf(file_path_or_buffer) -> str:
    return extract_text_from_file(file_path_or_buffer)


def extract_text_from_docx(file_path_or_buffer) -> str:
    return extract_text_from_file(file_path_or_buffer)


def load_job_description(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        # Check fallback directories
        alt_dirs = [
            Path("data/job_descriptions") / path.name,
            Path("data/job_desc") / path.name,
            Path(__file__).resolve().parent.parent / "data" / "job_descriptions" / path.name,
            Path(__file__).resolve().parent.parent / "data" / "job_desc" / path.name,
        ]
        for alt in alt_dirs:
            if alt.exists():
                path = alt
                break
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
