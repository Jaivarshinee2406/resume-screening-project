import pdfplumber
import docx


def extract_text_from_pdf(file) -> str:
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file) -> str:
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)


def extract_text_from_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def extract_resume_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
