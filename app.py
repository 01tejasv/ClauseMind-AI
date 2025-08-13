import streamlit as st
import fitz  # PyMuPDF
import docx
import email
from bs4 import BeautifulSoup
import chardet
import pandas as pd
from backend import process_and_index

st.set_page_config(page_title="ClauseMind AI 🔍", layout="centered")
st.title("ClauseMind AI 🔍")
st.markdown("Welcome to **ClauseMind AI** – An LLM-Powered Document Query & Decision Engine.")

st.header("Upload Your Document")
uploaded_file = st.file_uploader(
    "Choose a PDF, DOCX, Email, or Text/CSV file", 
    type=["pdf", "docx", "eml", "txt", "csv"],
    help="Limit 200MB per file • PDF, DOCX, EML, TXT, CSV"
)

# -------------------------
# Parsing functions
# -------------------------
def parse_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num, page in enumerate(doc, start=1):
        text += f"\n[Page {page_num}]\n" + page.get_text()
    return text

def parse_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_eml(file):
    msg = email.message_from_bytes(file.read())
    body = ""
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            body += part.get_payload(decode=True).decode(errors="ignore")
        elif part.get_content_type() == "text/html":
            html = part.get_payload(decode=True).decode(errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            body += soup.get_text()
    return body

def parse_txt(file):
    raw = file.read()
    encoding = chardet.detect(raw)["encoding"] or "utf-8"
    return raw.decode(encoding, errors="ignore")

def parse_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)

# -------------------------
# Main app logic
# -------------------------
if uploaded_file:
    st.success("File uploaded successfully!")
    st.json({
        "filename": uploaded_file.name,
        "type": uploaded_file.type,
        "size": uploaded_file.size
    })

    try:
        # Detect type
        if uploaded_file.type == "application/pdf":
            content = parse_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = parse_docx(uploaded_file)
        elif uploaded_file.type == "message/rfc822":
            content = parse_eml(uploaded_file)
        elif uploaded_file.type in ["text/plain", "application/txt"]:
            content = parse_txt(uploaded_file)
        elif uploaded_file.type in ["text/csv", "application/vnd.ms-excel"]:
            content = parse_csv(uploaded_file)
        else:
            st.error("Unsupported file type.")
            content = None

        if content:
            st.subheader("Parsed Text Content:")
            st.text_area("Document Text", content, height=300)

            if st.button("Process & Index Document"):
                with st.spinner("Processing and indexing document..."):
                    result = process_and_index(content, uploaded_file.name)
                    st.success(f"Document indexed successfully with {result['chunks_indexed']} chunks.")

    except Exception as e:
        st.error(f"Failed to parse the document: {e}")
