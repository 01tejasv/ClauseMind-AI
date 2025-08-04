import streamlit as st
import fitz  # PyMuPDF
import docx
import email
from bs4 import BeautifulSoup

st.set_page_config(page_title="ClauseMind AI 🔍", layout="centered")

st.title("ClauseMind AI 🔍")
st.markdown("Welcome to **ClauseMind AI** – An LLM-Powered Document Query & Decision Engine.")

st.header("Upload Your Document")
uploaded_file = st.file_uploader(
    "Choose a PDF, DOCX, or email file", 
    type=["pdf", "docx", "eml"], 
    help="Limit 200MB per file • PDF, DOCX, EML"
)

def parse_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def parse_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_eml(file):
    msg = email.message_from_bytes(file.read())
    body = ""
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            body += part.get_payload(decode=True).decode()
        elif part.get_content_type() == "text/html":
            html = part.get_payload(decode=True).decode()
            soup = BeautifulSoup(html, "html.parser")
            body += soup.get_text()
    return body

if uploaded_file:
    st.success("File uploaded successfully!")
    st.json({
        "filename": uploaded_file.name,
        "type": uploaded_file.type,
        "size": uploaded_file.size
    })

    try:
        if uploaded_file.type == "application/pdf":
            content = parse_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = parse_docx(uploaded_file)
        elif uploaded_file.type == "message/rfc822":
            content = parse_eml(uploaded_file)
        else:
            st.error("Unsupported file type.")
            content = None

        if content:
            st.subheader("Parsed Text Content:")
            st.text_area("Document Text", content, height=300)

    except Exception as e:
        st.error(f"Failed to parse the document: {e}")
