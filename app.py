import streamlit as st
import os

st.set_page_config(page_title="ClauseMind AI", page_icon="🔍")

st.title("ClauseMind AI 🔍")
st.subheader("Welcome to ClauseMind AI – An LLM-Powered Document Query & Decision Engine.")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    .stFileUploader > div:first-child {
        color: #fff;
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("### Upload Your Document")

uploaded_file = st.file_uploader(
    "Choose a PDF, DOCX, or email file",
    type=["pdf", "docx", "eml"],
    help="Limit 200MB per file • PDF, DOCX, EML"
)

if uploaded_file is not None:
    file_details = {
        "filename": uploaded_file.name,
        "type": uploaded_file.type,
        "size": uploaded_file.size
    }
    st.write("File details:", file_details)
    st.success("File uploaded successfully! (But parsing logic is yet to be added)")

else:
    st.info("Please upload a document to begin.")
