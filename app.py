import streamlit as st

st.set_page_config(page_title="ClauseMind AI", layout="wide")

st.title("ClauseMind AI 🔍")
st.write("Welcome to ClauseMind AI – An LLM-Powered Document Query & Decision Engine.")

st.header("Upload Your Document")
uploaded_file = st.file_uploader("Choose a PDF, DOCX, or email file", type=["pdf", "docx", "eml"])

if uploaded_file:
    st.success(f"File `{uploaded_file.name}` uploaded successfully!")
    # Placeholder logic
    st.info("Document processing and semantic query features will be integrated here.")
else:
    st.warning("Please upload a document to begin.")
