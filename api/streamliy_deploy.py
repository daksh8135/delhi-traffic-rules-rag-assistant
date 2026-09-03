# streamlit_deploy.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
import PyPDF2
from generator import Generator

st.set_page_config(page_title="Delhi Traffic Rules Assistant", page_icon="🚦", layout="centered")

# --- Header ---
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1>🚦 Delhi Traffic Rules Assistant</h1>
        <p style="color: gray; font-size: 16px;">
            A hybrid RAG assistant for Delhi Motor Vehicle laws.<br>
            Supports English and Hindi.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


@st.cache_resource
def load_generator():
    return Generator()

generator = load_generator()


def extract_uploaded_pdf_text(uploaded_file) -> str:
    """Extracts text from a user-uploaded challan PDF."""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


# --- Example question chips ---
st.markdown("**Try asking:**")
example_cols = st.columns(3)
examples = [
    "Fine for not wearing a helmet?",
    "Penalty for drunk driving?",
    "सड़क पार करते समय क्या सावधानियां बरतनी चाहिए?"
]

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

for col, example in zip(example_cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state.query_input = example

# --- Text question input ---
query = st.text_input(
    "Your question:",
    value=st.session_state.query_input,
    placeholder="e.g. What is the fine for not wearing a helmet?"
)

ask_clicked = st.button("Ask", type="primary", use_container_width=True)

if query and ask_clicked:
    with st.spinner("Searching Delhi traffic law documents..."):
        try:
            result = generator.ask(query)
            st.divider()
            st.markdown("### 📋 Answer")
            st.success(result["answer"])
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# --- Challan upload section ---
st.divider()
st.markdown("### 📄 Upload your challan")
st.info("⚠️ Please upload a **PDF file only**. Scanned images are not supported.")

uploaded_file = st.file_uploader("Upload challan (PDF only)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Reading your challan..."):
        challan_text = extract_uploaded_pdf_text(uploaded_file)

    if not challan_text or len(challan_text.strip()) < 20:
        st.error(
            "Couldn't extract readable text from this PDF. "
            "It looks like a scanned image rather than a text-based PDF."
        )
    else:
        with st.spinner("Analyzing against Delhi traffic law..."):
            question = (
                f"A user uploaded the following traffic challan document. "
                f"Explain why this challan was likely issued, which rule or section "
                f"it relates to, and the associated fine.\n\n"
                f"Challan content:\n{challan_text[:3000]}"
            )
            try:
                result = generator.ask(question)
                st.markdown("### Explanation")
                st.success(result["answer"])
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# --- Footer ---
st.divider()
st.caption(
    "Built with LangChain, FAISS, BM25 hybrid retrieval, and Groq (Llama/GPT-OSS). "
    "For general awareness only — not a substitute for legal advice."
)