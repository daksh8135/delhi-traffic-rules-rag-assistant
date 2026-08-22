# api/app.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
from generator import Generator

st.set_page_config(
    page_title="Delhi Traffic Rules Assistant",
    page_icon="🚦",
    layout="centered"
)

# --- Header ---
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1>🚦 Delhi Traffic Rules Assistant</h1>
        <p style="color: gray; font-size: 16px;">
            A hybrid RAG assistant for Delhi Motor Vehicle laws — built with FAISS + BM25 + Groq.<br>
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

# --- Example questions as clickable chips ---
st.markdown("**Try asking:**")
example_cols = st.columns(3)
examples = [
    "Fine for not wearing a helmet?",
    "Penalty for drunk driving?",
    "सड़क पार करते समय सावधानियां?"
]

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

for col, example in zip(example_cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state.query_input = example

# --- Input box ---
query = st.text_input(
    "Your question:",
    value=st.session_state.query_input,
    placeholder="e.g. What is the fine for not wearing a helmet?"
)

ask_clicked = st.button("Ask", type="primary", use_container_width=True)

# --- Answer ---
if query and ask_clicked:
    with st.spinner("Searching Delhi traffic law documents..."):
        try:
            result = generator.ask(query)
            st.divider()
            st.markdown("### 📋 Answer")
            st.success(result["answer"])
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# --- Footer ---
st.divider()
st.caption(
    "Built with LangChain, FAISS, BM25 hybrid retrieval, and Groq (Llama/GPT-OSS). "
    "For general awareness only — not a substitute for legal advice."
)