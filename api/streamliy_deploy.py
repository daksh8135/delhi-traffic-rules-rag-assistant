# streamlit_deploy.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
from generator import Generator

st.set_page_config(page_title="Delhi Traffic Rules Assistant", page_icon="🚦")

st.title("🚦 Delhi Traffic Rules Assistant")
st.write("Ask questions about Delhi traffic rules, fines, and procedures. Supports English and Hindi.")

@st.cache_resource
def load_generator():
    return Generator()

generator = load_generator()

query = st.text_input("Your question:", placeholder="e.g. What is the fine for not wearing a helmet?")

if query:
    with st.spinner("Thinking..."):
        try:
            result = generator.ask(query)
            st.markdown("### Answer")
            st.text(result["answer"])
        except Exception as e:
            st.error(f"Something went wrong: {e}")