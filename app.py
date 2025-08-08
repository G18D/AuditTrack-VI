# app.py
import streamlit as st
import tempfile
from audit_pipeline import run_audit

st.set_page_config(page_title="AuditTrack VI", layout="centered")
st.title("📄 AuditTrack VI - AI Audit Checker")

uploaded_file = st.file_uploader("Upload an audit PDF", type=["pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.write("Processing document...")
    try:
        output_pdf = run_audit(tmp_path)
        with open(output_pdf, "rb") as f:
            st.download_button("⬇ Download Audit Report", f, file_name=output_pdf.split("/")[-1], mime="application/pdf")
        st.success("Audit completed successfully.")
    except Exception as e:
        st.error(f"Error: {e}")
