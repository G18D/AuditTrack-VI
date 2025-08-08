# utils.py
import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import pytesseract
from PIL import Image
import cv2
import numpy as np
import openai
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from fpdf import FPDF

# ---------- OCR & Text Extraction ----------
def extract_text_from_pdf(pdf_path):
    text = ""
    used_ocr = False
    try:
        # Try direct text extraction
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t
    except:
        pass

    # OCR fallback
    if not text.strip():
        used_ocr = True
        images = convert_from_path(pdf_path)
        for img in images:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            img_pil = Image.fromarray(thresh)
            text += pytesseract.image_to_string(img_pil, config="--psm 6 -l eng") + "\n"

    return text, used_ocr

# ---------- Field Checks ----------
def check_fields_extracted(text):
    status = {}
    status['Date'] = '✅ Complete' if any(y in text for y in ['2022', '2023', '2024']) else '❌ Missing'
    status['Vendor Name'] = '✅ Complete' if any(x in text for x in ['LLC', 'Inc', 'Corp']) else '❌ Missing'
    status['Total Amount'] = '✅ Complete' if '$' in text or 'USD' in text else '❌ Missing'
    status['Account Code'] = '✅ Complete' if 'Account Code:' in text else '❌ Missing'
    status['Department or Project'] = '✅ Complete' if 'Department or Project:' in text else '❌ Missing'
    status['Signature present'] = '⚠ Problem' if 'signature' not in text.lower() else '✅ Complete'
    return status

# ---------- GPT Field Extraction ----------
def extract_fields_with_gpt(doc_text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"""
    You are a compliance audit assistant. Extract the following:
    - Vendor Name
    - Date
    - Total Amount
    - Account Code
    - Department or Project
    - Signature present (Yes/No)
    Then list any that are missing or incorrect.

    Document Text:
    \"\"\"{doc_text}\"\"\"
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling GPT: {e}"

# ---------- Compliance Check ----------
def perform_compliance_analysis(doc_text):
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name="rules-knowledge-base",
            embedding=embeddings
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )
        query = f"Analyze this document for compliance issues:\n\n{doc_text[:4000]}"
        return qa_chain.invoke({"query": query})
    except Exception as e:
        return {"result": f"Error in compliance analysis: {e}", "source_documents": []}

# ---------- Report Building ----------
def synthesize_report(file_name, field_status, compliance_response, gpt_field_results):
    return {
        "Document_Name": file_name,
        "Extracted_Fields_Status_Basic": field_status,
        "Extracted_Fields_Results_GPT": gpt_field_results,
        "Compliance_Assessment": compliance_response.get("result", "N/A"),
        "Supporting_Regulations": compliance_response.get("source_documents", [])
    }

def export_audit_to_pdf(report_data, output_filename="audit_report.pdf", document_name="Uploaded File"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Audit Results Report", ln=True, align='C')
    pdf.cell(0, 10, f"Document: {document_name}", ln=True)
    pdf.ln(10)
    for field, status in report_data["Extracted_Fields_Status_Basic"].items():
        pdf.cell(80, 8, field, 1)
        pdf.cell(80, 8, status, 1, ln=1)
    pdf.output(output_filename)
    return output_filename
