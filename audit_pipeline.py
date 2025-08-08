# audit_pipeline.py
import os
from utils import (
    extract_text_from_pdf, check_fields_extracted,
    extract_fields_with_gpt, perform_compliance_analysis,
    synthesize_report, export_audit_to_pdf
)

def run_audit(file_path):
    text, used_ocr = extract_text_from_pdf(file_path)
    if not text.strip():
        raise ValueError("No text extracted from PDF.")

    field_status = check_fields_extracted(text)
    gpt_results = extract_fields_with_gpt(text)
    compliance_response = perform_compliance_analysis(text)

    report = synthesize_report(
        os.path.basename(file_path),
        field_status,
        compliance_response,
        gpt_results
    )

    output_pdf = f"{os.path.splitext(file_path)[0]}_audit_report.pdf"
    export_audit_to_pdf(report, output_filename=output_pdf, document_name=os.path.basename(file_path))
    return output_pdf
