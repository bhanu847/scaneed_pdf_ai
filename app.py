import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
import pandas as pd
import re
import streamlit as st
import tempfile
import os

# ✅ Load model only once (caching improves speed)
@st.cache_resource
def load_model():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# ✅ Regex patterns
patterns = {
    "member_id": r"(?:ID\s*#?|#)(\d{7,10})",
    "rx_number": r"RX[#:\s]*([0-9]{5,})",
    "date_of_fill": r"\b\d{2}/\d{2}/\d{2,4}\b",
    "ndc": r"NDC[:\s]*([0-9]{8,})",
    "day_supply": r"(\d+)\s*(?:day|days)\s*supply",
    "quantity": r"Qty[:\s]*(\d+)",
    "pharmacy_npi": r"PHARMACY\s*NPI[:\s]*([0-9]{10})",
    "prescriber_npi": r"PRESCRIBER\s*NPI[:\s]*([0-9]{10})"
}

def extract_fields(text):
    extracted = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        extracted[key] = matches
    return extracted

def align_rows(data):
    num_rows = max(len(v) for v in data.values() if v)
    rows = []
    for i in range(num_rows):
        row = {}
        for key, values in data.items():
            row[key] = values[i] if i < len(values) else ""
        rows.append(row)
    return rows

def process_pdf(pdf_path):
    # Convert PDF → images
    images = convert_from_path(pdf_path)
    all_text = ""

    for img in images:
        pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
        task_prompt = "<s_docvqa><s_question>extract all text<s_answer>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        all_text += generated_text + "\n"

    extracted_data = extract_fields(all_text)
    rows = align_rows(extracted_data)
    df = pd.DataFrame(rows, columns=[
        "member_id",
        "pharmacy_npi",
        "prescriber_npi",
        "rx_number",
        "date_of_fill",
        "ndc",
        "day_supply",
        "quantity"
    ])
    return df

# ✅ Streamlit UI
st.title("💊 Prescription PDF Extractor (Donut Model)")
st.write("Upload a scanned prescription PDF, and it will extract key fields into Excel.")

uploaded_pdf = st.file_uploader("📂 Upload Prescription PDF", type=["pdf"])

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name

    st.info("⏳ Processing PDF with Donut model...")
    df = process_pdf(tmp_path)

    st.success("✅ Extraction Complete!")
    st.write(df)

    # Save Excel for download
    excel_path = tmp_path.replace(".pdf", ".xlsx")
    df.to_excel(excel_path, index=False)

    with open(excel_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Extracted Excel",
            data=f,
            file_name="rx_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    os.remove(tmp_path)
    if os.path.exists(excel_path):
        os.remove(excel_path)
