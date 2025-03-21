from mistralai import Mistral

# ------ MISTRAL -------
MISTRAL_API_KEY = "<your-api-key>"
client = Mistral(api_key=MISTRAL_API_KEY)

# This does the ocr with mistral AI
def mistral_ocr(pdf_path):
    uploaded_pdf = client.files.upload(
        file={
            "file_name": "uploaded_file.pdf",
            "content": open(pdf_path, "rb"),
        },
        purpose="ocr"
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )  

    all_pages_text = []
    for page_obj in ocr_response.pages:
        page_index = page_obj.index
        page_text = page_obj.markdown
        text_block = f"Page {page_index}:\n{page_text}\n"
        all_pages_text.append(text_block)
    
    # Combine everything into one plain text string
    final_text = "\n".join(all_pages_text)
    return final_text
