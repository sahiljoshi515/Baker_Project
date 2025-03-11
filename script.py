import os
from mistralai import Mistral
from IPython.display import display, Markdown

api_key = "tfwTOcakQh0Wx0mYYVsouBCnOXbCLxAg"
client = Mistral(api_key=api_key)

uploaded_pdf = client.files.upload(
    file={
        "file_name": "uploaded_file.pdf",
        "content": open("/test/test.pdf", "rb"),
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



all_pages_markdown = []

for page_obj in ocr_response.pages:
    page_index = page_obj.index
    page_text = page_obj.markdown
    # Build a small block of markdown for this page
    md_block = f"## Page {page_index}\n\n{page_text}\n\n"

    # If you want to mention images (but not actually embed them):
    for img in page_obj.images:
        img_id = img.id
        # For actual embedding, you’d need to handle img["image_base64"] if present
        # But if image_base64 is None, you only have location info.
        # Example reference:
        md_block += f"![{img_id}]({img_id})\n\n"
        
    all_pages_markdown.append(md_block)

# Combine everything into one Markdown string
final_markdown = "\n".join(all_pages_markdown)

display(Markdown(final_markdown))