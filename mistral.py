from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from typing import List
import base64
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  # By default looks for .env file in current directory

# ------ MISTRAL -------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

def mistral_ocr(pdf_path) -> str:
    # load file
    uploaded_pdf = client.files.upload(
        file={
            "file_name": "uploaded_file.pdf",
            "content": open(pdf_path, "rb"),
        },
        purpose="ocr"
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    # process pdf
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url), 
        model="mistral-ocr-latest", 
        include_image_base64=True
    )
    pdf_text = pdf_response.pages[0].markdown
    
    # extract file which image was written to
    images = extract_images(pdf_response)
    image_ocr_markdown = ""
    if images:
        image_texts = []
        try:
            for img_path in images:
                # Ensure file exists
                image_file = Path(img_path)
                assert image_file.is_file()

                # Encode image as base64
                encoded = base64.b64encode(image_file.read_bytes()).decode()
                base64_data_url = f"data:image/jpeg;base64,{encoded}"

                # OCR the image
                image_response = client.ocr.process(
                    document=ImageURLChunk(image_url=base64_data_url),
                    model="mistral-ocr-latest"
                )
                image_texts.append(image_response.pages[0].markdown)

            # Combine all image OCR text
            image_ocr_markdown = "\n".join(image_texts)

        except Exception as e:
            print(f"Image OCR failed: {e}")
            # Optional: return PDF-only OCR, or raise error
            image_ocr_markdown = ""  # or: raise

    return image_ocr_markdown + pdf_text

def extract_images(ocr_response: OCRResponse) -> List[str]:
    """
    Extract all images from the OCR response, save them to disk, and return a list of the image file names.

    Args:
        ocr_response (OCRResponse): The OCR response from the Mistral API

    Returns:
        List[str]: A list of the image file names
    """
    images = []
    for page in ocr_response.pages:
        for img in page.images:
            header, encoded = img.image_base64.split(",", 1)
            with open(img.id, "wb") as f:
                f.write(base64.b64decode(encoded))
            images.append(img.id)
    return images