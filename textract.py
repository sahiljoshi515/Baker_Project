from mistralai import Mistral
from pdf2image import convert_from_path
import boto3
import io
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  # By default looks for .env file in current directory

# ------ MISTRAL -------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

# This computes OCR with the help of Amazon Textract
def textract_ocr(pdf_path):
    # Convert PDF to images (300 DPI for better accuracy)
    images = convert_from_path(pdf_path, dpi=250)

    textract = boto3.client('textract')
    all_pages_text = []

    for i, img in enumerate(images):
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)  # Move pointer to start of file
        # Process each image with Textract
        response = textract.detect_document_text(Document={'Bytes': img_bytes.getvalue()})

        # Extract text from response
        page_text = []
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                page_text.append(item["Text"])
        
        all_pages_text.append("\n".join(page_text))

        print(f"OCR Completed for Page {i+1}")  # Debugging statement

    # Combine text from all pages
    final_text = "\n\n".join(all_pages_text)

    return final_text