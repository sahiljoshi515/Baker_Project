from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from typing import List
import base64
from pathlib import Path
import openai
import json

# ------ OPENAI -------
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# ------ MISTRAL -------
MISTRAL_API_KEY = "tfwTOcakQh0Wx0mYYVsouBCnOXbCLxAg"
client = Mistral(api_key=MISTRAL_API_KEY)

# This does the ocr with mistral AI
def mistral_ocr_legacy(pdf_path):
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

    system_prompt = "You are an assistant that specializes in filling json forms with OCR data. Please fill accurate entries in the fields provided and output a json file only!"
    user_prompt = f"This is a pdf's OCR in markdown:\n\n{image_ocr_markdown + pdf_text}\n.\n" + "Convert this into a sensible structured json response containing full_text, doc_id,  Title, Language, Subject, Format, Genre, Administration, People and Organizations, Time Span, Date, Summary"

    chat_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
    )

    # Parse and return JSON response
    response_dict = json.loads(chat_response.choices[0].message.content, strict=False)

    return json.dumps(response_dict, indent=4)


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