import os
import time
import requests
import openai
from mistralai import Mistral
from pathlib import Path
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
import json
from mistralai.models import OCRResponse
import base64


# ------ AZURE ------
AZURE_ENDPOINT = "https://<your-region>.api.cognitive.microsoft.com"
AZURE_API_KEY = "<your_azure_api_key>"
READ_API_URL = f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze"


# ------ OPENAI -------
OPENAI_API_KEY = "<your_openai_api_key>"
openai.api_key = OPENAI_API_KEY


# ------ MISTRAL -------
MISTRAL_API_KEY = "<your_mistral_api_key>"
client = Mistral(api_key=MISTRAL_API_KEY)

# --- Functions ---

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

# This does the ocr with azure AI
def azure_ocr(pdf_path):
    """
    Sends a PDF file to Azure OCR (Read API) and returns the extracted text.
    """
    with open(pdf_path, "rb") as f:
        file_data = f.read()

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
        "Content-Type": "application/pdf"
    }
    
    # Send the PDF for analysis
    response = requests.post(READ_API_URL, headers=headers, data=file_data)
    if response.status_code != 202:
        print(f"Error processing {pdf_path}: {response.status_code}, {response.text}")
        return None

    # Retrieve the URL to poll for the results
    operation_url = response.headers.get("Operation-Location")
    if not operation_url:
        print(f"Operation-Location header missing for {pdf_path}")
        return None

    # Poll the operation URL until the analysis is complete
    while True:
        result_response = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": AZURE_API_KEY})
        result_json = result_response.json()
        status = result_json.get("status")
        if status == "succeeded":
            break
        elif status == "failed":
            print(f"OCR processing failed for {pdf_path}")
            return None
        time.sleep(1)  # Wait a bit before polling again

    # Extract recognized text from the analysis result.
    text = ""
    read_results = result_json.get("analyzeResult", {}).get("readResults", [])
    for page in read_results:
        for line in page.get("lines", []):
            text += line.get("text", "") + "\n"
    return text

def summarize_text(text):
    """
    Sends the provided text to the ChatGPT API to get a summary.
    """
    system_prompt = "You are an assistant that analyzes the contents of a text chunk \
                     and provides a short summary"
    user_prompt = f"Please provide a concise summary of the following content:\n\n{text}"
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
    )
    summary = response.choices[0].message['content']
    return summary


"""
The functions below are used to convert multimodal PDF's to structured JSON.
The functions have not been tested on videos, pure images, documents which
do not fit in the context length of the mistral model, and documents with multiple images.
"""
def mistral_ocr(pdf_path):
    # load file
    pdf_file = Path("/content/wjc_0162.tif.pdf")
    assert pdf_file.is_file()
    uploaded_file = client.files.upload(
    file={
        "file_name": pdf_file.stem,
        "content": pdf_file.read_bytes(),
    },
    purpose="ocr",
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
    # process pdf
    pdf_response = client.ocr.process(document=DocumentURLChunk(document_url=signed_url.url), model="mistral-ocr-latest", include_image_base64=True)
    pdf_text = pdf_response.pages[0].markdown
    
    # extract file which image was written to
    img_path_name = extract_images(pdf_response)[0]
    # image ocr

    # Verify image exists
    image_file = Path(img_path_name)
    assert image_file.is_file()

    # Encode image as base64 for API
    encoded = base64.b64encode(image_file.read_bytes()).decode()
    base64_data_url = f"data:image/jpeg;base64,{encoded}"

    # Process image with OCR
    image_response = client.ocr.process(
        document=ImageURLChunk(image_url=base64_data_url),
        model="mistral-ocr-latest"
    )

    # Convert response to JSON
    # response_dict = json.loads(image_response.model_dump_json())
    # json_string = json.dumps(response_dict, indent=4)
    # print(json_string)

    # Combine text from image and markdown and extract JSON metadata
    image_ocr_markdown = image_response.pages[0].markdown

    # Get structured response from model
    chat_response = client.chat.complete(
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": (
                    f"This is a pdf's OCR in markdown:\n\n{image_ocr_markdown + pdf_text}\n.\n"
                    "Convert this into a sensible structured json response containing full_text, doc_id,  Title, Language, Subject, Format, Genre, Administration, People and Organizations, Time Span, Date, Summary"
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    # Parse and return JSON response
    response_dict = json.loads(chat_response.choices[0].message.content)
    # with open(pdf_path.replace(".pdf",".json"), 'w', encoding='utf-8') as f:
    #     json.dump(response_dict, f, ensure_ascii=False, indent=4)
    return json.dumps(response_dict, indent=4)


def extract_images(ocr_response: OCRResponse) -> str:
  images = []
  for page in ocr_response.pages:
    # image_data = {}
    for img in page.images:
      # image_data[img.id] = img.image_base64
      # print(base64.b64decode(img.image_base64))
      header, encoded = img.image_base64.split(",", 1)
      with open(img.id, "wb") as f:
        f.write(base64.b64decode(encoded))
      images.append(img.id)
  return images

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
        # images_dict[img_name] = im
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
  markdowns: list[str] = []
  for page in ocr_response.pages:
    image_data = {}
    for img in page.images:
      image_data[img.id] = img.image_base64
    markdowns.append(replace_images_in_markdown(page.markdown, image_data))

"""
The functions above are used to convert multimodal PDF's to structured JSON.
The functions have not been tested on videos, pure images, and documents which
do not fit in the context length of the mistral model.
"""


def main():
    input_folder = "/sample_data"    # Folder containing PDF files
    output_folder = "/output" # Folder to save OCR texts and summaries
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing {pdf_path} ...")
            
            # Get OCR text from Azure
            # ocr_text = azure_ocr(pdf_path)
            ocr_text = mistral_ocr(pdf_path)
            if not ocr_text:
                print(f"Skipping {filename} due to OCR error.")
                continue

            # Save the OCR text
            base_filename = os.path.splitext(filename)[0]
            ocr_filename = os.path.join(output_folder, base_filename + "_ocr.txt")
            with open(ocr_filename, "w", encoding="utf-8") as f:
                f.write(ocr_text)
            print(f"OCR text saved to {ocr_filename}")

            # Summarize the OCR text using ChatGPT
            summary = summarize_text(ocr_text)
            summary_filename = os.path.join(output_folder, base_filename + "_summary.txt")
            with open(summary_filename, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Summary saved to {summary_filename}\n")


if __name__ == "__main__":
    main()
