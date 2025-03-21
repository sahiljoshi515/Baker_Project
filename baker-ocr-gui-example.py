import os
import time
import requests
import openai
import gradio as gr
import tiktoken
from mistralai import Mistral

# ------ AZURE ------
AZURE_ENDPOINT = "https://<your-region>.api.cognitive.microsoft.com"
AZURE_API_KEY = "<your_azure_api_key>"
READ_API_URL = f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze"


# ------ OPENAI -------
OPENAI_API_KEY = "<your-api-key>"
openai.api_key = OPENAI_API_KEY


# ------ MISTRAL -------
MISTRAL_API_KEY = "<your-api-key>"
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

# Function to split text into chunks based on token limits
def split_text_into_chunks(text, max_tokens=1000):
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
        elif ocr_engine == "Azure":
            ocr_text = azure_ocr(pdf_path)
        elif ocr_engine == "Textract":
            ocr_text = textract_ocr(pdf_path)
        else:
            summaries.append(f"Invalid OCR engine selected for {pdf_path}.")
            continue

        if not ocr_text:
            summaries.append(f"OCR failed for {pdf_path}.")
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
