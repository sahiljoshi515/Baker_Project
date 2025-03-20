import os
import time
import requests
import openai
import gradio as gr
import tiktoken
from mistralai import Mistral
from pdf2image import convert_from_path
import boto3
import io

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
    Splits a text into chunks of max_tokens size (approximation) to fit within the model's limits.
    """
    encoding = tiktoken.encoding_for_model("gpt-4")  # Adjust for different models if needed
    words = text.split()  # Simple word-based split (can improve with sentence splitting)
    
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        word_tokens = len(encoding.encode(word))  # Count tokens for the word
        if current_tokens + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_text(text, max_tokens=1000):
    """
    Summarizes long text by splitting it into smaller chunks, summarizing each chunk, 
    and then recursively summarizing the combined summary.
    """
    chunks = split_text_into_chunks(text, max_tokens=max_tokens)
    
    summaries = []
    for chunk in chunks:
        system_prompt = "You are an assistant that analyzes a text chunk and provides a short summary."
        user_prompt = f"Please summarize the following content:\n\n{chunk}"

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
        )

        summaries.append(response.choices[0].message.content)

    # If multiple summaries, recursively summarize them
    combined_summary = " ".join(summaries)
    
    if len(summaries) > 1:
        print("Summarizing combined text...")
        return summarize_text(combined_summary, max_tokens=max_tokens)
    
    return combined_summary


def process_pdfs(files, ocr_engine):
    """
    Process multiple PDF files and return their summaries.

    Args:
        files (list): List of PDF files as file objects
        ocr_engine (str): OCR engine to use (Mistral, Azure, or Textract)

    Returns:
        str: Summaries of the PDF files
    """
    if not files:
        return "No files uploaded."
    
    summaries = []
    
    for file in files:
        pdf_path = file.name
        print(f"Processing {pdf_path} with {ocr_engine} OCR...")

        # Perform OCR
        if ocr_engine == "Mistral":
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

        # Summarize text
        summary = summarize_text(ocr_text)
        summaries.append(f"### {os.path.basename(pdf_path)}\n{summary}\n\n")
    return "\n".join(summaries)

def main():
    # Gradio UI with multiple file selection
    with gr.Blocks() as demo:
        gr.Markdown("## OCR & Summarization Tool")
        
        files_input = gr.Files(label="Upload PDFs", type="filepath", file_types=[".pdf"], interactive=True)
        engine_dropdown = gr.Dropdown(
            choices=["Mistral", "Azure", "Textract"],
            label="Select OCR Engine",
            value=None,
            interactive=True,
            info="Select the OCR engine for processing."
        )

        process_button = gr.Button("Process")

        with gr.Accordion("Summaries", open=False): 
            summary_output = gr.Markdown(label="OCR Output")

        process_button.click(process_pdfs, inputs=[files_input, engine_dropdown], outputs=[summary_output])

    demo.launch(share=True)
    
if __name__ == "__main__":
    main()
