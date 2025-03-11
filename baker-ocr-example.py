import os
import time
import requests
import openai

AZURE_ENDPOINT = "https://<your-region>.api.cognitive.microsoft.com"
AZURE_API_KEY = "<your_azure_api_key>"
READ_API_URL = f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze"

OPENAI_API_KEY = "<your_openai_api_key>"
openai.api_key = OPENAI_API_KEY

# --- Functions ---

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
    prompt = f"Please provide a concise summary of the following content:\n\n{text}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )
    summary = response.choices[0].message['content']
    return summary

def main():
    input_folder = "./pdfs"    # Folder containing PDF files
    output_folder = "./output" # Folder to save OCR texts and summaries
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing {pdf_path} ...")
            
            # Get OCR text from Azure
            ocr_text = azure_ocr(pdf_path)
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
