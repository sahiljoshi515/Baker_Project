import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()  # By default looks for .env file in current directory

# ------ AZURE ------
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
READ_API_URL = os.getenv("READ_API_URL")

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