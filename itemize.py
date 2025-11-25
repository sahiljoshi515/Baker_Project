import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
gemini_via_openai_client = OpenAI(
    api_key=GEMINI_API_KEY, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def itemize_with_gemini(ocr_response):
    print("Itemization has started!")
    system_prompt = """You are an expert document structuring assistant. Your task is to analyze long OCR-scanned text and intelligently group related content into logical sections. 
    First identify logical sections or documents, then group related pages/content together, finally label those sections intelligently. For each section, extract structured information and present it in markdown format. Only return in markdown format. Please make sure no information is lost, everything from the ocr version should be included."""
    user_prompt = f"This is a pdf's OCR in markdown:\n\n{ocr_response}\n.\n" + "Convert this into a sensible itemized markdown version of the document."

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = gemini_via_openai_client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=prompts
    )

    return response.choices[0].message.content