import os
import time
import requests
import openai
import gradio as gr
import tiktoken
from pdf2image import convert_from_path
import boto3
import io
import json
from mistral import mistral_ocr
from azure import azure_ocr
from textract import textract_ocr

# ------ OPENAI -------
OPENAI_API_KEY = "<your-api-key>"
openai.api_key = OPENAI_API_KEY

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


def extract_metadata_from_pdfs(files, ocr_engine):
    if not files:
        return "No files uploaded."
    
    forms = []
    
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
            forms.append(f"Invalid OCR engine selected for {pdf_path}.")
            continue

        if not ocr_text:
            forms.append(f"OCR failed for {pdf_path}.")
            continue

        # Summarize text
        metadata = extract_metadata("/Users/sahiljoshi/Documents/Baker_Project/form.json", ocr_text)
        form = merge_filled_forms(metadata)
    return form

import json
import re

def extract_metadata(form_path, text, max_tokens=1000):
    # Load form.json from file
    with open(form_path, 'r') as f:
        form = json.load(f)

    form_str = json.dumps(form, indent=2)

    # Split the input OCR text into manageable chunks
    chunks = split_text_into_chunks(text, max_tokens=max_tokens)

    filled_forms = []

    for chunk in chunks:
        system_prompt = (
            "You are an assistant that needs to fill in the form.json with OCR data accurately. "
            "You will get 2 inputs: form.json and a chunk of OCR text. "
            "You must output a properly filled form.json with the OCR data, and nothing else. "
            "If you are not sure, please leave the field blank."
        )

        user_prompt = f"Form:\n{form_str}\n\nOCR Text:\n{chunk}"

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON from the response content
        try:
            json_start = content.index('{')
            json_text = content[json_start:]
            filled_form = json.loads(json_text)
            filled_forms.append(filled_form)
        except Exception as e:
            print("Error parsing JSON from model response:", e)
            print("Raw response:", content)

    return filled_forms


def merge_filled_forms(filled_forms):
    if not filled_forms:
        return {}

    merged_form = {}

    # Get all keys from the form
    all_keys = set().union(*[form.keys() for form in filled_forms])

    # For each key, take the last non-empty value across all filled forms
    for key in all_keys:
        for form in reversed(filled_forms):  # Just reverse the order here
            value = form.get(key, "").strip()
            if value:
                merged_form[key] = value
                break
        else:
            merged_form[key] = ""  # No non-empty value found

    return merged_form


def main():
    with gr.Blocks() as demo:
        gr.Markdown("## OCR & Metadata Extraction Tool")

        files_input = gr.Files(label="Upload PDFs", type="filepath", file_types=[".pdf"], interactive=True)

        engine_dropdown = gr.Dropdown(
            choices=["Mistral", "Azure", "Textract"],
            label="Select OCR Engine",
            value=None,
            interactive=True,
            info="Select the OCR engine for processing."
        )

        # Two buttons
        summarize_button = gr.Button("Summarize")
        extract_json_button = gr.Button("Extract Metadata (JSON)")

        # Outputs
        with gr.Accordion("Summary Output", open=False):
            summary_output = gr.Markdown(label="OCR Summary")

        with gr.Accordion("JSON Output", open=False):
            json_output = gr.JSON(label="Filled Metadata Form")

        # Connect each button to its function
        summarize_button.click(
            fn=process_pdfs,
            inputs=[files_input, engine_dropdown],
            outputs=[summary_output]
        )

        extract_json_button.click(
            fn=extract_metadata_from_pdfs,
            inputs=[files_input, engine_dropdown],
            outputs=[json_output]
        )

    demo.launch(share=True)

    
if __name__ == "__main__":
    main()
