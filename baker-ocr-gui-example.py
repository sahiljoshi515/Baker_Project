import os
import time
import requests
import gradio as gr
import tiktoken
from mistral import mistral_ocr
from azure import azure_ocr
from textract import textract_ocr


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

# Currently only supports 1 pdf file
# TODO: Support multiple pdf files
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
    
    json_formats = []
    
    for file in files:
        pdf_path = file.name
        print(f"Processing {pdf_path} with {ocr_engine} OCR...")

        # Perform OCR
        if ocr_engine == "Mistral":
            ocr_json = mistral_ocr(pdf_path)
        elif ocr_engine == "Azure":
            ocr_json = azure_ocr(pdf_path)
        elif ocr_engine == "Textract":
            ocr_json = textract_ocr(pdf_path)

        if not ocr_json:
            return f"OCR failed for {pdf_path}."

    print(f"Processing {pdf_path} with {ocr_engine} OCR completed!")
    return ocr_json

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

        # One Button
        extract_json_button = gr.Button("Extract Metadata (JSON)")

        # Outputs
        with gr.Accordion("JSON Output", open=False):
            json_output = gr.JSON(label="Filled Metadata Form")

        # Connect each button to its function
        extract_json_button.click(
            fn=process_pdfs,
            inputs=[files_input, engine_dropdown],
            outputs=[json_output]
        )

    demo.launch(share=True)

    
if __name__ == "__main__":
    main()
