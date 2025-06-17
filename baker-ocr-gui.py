import gradio as gr
import tiktoken
from mistral import mistral_ocr
# from textract import textract_ocr
import os
from dotenv import load_dotenv
from openai import OpenAI
# import anthropic
import openai
import json
import markdown2
from weasyprint import HTML
import tempfile
import google.generativeai as genai
import numpy as np
from PyPDF2 import PdfReader, PdfWriter
import shutil

# import re

# Load environment variables from .env file
load_dotenv() 

# ------ OPENAI -------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI()

# ------ DEEPSEEK -------
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# deep_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ------ ANTHROPIC -------
# ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
# claude = anthropic.Anthropic()

# ------ GEMINI -------
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
# response = model.generate_content('Teach me about how an LLM works')

# print(response.text)
# gemini_via_openai_client = OpenAI(
#     api_key=GEMINI_API_KEY, 
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )


def run_ocr(files, ocr_engine):
    """
    Process multiple PDF files and return their summaries.

    Args:
        files (list): List of PDF files as file objects
        ocr_engine (str): OCR engine to use (Mistral, or Textract)

    Returns:
        str: Summaries of the PDF files
    """
    if not files:
        return "No files uploaded."

    ocr_response = []  # Initialize empty string to concatenate all OCR responses
    markdown_response = ""

    for file in files:
        pdf_path = file.name
        print(f"Processing {pdf_path} with {ocr_engine} OCR...")

        # Perform OCR
        if ocr_engine == "Mistral":
            file_ocr_response, file_markdown_response = mistral_ocr(pdf_path)
        # elif ocr_engine == "Textract":
        #     file_ocr_response = textract_ocr(pdf_path)

        if not file_ocr_response:
            return f"OCR failed for {pdf_path}."
        
        # ocr_response += "\n\n" + file_ocr_response  # Concatenate the OCR result
        markdown_response += "\n\n" + file_markdown_response

        print(f"Processing {pdf_path} with {ocr_engine} OCR completed!")
        ocr_response += file_ocr_response  # Concatenate the OCR result
        
    yield "OCR COMPLETED", ocr_response, markdown_response


def gpt_extract(ocr_response: str) -> str:
    system_prompt = "You are an assistant that specializes in filling json forms with OCR data. Please fill accurate entries in the fields provided and output a json file only!"
    user_prompt = f"This is the OCR of pdf(s) in markdown:\n\n{ocr_response}\n.\n" + "Convert this into a sensible structured json response containing doc_id,  Title, Language, Subject, Format, Genre, Administration, People and Organizations, Time Span, Date, Summary. You may need multiple summaries if there are multiple documents."


    response = client.responses.create(
        model="gpt-4o",
        input= system_prompt + " " + user_prompt
    )

    # print(response.output_text)
    # chat_response = openai.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ],
    #     temperature=0.5,
    # )

    # Parse and return JSON response
    # response_dict = json.loads(chat_response.choices[0].message.content, strict=False)
    # try:
    #     response_dict = json.loads(response.output_text)
    #     return json.dumps(response_dict, indent=4)
    # except:
    return response.output_text

enc = tiktoken.encoding_for_model("gpt-4-turbo")
def num_tokens_by_tiktoken(text: str) -> int:
    return len(enc.encode(text))

def segment_and_split_pdf(file, ocr_pages, threshold=0.51):
    """
    Segments the input PDF using similarity of page embeddings and returns a zip of split PDFs.
    Args:
        file: File object from gr.File (uploaded PDF)
        ocr_pages: List[str] (each page's OCR text)
        threshold: Similarity threshold to segment

    Returns:
        Path to .zip file containing all splits
    """
    # Compute embeddings for each page
    print("segmenting pdf ...\n")
    for idx in range(len(ocr_pages)):
        num_tokens = num_tokens_by_tiktoken(ocr_pages[idx])
        num_tokens_left = num_tokens
        curr_num = 0
        insert_idx = idx
        text = ocr_pages[idx]
        while(num_tokens_left > 8192):
            curr_text = text[curr_num:curr_num + 8192]
            ocr_pages.insert(insert_idx, curr_text)
            curr_num += 8192
            num_tokens_left -= 8192
            insert_idx += 1
        
    embeddings = [embed_text(page) for page in ocr_pages]

    # Compute similarities between consecutive pages
    similarities = [np.dot(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]
    boundaries = [1 if sim < threshold else 0 for sim in similarities]
    boundaries.append(0 if boundaries[-1] == 0 else 1)  # Make sure boundaries list matches pages


    # Split PDF based on boundaries
    reader = PdfReader(file[0].name)
    num_pages = len(reader.pages)
    if len(boundaries) != num_pages:
        raise ValueError(f"Mismatch: boundaries ({len(boundaries)}) vs pages ({num_pages})")
    out_dir = tempfile.mkdtemp(prefix="splits_")
    split_paths = []
    start = 0
    doc_num = 1

    for i, flag in enumerate(boundaries):
        if flag == 1:
            writer = PdfWriter()
            for j in range(start, i+1):
                writer.add_page(reader.pages[j])
            output_path = os.path.join(out_dir, f"split_{doc_num:03d}.pdf")
            with open(output_path, "wb") as out_f:
                writer.write(out_f)
            split_paths.append(output_path)
            doc_num += 1
            start = i + 1

    # Zip all splits
    zip_path = os.path.join(out_dir, "all_splits.zip")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', out_dir)
    return zip_path


    
def embed_text(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def markdown_to_pdf(markdown_text):
    # ADDITIONS:
    # pattern = re.compile()
    # Convert markdown to HTML
    html = markdown2.markdown(markdown_text)
    print("Converting the markdown to PDF!")
    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        HTML(string=html).write_pdf(tmp_pdf.name)
        return tmp_pdf.name

def extract_metadata(ocr_response, llm_engine):
    """
    Extract metadata from the OCR response based on the selected LLM engine.

    Args:
        ocr_response (dict): OCR response from the OCR engine
        llm_engine (str): LLM engine to use (DeepSeek, GPT-4, or Claude)

    Returns:
        dict: Metadata extracted from the OCR response
    """
    if not ocr_response:
        return "No response provided."

    print(f"Extracting metadata using {llm_engine}...")

    # if llm_engine == "DeepSeek":
    #     metadata = deepseek_extract(ocr_response)
    if llm_engine == "GPT-4":
        metadata = gpt_extract(ocr_response)
    # elif llm_engine == "Claude":
    #     metadata = claude_extract(ocr_response)
    
    if not metadata:
        return f"Failed to extract metadata using {llm_engine}."
        
    print(f"Metadata extraction completed!")
    return metadata

def main():
    with gr.Blocks(css=".gr-button { font-size: 16px !important; } .gr-dropdown, .gr-textbox { font-size: 15px !important; }") as demo:
        gr.Markdown("""
        # 🧾 OCR & Metadata Extraction Tool
        
        Welcome! This tool helps you:
        1. **Extract text** from PDF files using a selected OCR engine
        2. **Convert** that text into structured metadata using an LLM
        
        ---
        """)

        ocr_done_state = gr.State()

        # --- Step 1: Upload PDFs and select OCR engine ---
        gr.Markdown("### 📄 Step 1: Upload PDF and Select OCR Engine")

        with gr.Row():
            files_input = gr.Files(
                label="📂 Upload PDF(s)", 
                type="filepath", 
                file_types=[".pdf"]
            )

            engine_dropdown = gr.Dropdown(
                choices=["Mistral", "Textract"],
                label="🧠 OCR Engine",
                info="Choose the engine for text extraction",
                interactive=True
            )

        run_ocr_btn = gr.Button("▶️ Run OCR", variant="primary")

        ocr_status = gr.Textbox(
            label="Status",
            interactive=False,
            visible=True,
            show_label=False
        )

        # Add markdown display for OCR results
        with gr.Accordion("📝 View OCR Results in Markdown", open=False):
            markdown_display = gr.Markdown(label="OCR Results")

        # --- Step 2A: Itemize Document with Gemini ---
        gr.Markdown("### 📊 Step 2A: Itemize Document with OpenAI Embeddings")

        itemize_btn = gr.Button("📋 Itemize with OpenAI", interactive=False)
        download_pdf = gr.File(label="📥 Download Itemized PDFS", visible=False)

        # --- Hidden: Only needed internally for markdown passing ---
        # itemized_markdown = gr.Textbox(visible=False)

        # --- Step 2B: Metadata Extraction ---
        gr.Markdown("### 🧠 Step 2B: Extract Metadata with an LLM")

        llm_dropdown = gr.Dropdown(
            choices=["DeepSeek", "GPT-4", "Claude"],
            label="🤖 LLM Engine",
            interactive=False,
            info="Select a model to generate structured metadata"
        )

        extract_btn = gr.Button("📤 Extract Metadata (JSON)", interactive=False)

        with gr.Accordion("🧾 View Extracted Metadata", open=False):
            json_output = gr.Markdown(label="Structured Metadata")

        # --- Wiring ---
        run_ocr_btn.click(
            fn=run_ocr,
            inputs=[files_input, engine_dropdown],
            outputs=[ocr_status, ocr_done_state, markdown_display]
        ).then(
            fn=lambda: gr.update(interactive=False),  # Disable the 'Run OCR' button as soon as it's clicked
            inputs=None,
            outputs=[run_ocr_btn]
        ).then(
            fn=lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)),  # Enable the other buttons (LLM dropdown, Itemize button, Extract button)
            inputs=None,
            outputs=[llm_dropdown, extract_btn, itemize_btn]
        ).then(
            fn=lambda: gr.update(interactive=True),  # Re-enable the 'Run OCR' button once OCR is completed
            inputs=None,
            outputs=[run_ocr_btn]
        )

        itemize_btn.click(
            fn=segment_and_split_pdf,
            inputs=[files_input, ocr_done_state],
            outputs=[download_pdf]
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[download_pdf]
        )
        # .then(
        #     fn=markdown_to_pdf,
        #     inputs=[download_pdf],
        #     outputs=[download_pdf]
        # ).then(
        #     fn=lambda: gr.update(visible=True),  # Make the download button visible
        #     inputs=None,
        #     outputs=[download_pdf]
        # ).then(
        #     fn=lambda: gr.update(interactive=True),  # Enable the itemize button after process completion
        #     inputs=None,
        #     outputs=[itemize_btn]
        # )


        # itemize_btn.click(
        #     fn=itemize_with_openai,
        #     inputs=[ocr_done_state],
        #     outputs=[download_pdf]
        # ).then(
        #     fn=lambda: gr.update(visible=True),
        #     inputs=None,
        #     outputs=[download_pdf]
        # )

        extract_btn.click(
            fn=extract_metadata,
            inputs=[ocr_done_state, llm_dropdown],
            outputs=[json_output]
        ).then(
            fn=lambda: gr.update(interactive=True),  # Enable the extract button once the extraction is completed
            inputs=None,
            outputs=[extract_btn]
        )
    demo.launch(share=True)

    
if __name__ == "__main__":
    main()
