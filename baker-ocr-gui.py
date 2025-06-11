import gradio as gr
import tiktoken
from mistral import mistral_ocr
from textract import textract_ocr
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import openai
import json
import markdown2
import tempfile
import pypandoc
import csv
import time
import re

# Load environment variables from .env file
load_dotenv() 

# ------ OPENAI -------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ------ DEEPSEEK -------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
deep_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ------ ANTHROPIC -------
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
claude = anthropic.Anthropic()

# ------ GEMINI -------
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
gemini_via_openai_client = OpenAI(
    api_key=GEMINI_API_KEY, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def run_ocr(files, ocr_engine):
    """
    Process multiple PDF files and return their summaries.

    Args:
        files (list): List of PDF files as file objects
        ocr_engine (str): OCR engine to use (Mistral)

    Returns:
        str: Summaries of the PDF files
    """
    if not files:
        return "No files uploaded."

    ocr_response = ""  # Initialize empty string to concatenate all OCR responses
    markdown_response = ""

    for file in files:
        pdf_path = file.name
        print(f"Processing {pdf_path} with {ocr_engine} OCR...")

        # Perform OCR
        if ocr_engine == "Mistral":
            file_ocr_response, file_markdown_response = mistral_ocr(pdf_path)
        elif ocr_engine == "Textract":
            file_ocr_response, file_markdown_response = textract_ocr(pdf_path)

        if not file_ocr_response:
            return f"OCR failed for {pdf_path}."
        
        ocr_response += "\n\n" + file_ocr_response  # Concatenate the OCR result
        markdown_response += "\n\n" + file_markdown_response

        print(f"Processing {pdf_path} with {ocr_engine} OCR completed!")
        ocr_response += "\n\n" + file_ocr_response  # Concatenate the OCR result
        
    yield "OCR COMPLETED", ocr_response, markdown_response

def gpt_extract(ocr_response: str) -> str:
    system_prompt = "You are an assistant that specializes in filling json forms with OCR data. Please fill accurate entries in the fields provided and output a json file only!"
    user_prompt = f"This is a pdf's OCR in markdown:\n\n{ocr_response}\n.\n" + "Convert this into a sensible structured json response containing doc_id,  Title, Language, Subject, Format, Genre, Administration, People and Organizations, Time Span, Date, Summary"

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

def split_by_char_count(text: str, max_chars: int = 3000):
    chunks = []
    current = 0
    while current < len(text):
        end = min(current + max_chars, len(text))
        
        # Optional: Try to split at the last newline before max_chars
        split_point = text.rfind('\n', current, end)
        if split_point == -1 or split_point <= current:
            split_point = end
        
        chunks.append(text[current:split_point].strip())
        current = split_point
    return chunks


def gpt_extract_to_csv(ocr_response: str, csv_path: str = "output.csv"):
    system_prompt = (
        "You are an assistant that extracts structured data from OCR text.\n"
        "From each page, extract all relevant people entries as CSV rows with exactly 7 fields in this order:\n\n"
        "1. Name of Person\n"
        "2. Person color (use \"c\" if indicated, otherwise \"null\")\n"
        "3. Address\n"
        "4. Residence type or room count (e.g., 'h', 'r', '3 rms', 'b', 'bds')\n"
        "5. Job title\n"
        "6. Job name (e.g., company or organization)\n"
        "7. Job address\n\n"
        "⚠️ Format rules:\n"
        "- Output only raw CSV rows, one per line\n"
        "- Each row must have exactly 7 fields\n"
        "- If a field is missing or unknown, write \"null\"\n"
        "- Do NOT include column headers, explanations, or extra commas inside fields\n"
        "- Output must be valid CSV (use double quotes for all fields)\n\n"
        "Example row:\n"
        "\"John Doe\",\"c\",\"123 Main St\",\"3 rms\",\"Manager\",\"ABC Co.\",\"456 Elm St\""
    )


    # Prepare CSV and write headers
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "(c)", "Address", "h/r/rms/b/bds", "Job", "Job Name", "Job Address"])

    # Split OCR response by page
    pages = split_by_char_count(ocr_response, max_chars=7000)

    for i, page_text in enumerate(pages):
        user_prompt = f"Page {i+1} OCR content:\n\n{page_text}\n\nExtract the data as a CSV row."

        try:
            chat_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
            )

            csv_output = chat_response.choices[0].message.content.strip()
            print(csv_output)  # Log full response
            rows = list(csv.reader(csv_output.splitlines()))
            valid_rows = [row for row in rows if len(row) == 7]

            if not valid_rows:
                print(f"⚠️ Page {i+1} returned no valid rows.")
            else:
                with open(csv_path, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(valid_rows)
                print(f"✅ Processed page {i+1} with {len(valid_rows)} row(s).")

        except Exception as e:
            print(f"Error processing page {i+1}: {e}")

        time.sleep(5)  # Delay between requests

    return json.dumps({"status": "CSV extraction completed", "file": csv_path}, indent=4)

def deepseek_extract(ocr_response):
    system_prompt = "You are an assistant that specializes in filling json forms with OCR data. Please fill accurate entries in the fields provided and output a json file only!"
    user_prompt = f"This is a pdf's OCR in markdown:\n\n{ocr_response}\n.\n" + "Convert this into a sensible structured json response containing doc_id,  Title, Language, Subject, Format, Genre, Administration, People and Organizations, Time Span, Date, Summary"

    # Get structured response from model
    chat_response = deep_client.chat.completions.create(
        model="deepseek-chat",  # or another model name
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},  # If supported
        temperature=0.3,  # Lower for more deterministic output
    )

    # Parse and return JSON response
    response_dict = json.loads(chat_response.choices[0].message.content)

    return json.dumps(response_dict, indent=4)

def deepseek_extract_to_csv(ocr_response: str, csv_path: str = "output.csv"):
    system_prompt = (
        "You are an assistant that extracts structured data from OCR text.\n"
        "From each page, extract all relevant individual person records as CSV rows with exactly 7 fields in this order:\n\n"
        "1. Full Name of Person\n"
        "2. Person color (use \"c\" if indicated, otherwise \"null\")\n"
        "3. Address\n"
        "4. Residence type or room count (e.g., 'h', 'r', '3 rms', 'b', 'bds')\n"
        "5. Job title\n"
        "6. Job name (e.g., company or organization)\n"
        "7. Job address\n\n"
        "⚠️ Format rules:\n"
        "- Output only raw CSV rows, one per line\n"
        "- Each row must have exactly 7 fields\n"
        "- If a field is missing or unknown, write \"null\"\n"
        "- Do NOT include column headers, explanations, or extra commas inside fields\n"
        "- Output must be valid CSV (use double quotes for all fields)\n\n"
        "Example row:\n"
        "\"John Doe\",\"c\",\"123 Main St\",\"3 rms\",\"Manager\",\"ABC Co.\",\"456 Elm St\""
    )

    # Prepare CSV and write headers
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "(c)", "Address", "h/r/rms/b/bds", "Job", "Job Name", "Job Address"])

    # Split OCR response by page
    pages = split_by_char_count(ocr_response, max_chars=4000)

    for i, page_text in enumerate(pages):
        user_prompt = f"Page {i+1} OCR content:\n\n{page_text}\n\nExtract the data as a CSV row."

        try:
            chat_response = deep_client.chat.completions.create(
                model="deepseek-chat",  # or another model name
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower for more deterministic output
            )

            csv_output = chat_response.choices[0].message.content.strip()
            print(csv_output)  # Log full response
            rows = list(csv.reader(csv_output.splitlines()))
            valid_rows = [row for row in rows if len(row) == 7]

            if not valid_rows:
                print(f"⚠️ Page {i+1} returned no valid rows.")
            else:
                with open(csv_path, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(valid_rows)
                print(f"✅ Processed page {i+1} with {len(valid_rows)} row(s).")

        except Exception as e:
            print(f"Error processing page {i+1}: {e}")

        time.sleep(5)  # Delay between requests

    return json.dumps({"status": "CSV extraction completed", "file": csv_path}, indent=4)

def claude_extract(ocr_response):
    system_prompt = "You are an assistant that specializes in filling json forms with OCR data. Please fill accurate entries in the fields provided and output a json file only!"
    user_prompt = f"This is a pdf's OCR in markdown:\n\n{ocr_response}\n.\n" + "Convert this into a sensible structured json response containing doc_id,  Title, Language, Subject, Format, Genre, Administration, People and Organizations, Time Span, Date, Summary"
    
    chat_response = claude.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1000,
        temperature=0.7,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ]
    )
    
    # Parse and return JSON response
    response_dict = json.loads(chat_response.content[0].text)

    return json.dumps(response_dict, indent=4)


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
        model="gemini-2.0-flash-exp",
        messages=prompts
    )

    return response.choices[0].message.content

def gemini_extract(ocr_response):
    system_prompt = (
        "You are an assistant that fills structured JSON forms based on OCR text input. "
        "You must return a single, valid JSON object with only the specified fields. "
        "Do not include any explanation, markdown, or formatting like triple backticks. "
        "If a field is missing or unknown, set its value to null."
    )

    user_prompt = (
        f"This is OCR content extracted from a document:\n\n{ocr_response}\n\n"
        "Please convert this into one structured JSON object with exactly the following fields:\n"
        " - Asset_Id\n"
        " - Quartex Unique ID\n"
        " - Quartex Name\n"
        " - Collection(s)\n"
        " - Published URL\n"
        " - Title\n"
        " - People and Organizations\n"
        " - Approximate Date\n"
        " - Abstract\n"
        " - Description\n"
        " - Language\n"
        " - Publisher\n"
        " - Date\n"
        " - Rights\n"
        " - Source\n"
        " - Subject\n"
        " - Format Genre\n"
        " - Repository\n"
        " - Special Collections\n"
        " - University Archives\n"
        " - Time Span\n"
        " - Date Digital\n"
        " - Digitization Specification\n"
        " - Original Handle\n"
        " - Rights Summary\n"
        " - Accessibility Summary\n"
        " - Legacy Subjects\n\n"
        "⚠️ Output one valid JSON object only. No markdown, no backticks, no explanation. "
        "If a field is missing, use null."
    )

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    chat_response = gemini_via_openai_client.chat.completions.create(
        model="gemini-2.0-flash-exp",
        messages=prompts
    )

    content = chat_response.choices[0].message.content
    print("🔍 Gemini raw response:\n", content)

    def strip_code_fence(text):
        """
        Removes Markdown-style code fences like ```json ... ```
        """
        lines = text.strip().splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    cleaned = strip_code_fence(content)

    try:
        response_dict = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return json.dumps({"error": "Model returned invalid JSON", "raw_output": cleaned}, indent=4)

    return json.dumps(response_dict, indent=4)


def markdown_to_pdf(markdown_text):
    # Convert markdown to HTML
    html = markdown2.markdown(markdown_text)
    print("Converting the markdown to PDF!")
    # Add PDF-specific formatting options
    extra_args = [
        '--pdf-engine=xelatex',  # Better unicode support
        '--variable', 'geometry:margin=1in',  # Reasonable margins
        '--variable', 'geometry:a4paper',  # Standard paper size
        '--variable', 'mainfont=Helvetica'  # Ensures proper font handling
    ]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pypandoc.convert_text(
            markdown_text,
            'pdf',
            format='md',
            outputfile=tmp_pdf.name,
            extra_args=extra_args
        )
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

    if llm_engine == "DeepSeek":
        metadata = deepseek_extract(ocr_response)
    elif llm_engine == "GPT-4":
        metadata = gpt_extract(ocr_response)
    elif llm_engine == "Claude":
        metadata = claude_extract(ocr_response)
    elif llm_engine == "Gemini":
        metadata = gemini_extract(ocr_response)
    
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
        gr.Markdown("### 📊 Step 2A: Itemize Document with Gemini")

        itemize_btn = gr.Button("📋 Itemize with Gemini", interactive=False)
        download_pdf = gr.File(label="📥 Download Itemized PDF", visible=False)

        # --- Hidden: Only needed internally for markdown passing ---
        itemized_markdown = gr.Textbox(visible=False)

        # --- Step 2B: Metadata Extraction ---
        gr.Markdown("### 🧠 Step 2B: Extract Metadata with an LLM")

        llm_dropdown = gr.Dropdown(
            choices=["DeepSeek", "GPT-4", "Claude", "Gemini"],
            label="🤖 LLM Engine",
            interactive=False,
            info="Select a model to generate structured metadata"
        )

        extract_btn = gr.Button("📤 Extract Metadata (JSON)", interactive=False)

        with gr.Accordion("🧾 View Extracted Metadata", open=False):
            json_output = gr.JSON(label="Structured Metadata")

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
            fn=itemize_with_gemini,
            inputs=[ocr_done_state],
            outputs=[itemized_markdown]
        ).then(
            fn=markdown_to_pdf,
            inputs=[itemized_markdown],
            outputs=[download_pdf]
        ).then(
            fn=lambda: gr.update(visible=True),  # Make the download button visible
            inputs=None,
            outputs=[download_pdf]
        ).then(
            fn=lambda: gr.update(interactive=True),  # Enable the itemize button after process completion
            inputs=None,
            outputs=[itemize_btn]
        )

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
