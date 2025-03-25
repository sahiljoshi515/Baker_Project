import gradio as gr
import tiktoken
from mistral import mistral_ocr
from azure import azure_ocr
from textract import textract_ocr
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import openai
import json

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

def run_ocr(files, ocr_engine):
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
    
    for file in files:
        pdf_path = file.name
        print(f"Processing {pdf_path} with {ocr_engine} OCR...")

        # Perform OCR
        if ocr_engine == "Mistral":
            ocr_response = mistral_ocr(pdf_path)
        elif ocr_engine == "Azure":
            ocr_response = azure_ocr(pdf_path)
        elif ocr_engine == "Textract":
            ocr_response = textract_ocr(pdf_path)

        if not ocr_response:
            return f"OCR failed for {pdf_path}."
        
        print(f"Processing {pdf_path} with {ocr_engine} OCR completed!")
        yield "OCR COMPLETED", ocr_response

def gpt_extract(ocr_response):
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
                choices=["Mistral", "Azure", "Textract"],
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

        # --- Step 2: Select LLM and extract metadata ---
        gr.Markdown("### 🧠 Step 2: Extract Metadata with an LLM")

        llm_dropdown = gr.Dropdown(
            choices=["DeepSeek", "GPT-4", "Claude"],
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
            outputs=[ocr_status, ocr_done_state]
        ).then(
            fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
            inputs=None,
            outputs=[llm_dropdown, extract_btn]
        )

        extract_btn.click(
            fn=extract_metadata,
            inputs=[ocr_done_state, llm_dropdown],
            outputs=[json_output]
        )

    demo.launch(share=True)

    
if __name__ == "__main__":
    main()
