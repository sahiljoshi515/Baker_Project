# Baker_Project

This tool helps you:

- Extract text from PDF files using a selected OCR engine
- Convert that text into structured metadata using an LLM

## Getting Started

Follow these steps to run the application:

1. **Create a Python virtual environment**

   ```bash
   python3 -m venv myenv
   ```

2. **Activate the virtual environment**

  ```bash
  source myenv/bin/activate  # macOS/Linux
  myenv\Scripts\activate.bat  # Windows CMD
  myenv\Scripts\Activate.ps1  # Windows PowerShell
  ```

3. **Install all dependencies**

  ```bash
  pip install -r requirements.txt
  ```

4. **Run the application**

  ```bash
  python3 baker-ocr-gui.py
  ```
The Gradio UI will launch at: http://localhost:7860

## Options Available
### OCR Engines

- Mistral – Lightweight, fast OCR
- Textract – Amazon Textract for high-quality document OCR

### Metadata Extraction (LLMs)

- ChatGPT – OpenAI GPT-4-based extraction
- Claude – Anthropic's Claude model
- Deepseek – Efficient and scalable LLM

### Itemization Models

- Gemini – Google's multimodal model
- Custom Trained Model (To be added)