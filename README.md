# Baker_Project (made by Amar Kanakamedala and Sahil Joshi)

This tool helps you:

- Extract text from PDF files using a selected OCR engine
- Convert that text into structured metadata using an LLM
- Store that metadata in our database
- Search texts in our database based on keywords

<!-- - POST /api/search/query accepts SearchQuery { subject?, people?, date?, collection? } and returns SearchResponse { items, total, implementationStatus }, with implementationStatus: "stub" until Elasticsearch is added. -->

More information can be found on OpenAPI (the automatic interactive documentation systems).

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

4. **Set Environment Variables**

- APP_ENV, DEBUG, HOST, PORT
- DATABASE_URL
- OPENAI_API_KEY
- MISTRAL_API_KEY
- CORS_ORIGINS
- FRONTEND_DIST_DIR
- ELASTICSEARCH_URL

5. **Run the application**
  Run 
    - fastapi dev src/backend/main.py
    - python src/frontend/frontend.py


5. **Understand the application**

  Documentation at http://127.0.0.1:8000/docs
  Server started at http://127.0.0.1:8000
  Click on "Process Data" in frontend to see functionality (Search functionality to be added)


## Options Available
### OCR Engine

- Mistral – Lightweight, fast OCR

### Metadata Extraction (LLMs)

- ChatGPT – OpenAI GPT-4-based extraction

