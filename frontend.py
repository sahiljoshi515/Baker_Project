from nicegui import ui
from datetime import datetime
# import axios
import asyncio
from io import BytesIO
import os
# from textract import textract_ocr
from markdown2 import markdown  # or mistune, markdown-it-py, etc.
# from weasyprint import HTML as WeasyHTML     # for HTML → PDF conversion
# from itemize import itemize_with_gemini
from extract import deepseek_extract, gemini_extract, gpt_extract
import json
import httpx
from urllib.error import HTTPError

markdown_display = ""
all_pages = []


# HomePage
@ui.page('/')
def page_home():
    # Header Section
    with ui.header().classes('flex justify-between items-center p-4 bg-blue-800 text-white shadow-lg'):
        ui.label('HERITAGE METADATA EXPLORER').classes('text-2xl font-bold tracking-wide')
        
        with ui.row().classes('gap-6 items-center'):
            ui.link('Rice Archives', 
                'https://digitalcollections.rice.edu/white-house-scientist-and-science-policy-dynamic-digital-archive') \
                .classes('bg-blue-800 text-white hover:text-blue-700 px-4 py-2 rounded-md font-large transition-colors')
            
            ui.link('Process Data', '/extract') \
                .classes('bg-blue-800 hover:text-blue-700 text-white px-4 py-2 rounded-md font-large transition-colors')
    
    # Main Content
    with ui.column().classes('max-w-4xl mx-auto p-8 gap-6'):
        # Search Panel
        with ui.card().classes('w-full p-6 border border-gray-800 rounded-lg shadow-sm'):
            with ui.column().classes('w-full gap-4'):
                ui.label('Search for Documents').classes('text-lg font-semibold')
                
                # Search inputs
                subject_input = ui.input(placeholder='Search by Subject...') \
                    .classes('w-full')
                
                # Search inputs
                people_input = ui.input(placeholder='Search by People...') \
                    .classes('w-full')
                
                date_input = ui.date(value=datetime.now()) \
                    .classes('w-full')
                
                collection_input = ui.input(placeholder='Search by collection...') \
                    .classes('w-full')
                
                # Search button
                def handle_search():
                    search_params = {
                        'subject': subject_input.value,
                        'people': people_input.value,
                        'date': date_input.value,
                        'collection': collection_input.value
                    }
                    ui.notify(f"Searching with: {search_params}")
                    # ES logic 
                
                ui.button('Search', on_click=handle_search) \
                    .classes('w-full bg-blue-600 text-white hover:bg-blue-600')

# Extraction Page
@ui.page('/extract')
def extraction_tool():
    # uploaded_file_paths = {}
    # uploaded_files = []
    # markdown_display = None

    spinner_overlay = ui.row().classes(
        'fixed inset-0 bg-black bg-opacity-50 z-50 justify-center items-center'
    )
    spinner = ui.spinner(size='lg', color='white')
    spinner_overlay.visible = False  # initially hidden

    with spinner_overlay:
        with ui.column().classes('items-center'):
            ui.spinner(size='lg', color='white')
            ui.label('Processing...').classes('text-white mt-4')


    async def handle_upload(e):
        if not e.name.endswith('.pdf'):
            ui.notify("Only PDF files are allowed!", type='negative')
            return
        
        # send PDF to backend to process (POST REQUEST)
        """
        UploadFile has the following attributes:
        filename: A str with the original file name that was uploaded (e.g. myimage.jpg).
        content_type: A str with the content type (MIME type / media type) (e.g. image/jpeg).
        file: A SpooledTemporaryFile (a file-like object). This is the actual Python file object that you can pass directly to other functions or libraries that expect a "file-like" object.
        """
        f = {
            'e': (e.name, e.content, 'application/pdf')  # 'e' must match parameter name in your FastAPI route
        }       
        async with httpx.AsyncClient() as client:
            resp = await client.post('http://localhost:8000/api/pdf/ocr', files = f, timeout=None)
            if resp.status_code == 422:
                ui.notify(f"File {e.name} uploaded in improper format")
                # ui.notify(f"error: {resp}")
            else:
                ui.notify(f"file {e.name} uploaded")

        # data = resp.json()
        # pages = data["pages"]
        markdown_to_display = resp.json()
        # print(markdown_to_display)
        all_pages = markdown_to_display['pages']
        if all_pages == None:
            ui.notify(f"Failed to process PDF with error {markdown_to_display['markdown']}")

        # change if scaling to handle multiple PDF's 
        all_pages = markdown_to_display['pages']
        markdown_display.set_content(markdown_to_display['markdown'])
         


    # Header
    with ui.header().classes('bg-blue-900 text-white p-4 shadow'):
        with ui.row().classes('w-full justify-between items-center'):
            ui.label('OCR & Metadata Extraction Tool').classes('text-2xl font-semibold')
            ui.link('Home', '/').classes('hover:underline text-white text-base')

    # Main Body
    with ui.column().classes('w-full max-w-screen-lg mx-auto p-6 gap-y-6'):
        
        # Introduction
        ui.markdown('''
        ## Welcome!

        This tool helps you:
        1. **Extract text** from PDF files using your chosen OCR engine  
        2. **Itemize** and **Tag** content for document segmentation  
        3. **Generate structured metadata** using LLMs

        ---
        ''').classes('text-base')

        # Step 1 - OCR
        with ui.card().classes('w-full p-5 border border-gray-200 rounded-lg shadow-sm'):
            ui.label('Step 1: Extract Text with OCR').classes('text-lg font-semibold')

            with ui.row().classes('w-full items-end gap-4'):
                files_input = ui.upload(
                    label="📂 Upload PDF(s) for OCR'ing", 
                    multiple=True,
                    on_upload=handle_upload
                ).classes('flex-grow')

                # engine_dropdown = ui.select(
                #     options=["Textract", "Mistral"],
                #     label="🧠 OCR Engine",
                #     value="Textract"
                # ).classes('w-52')

            # run_ocr_btn = ui.button('Extract Text', color='primary').classes('mt-4 w-full')
            # ocr_status = ui.label('Status: Ready').classes('text-sm text-gray-500 mt-2')

            with ui.expansion('📄 View OCR Output').classes('w-full mt-4'):
                markdown_display = ui.markdown("In Progress").classes('p-3 bg-gray-50 rounded')

        # Step 2A - Itemize
        # with ui.card().classes('w-full p-5 border border-gray-200 rounded-lg shadow-sm'):
        #     ui.label('Step 2A: Itemize Document').classes('text-lg font-semibold')

        #     with ui.row().classes('w-full items-end gap-4'):
        #         item_method = ui.select(
        #             options=["Gemini", "Hash-based", "Clustering"],
        #             label="🧩 Method",
        #             value="Gemini"
        #         ).classes('flex-grow')

        #         itemize_btn = ui.button('Itemize', color='green').classes('w-48')

        # Step 2B - Tagging
        # with ui.card().classes('w-full p-5 border border-gray-200 rounded-lg shadow-sm'):
        #     ui.label('Step 2B: Tag PDF Content').classes('text-lg font-semibold')

        #     with ui.row().classes('w-full items-end gap-4'):
        #         tag_method = ui.select(
        #             options=["Adobe"],
        #             label="🏷️ Tagging Method",
        #             value="Adobe"
        #         ).classes('flex-grow')

        #         tag_btn = ui.button('Tag', color='green').classes('w-48')

        # Step 3 - Metadata Extraction
        with ui.card().classes('w-full p-5 border border-gray-200 rounded-lg shadow-sm'):
            ui.label('Step 3: Extract Structured Metadata').classes('text-lg font-semibold')

            with ui.row().classes('w-full items-end gap-4'):
                llm_dropdown = ui.select(
                    options=["DeepSeek", "GPT-4", "Gemini"],
                    label="🤖 LLM Engine",
                    value="DeepSeek"
                ).classes('flex-grow')

                extract_btn = ui.button('Generate JSON', color='purple').classes('w-48')

            with ui.expansion('📑 View Metadata JSON').classes('w-full mt-4'):
                json_output = ui.json_editor({'content': {'json': {}}}).classes('w-full')

    # Button Actions
    # async def run_ocr_click():
    #     global ocr_response
    #     if not uploaded_files:
    #         ui.notify("Please upload PDF files first!", type='negative')
    #         return

    #     run_ocr_btn.disable()
    #     spinner_overlay.visible = True  # Show full-screen overlay
    #     ocr_status.set_text('Processing...')
    #     await asyncio.sleep(0.1)  # Let UI refresh

    #     selected_engine = engine_dropdown.value
    #     pdf_files = [f for f in uploaded_files if f.name.endswith('.pdf')]

    #     if not pdf_files:
    #         ui.notify("No valid PDF files found!", type='negative')
    #         ocr_status.set_text('Failed')
    #         spinner_overlay.style('display: none')  # Hide overlay
    #         run_ocr_btn.enable()
    #         return

    #     all_text = ''
    #     for file in pdf_files:
    #         # Route to the selected OCR function
    #         file_path = uploaded_file_paths.get(file.name)
    #         if not file_path:
    #             ui.notify(f"Path for {file.name} not found!")
    #             continue
    #         loop = asyncio.get_running_loop()
    #         if selected_engine == "Mistral":
    #             text = await loop.run_in_executor(None, mistral_ocr, file_path)
    #         # elif selected_engine == "Textract":
    #         #     text = await loop.run_in_executor(None, textract_ocr, file_path)
    #         else:
    #             ui.notify(f"Unknown OCR engine: {selected_engine}", type='negative')
    #             continue

    #         all_text += f"### {file.name}\n{text}\n\n"

    #     spinner_overlay.visible = False  # Hide after work completes
    #     ocr_response = all_text
    #     markdown_display.set_content(f'## OCR Results\n\n{all_text}')
    #     ocr_status.set_text('Completed')
    #     run_ocr_btn.enable()
    #     itemize_btn.enable()
    #     tag_btn.enable()
    #     extract_btn.enable()
    
    # run_ocr_btn.on_click(run_ocr_click)
    
    # async def itemize_click():
    #     if not ocr_response:
    #         ui.notify("Please run OCR first", type='negative')
    #         return

    #     selected_method = item_method.value
    #     itemize_btn.disable()
    #     spinner_overlay.visible = True
    #     await asyncio.sleep(0.1)  # Give UI time to refresh

    #     try:
    #         ui.notify(f"Itemizing using {selected_method}...")
    #         loop = asyncio.get_running_loop()
    #         if selected_method == "Gemini":
    #             # Step 1: Get markdown content from Gemini
    #             itemized_markdown = await loop.run_in_executor(None, itemize_with_gemini, ocr_response)

    #             # Step 2: Convert markdown to HTML
    #             html_content = markdown(itemized_markdown)

    #             # # Step 3: Convert HTML to PDF
    #             output_dir = 'outputs'
    #             os.makedirs(output_dir, exist_ok=True)
    #             pdf_path = os.path.join(output_dir, 'itemized.pdf')
    #             WeasyHTML(string=html_content).write_pdf(pdf_path)

    #             # # Step 4: Offer the file for download
    #             ui.download(pdf_path, filename='itemized.pdf')
    #             ui.notify("Itemized PDF ready!")

    #         elif selected_method == "Hash-based":
    #             ui.notify("Hash-based itemization is not yet implemented", type='warning')

    #         elif selected_method == "Clustering":
    #             ui.notify("Clustering-based itemization is not yet implemented", type='warning')

    #         else:
    #             ui.notify(f"Unknown method: {selected_method}", type='negative')
            
    #     except Exception as e:
    #         ui.notify(f"Itemization failed: {e}", type='negative')

    #     finally:
    #         spinner_overlay.visible = False
    #         itemize_btn.enable()

    
    # itemize_btn.on_click(itemize_click)
    
    # async def extract_click():
    #     if not ocr_response:
    #         ui.notify("Please run OCR first", type='negative')
    #         return
        
    #     selected_llm = llm_dropdown.value
    #     extract_btn.disable()
    #     spinner_overlay.visible = True
    #     ui.notify(f"Extracting metadata using {selected_llm}...")
    #     await asyncio.sleep(0.1)  # Give UI time to refresh
    #     # Simulate extraction
    #     loop = asyncio.get_running_loop()
    #     try:
    #         if selected_llm == "Gemini":
    #             result_json = await loop.run_in_executor(None, gemini_extract, ocr_response)
    #         elif selected_llm == "DeepSeek":
    #             result_json = await loop.run_in_executor(None, deepseek_extract, ocr_response)
    #         elif selected_llm == "GPT-4":
    #             result_json = await loop.run_in_executor(None, gpt_extract, ocr_response)
    #         else:
    #             ui.notify(f"Unknown engine: {selected_llm}", type='negative')
    #             return

    #         json_output.props['properties']['content']['json'] = result_json
    #         json_output.update()
    #         ui.notify("✅ Metadata extraction complete!")

    #     except Exception as e:
    #         ui.notify(f"❌ Extraction failed: {e}", type='negative')

    #     finally:
    #         spinner_overlay.visible = False
    #         extract_btn.enable()
    
    # extract_btn.on_click(extract_click)

# 🚀 Start the server
ui.run()



