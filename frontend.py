from nicegui import ui
from datetime import datetime
import asyncio

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
                keyword_input = ui.input(placeholder='Search by keyword...') \
                    .classes('w-full')
                
                date_input = ui.date(value=datetime.now()) \
                    .classes('w-full')
                
                collection_input = ui.input(placeholder='Search by collection...') \
                    .classes('w-full')
                
                # Search button
                def handle_search():
                    search_params = {
                        'keyword': keyword_input.value,
                        'date': date_input.value,
                        'collection': collection_input.value
                    }
                    ui.notify(f"Searching with: {search_params}")
                    # Add your search logic here
                    ui.navigate.to('/extract')  # Navigate to process page
                
                ui.button('Search', on_click=handle_search) \
                    .classes('w-full bg-blue-600 text-white hover:bg-blue-600')
                
# Extraction Page
@ui.page('/extract')
def extraction_tool():
    # Header Section
    with ui.header().classes('bg-blue-800 text-white p-4 shadow-md'):
        with ui.row().classes('w-full justify-between items-center'):
            ui.label('OCR & Metadata Extraction Tool').classes('text-xl font-bold')
            ui.link('Home', '/') \
                .classes('bg-blue-800 hover:text-blue-700 text-white px-4 py-2 rounded-md font-large transition-colors')
    
    # Main Content
    with ui.column().classes('w-full mx-auto p-4 gap-4'):
        # Welcome Markdown
        ui.markdown(
        '''

        Welcome! This tool helps you: \n
        1. **Extract text** from PDF files using a selected OCR engine \n
        2. **Itemize** and **Tag** PDF's \n
        3. **Convert** the extracted text into structured metadata using an LLM engine

        ---
        ''').classes('text-lg')
        
        # --- Step 1: Upload PDFs and select OCR engine ---
        with ui.card().classes('w-full p-4 border border-gray-200 rounded-lg'):
            ui.label('Upload PDF and Select OCR Engine').classes('text-lg font-semibold')
            
            with ui.row().classes('w-full items-end gap-4'):
                # File upload with custom validation
                files_input = ui.upload(
                    label="📂 Upload PDF(s)", 
                    multiple=True,
                    on_upload=handle_upload
                ).classes('flex-grow')
                
                engine_dropdown = ui.select(
                    options=["Mistral", "Textract"],
                    label="🧠 OCR Engine",
                    value="Textract"
                ).classes('w-48')
                
            run_ocr_btn = ui.button('Run OCR', color='primary').classes('w-full')
            ocr_status = ui.label('Ready').classes('text-sm text-gray-500')
            
            with ui.expansion('📝 View OCR Results in Markdown').classes('w-full'):
                markdown_display = ui.markdown('No results yet').classes('p-2 bg-gray-50 rounded')
        
        # --- Step 2A: Itemize Document with Gemini ---
        with ui.card().classes('w-full p-4 border border-gray-200 rounded-lg'):
            ui.label('Itemize Document').classes('text-lg font-semibold')

            with ui.row().classes('w-full items-end gap-4'):
                method_dropdown = ui.select(
                    options=["Hash-based", "Clustering", "Gemini"],
                    label="🧩 Itemization Method",
                    value="Gemini"
                ).classes('flex-grow')

                itemize_btn = ui.button('Itemize', color='green').classes('w-48')

        with ui.card().classes('w-full p-4 border border-gray-200 rounded-lg'):
            ui.label('Tag PDF').classes('text-lg font-semibold')

            with ui.row().classes('w-full items-end gap-4'):
                method_dropdown = ui.select(
                    options=["Adobe"],
                    label="🏷️ Tagging Method",
                    value="Adobe"
                ).classes('flex-grow')

                tag_btn = ui.button('Tag', color='green').classes('w-48')
        # --- Step 2B: Metadata Extraction ---
        with ui.card().classes('w-full p-4 border border-gray-200 rounded-lg'):
            ui.label('Extract Metadata').classes('text-lg font-semibold')
            
            with ui.row().classes('w-full items-end gap-4'):
                llm_dropdown = ui.select(
                    options=["DeepSeek", "GPT-4", "Claude", "Gemini"],
                    label="🤖 LLM Engine",
                    value="DeepSeek"
                ).classes('flex-grow')
                
                extract_btn = ui.button('Extract (JSON)', color='purple')
            
            with ui.expansion('🧾 View Extracted Metadata').classes('w-full'):
                json_output = ui.json_editor({'status': 'No data yet'}).classes('w-full')
    
    # Button Actions
    async def run_ocr_click():
        if not files_input.files:
            ui.notify("Please upload PDF files first!", type='negative')
            return
            
        run_ocr_btn.disable()
        ocr_status.set_text('Processing...')
        
        # Simulate processing only PDF files
        pdf_files = [f for f in files_input.files if f.name.endswith('.pdf')]
        if not pdf_files:
            ui.notify("No valid PDF files found!", type='negative')
            ocr_status.set_text('Failed')
            run_ocr_btn.enable()
            return
            
        await asyncio.sleep(2)  # Simulate processing
        markdown_display.set_content(f'## OCR Results\n\nProcessed {len(pdf_files)} PDF(s)')
        ocr_status.set_text('Completed')
        run_ocr_btn.enable()
        itemize_btn.enable()
        extract_btn.enable()
    
    run_ocr_btn.on_click(run_ocr_click)
    
    def itemize_click():
        itemize_btn.disable()
        # In a real app, generate PDF here
        ui.notify("Itemized PDF ready for download")
        itemize_btn.enable()
    
    itemize_btn.on_click(itemize_click)
    
    def extract_click():
        extract_btn.disable()
        # Simulate extraction
        json_output.update({
            "document": {
                "title": "Sample Document",
                "author": "John Doe",
                "date": str(datetime.now()),
                "key_points": ["Point 1", "Point 2", "Point 3"]
            }
        })
        ui.notify("Metadata extraction complete!")
        extract_btn.enable()
    
    extract_btn.on_click(extract_click)

# 🚀 Start the server
ui.run()

def handle_upload(e):
    if not e.name.endswith('.pdf'):
        ui.notify("Only PDF files are allowed!", type='negative')
        return

    # Save the uploaded content to disk
    save_path = f'uploads/{e.name}'
    with open(save_path, 'wb') as f:
        f.write(e.content.read())  # `e.content` is a file-like object
    
    ui.notify(f'Saved to {save_path}')
