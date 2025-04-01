from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from typing import List
import base64
from pathlib import Path
import openai
import json
import tiktoken

# ------ OPENAI -------
OPENAI_API_KEY = "<your-api-key>"
openai.api_key = OPENAI_API_KEY

# ------ MISTRAL -------
MISTRAL_API_KEY = "<your-api-key>"
client = Mistral(api_key=MISTRAL_API_KEY)

# This does the ocr with mistral AI
def mistral_ocr_legacy(pdf_path):
    uploaded_pdf = client.files.upload(
        file={
            "file_name": "uploaded_file.pdf",
            "content": open(pdf_path, "rb"),
        },
        purpose="ocr"
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )  

    all_pages_text = []
    for page_obj in ocr_response.pages:
        page_index = page_obj.index
        page_text = page_obj.markdown
        text_block = f"Page {page_index}:\n{page_text}\n"
        all_pages_text.append(text_block)
    
    # Combine everything into one plain text string
    final_text = "\n".join(all_pages_text)
    return final_text

def mistral_ocr(pdf_path) -> str:
    # load file
    uploaded_pdf = client.files.upload(
        file={
            "file_name": "uploaded_file.pdf",
            "content": open(pdf_path, "rb"),
        },
        purpose="ocr"
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    # process pdf
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url), 
        model="mistral-ocr-latest", 
        include_image_base64=True
    )
    pdf_text = pdf_response.pages[0].markdown
    
    # extract file which image was written to
    images = extract_images(pdf_response)
    image_ocr_markdown = ""
    if images:
        image_texts = []
        try:
            for img_path in images:
                # Ensure file exists
                image_file = Path(img_path)
                assert image_file.is_file()

                # Encode image as base64
                encoded = base64.b64encode(image_file.read_bytes()).decode()
                base64_data_url = f"data:image/jpeg;base64,{encoded}"

                # OCR the image
                image_response = client.ocr.process(
                    document=ImageURLChunk(image_url=base64_data_url),
                    model="mistral-ocr-latest"
                )
                image_texts.append(image_response.pages[0].markdown)

            # Combine all image OCR text
            image_ocr_markdown = "\n".join(image_texts)

        except Exception as e:
            print(f"Image OCR failed: {e}")
            # Optional: return PDF-only OCR, or raise error
            image_ocr_markdown = ""  # or: raise

    system_prompt = "You are an assistant that specializes in filling json forms with OCR data. Please fill accurate entries in the fields provided and output a json file only!"
    user_prompt = f"This is a pdf's OCR in markdown:\n\n{image_ocr_markdown + pdf_text}\n.\n" + "Convert this into a sensible structured json response containing full_text, doc_id,  Title, Language, Subject, Format, Genre, Administration, People and Organizations, Time Span, Date, Summary"

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


def extract_images(ocr_response: OCRResponse) -> List[str]:
    """
    Extract all images from the OCR response, save them to disk, and return a list of the image file names.

    Args:
        ocr_response (OCRResponse): The OCR response from the Mistral API

    Returns:
        List[str]: A list of the image file names
    """
    images = []
    for page in ocr_response.pages:
        for img in page.images:
            header, encoded = img.image_base64.split(",", 1)
            with open(img.id, "wb") as f:
                f.write(base64.b64decode(encoded))
            images.append(img.id)
    return images

def num_tokens_by_tiktoken(text: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-4-turbo")
    return len(enc.encode(text))
  

def mistral_ocr_inplace(pdf_path) -> str:
  # load file
  uploaded_pdf = client.files.upload(
      file={
          "file_name": "uploaded_file.pdf",
          "content": open(pdf_path, "rb"),
      },
      purpose="ocr"
  )
  signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

  # process pdf
  wait_time = 1
  it = 0
  while(it < 5):
    try:
      pdf_response = client.ocr.process(
          document=DocumentURLChunk(document_url=signed_url.url), 
          model="mistral-ocr-latest", 
          include_image_base64=True
      )
      break
    except:
      it+=1
      if wait_time == 1 :
        wait_time *= 2
      else:
        wait_time **2
      time.sleep(wait_time)
      continue
    return "failed"
    
  pdf_text = pdf_response.pages[0].markdown
  
  markdown = get_combined_markdown(pdf_response)
  # print("--------")
  # print("num input tokens: ")
  num_tokens = num_tokens_by_tiktoken(markdown)
  # print(num_tokens)
  # print("--------")

  if(num_tokens > 123000):
    print("num tokens too large")
    return
  # print("--------")
  # print("markdown")
  # print(markdown)

  # Get structured response from model

  system_prompt = "You are an assistant that specializes in filling json forms with OCR data. Please fill accurate entries in the fields provided and output a json file only!"
  user_prompt = f"This is a pdf's OCR in markdown:\n\n{markdown}\n.\n" + "Convert this into a sensible structured json response containing full_text, doc_id,  Title, Language, Subject, Format, Genre, Administration, People and Organizations, Time Span, Date, Description"

  wait_time = 1
  it = 0
  while(it < 5):
    try:
      chat_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
  )
      break
    except:
      it+=1
      if wait_time == 1 :
        wait_time *= 2
      else:
        wait_time **2
      time.sleep(wait_time)
      continue
    return "failed"

  # Parse and return JSON response
  # print("response: ")
  # print(chat_response.choices[0].message.content.replace("\\", " and "))
  response_dict = json.loads(chat_response.choices[0].message.content.replace("\\", " and "), strict=False)

  return json.dumps(response_dict, indent=4)


def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
  for img_name, base64_str in images_dict.items():
        # print("replacing images")
        header, encoded = base64_str.split(",", 1)
        base64_data_url = f"data:image/jpeg;base64,{encoded}"
        # Process image with OCR
        # process pdf
        wait_time = 1
        it = 0
        while(it < 5):
          try:
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=base64_data_url),
                model="mistral-ocr-latest"
            )
            break
          except:
            it+=1
            if wait_time == 1 :
              wait_time *= 2
            else:
              wait_time **2
            time.sleep(wait_time)
            continue
          return "failed"
        # Combine text from image and markdown and extract JSON metadata
        image_ocr_markdown = image_response.pages[0].markdown
        empty_str = ""
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{empty_str}]({image_ocr_markdown})")
        # images_dict[img_name] = im
  # print("done replacing images")
  # print(markdown_str)
  return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
  markdowns: list[str] = []
  for page in ocr_response.pages:
    image_data = {}
    for img in page.images:
      image_data[img.id] = img.image_base64
    markdowns.append(replace_images_in_markdown(page.markdown, image_data))
  # print("done getting combined markdown")
  # print(markdowns)

  return "\n\n".join(markdowns)
