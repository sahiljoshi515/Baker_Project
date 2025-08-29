import os
from dotenv import load_dotenv
from openai import OpenAI
import openai
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
logger.info('Started extraction')

load_dotenv("/Users/amarkanaka/repos/Baker_Project/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI()


# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# deep_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
# gemini_via_openai_client = OpenAI(
#     api_key=GEMINI_API_KEY, 
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

async def gpt_extract(ocr_response: str) -> str:


    expected_fields = [
        "Title", "People and Organizations",
        "Description", "Date", "Subject"
    ]

    fields_str = str(expected_fields)

    # single_value_fields = [
    #     "Title", "Publisher", "Date", "Language"
    # ]

    combined_response = {field: None for field in expected_fields}

    system_prompt = (
        "You are an assistant that fills structured JSON forms based on OCR text input. "\
        "You must return a single, valid JSON object with only the specified fields. "\
        "Do not include any explanation, markdown, or formatting like triple backticks. "\
        f"If a field is missing or unknown, set its value to null. Fill in the fields {fields_str}." \
    )

    # Split into paragraph chunks
    # paragraphs = [p.strip() for p in ocr_response.split('\n\n') if p.strip()]
    # chunks = []
    # current_chunk = ""

    # for para in paragraphs:
    #     if len(current_chunk) + len(para) + 2 <= chunk_size:
    #         current_chunk += f"\n\n{para}" if current_chunk else para
    #     else:
    #         chunks.append(current_chunk)
    #         current_chunk = para
    # if current_chunk:
    #     chunks.append(current_chunk)

    # for i, chunk in enumerate(chunks):
    # user_prompt = (
    #     f"This is part {i+1}/{len(chunks)} of OCR content:\n\n{chunk}\n\n"
    #     "Extract or update these fields:\n" +
    #     "\n".join([f" - {field}" for field in expected_fields]) +
    #     "\nFor Title, Published URL, Publisher, Date, Source, and Format Genre - "
    #     "keep only the most relevant single value.\n"
    #     "For other fields, you may append multiple values separated by commas.\n"
    #     "⚠️ Return ONLY JSON with updated fields. Keep null for missing data."
    # )

    try:
        # logger.info(f"Filling in fields for doc:\n {ocr_response}")
        chat_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ocr_response}
            ],
            temperature=0.1,
        )

        # logger.info(f"response object: {chat_response}")
        content = chat_response.choices[0].message.content
        # logger.info(f"type: {type(content)}")
        # logger.info(f"content: {content}")

        # for field in expected_fields:
        #     if field in chunk_data and chunk_data[field]:
        #         if field in single_value_fields:
        #             if not combined_response[field]:
        #                 combined_response[field] = chunk_data[field]
        #         else:
        #             if combined_response[field]:
        #                 combined_response[field] += f", {chunk_data[field]}"
        #             else:
        #                 combined_response[field] = chunk_data[field]

    except Exception as e:
        # logger.info(f"⚠️ Error processing ocr response")
        return ""
    

    # Final cleanup
    # for field in combined_response:
    #     if combined_response[field] is None:
    #         combined_response[field] = "Null"

    return json.dumps(content, indent = 2)


# def deepseek_extract(ocr_response, chunk_size=10000):
#     system_prompt = (
#         "You are an assistant that fills structured JSON forms based on OCR text input. "
#         "You must return a single, valid JSON object with only the specified fields. "
#         "Do not include any explanation, markdown, or formatting like triple backticks. "
#         "If a field is missing or unknown, set its value to null."
#     )

#     expected_fields = [
#         "Title", "People and Organizations",
#         "Description", "Date", "Subject"
#     ]

#     single_value_fields = [
#         "Title", "Date", "Language"
#     ]

#     combined_response = {field: None for field in expected_fields}

#     # Split into paragraph chunks
#     paragraphs = [p.strip() for p in ocr_response.split('\n\n') if p.strip()]
#     chunks = []
#     current_chunk = ""

#     for para in paragraphs:
#         if len(current_chunk) + len(para) + 2 <= chunk_size:
#             current_chunk += f"\n\n{para}" if current_chunk else para
#         else:
#             chunks.append(current_chunk)
#             current_chunk = para
#     if current_chunk:
#         chunks.append(current_chunk)

#     for i, chunk in enumerate(chunks):
#         user_prompt = (
#             f"This is part {i+1}/{len(chunks)} of OCR content:\n\n{chunk}\n\n"
#             "Extract or update these fields:\n" +
#             "\n".join([f" - {field}" for field in expected_fields]) +
#             "\nFor Title, Published URL, Publisher, Date, Source, and Format Genre - "
#             "keep only the most relevant single value.\n"
#             "For other fields, you may append multiple values separated by commas.\n"
#             "⚠️ Return ONLY JSON with updated fields. Keep null for missing data."
#         )

#         try:
#             chat_response = deep_client.chat.completions.create(
#                 model="deepseek-chat",
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 response_format={"type": "json_object"},
#                 temperature=0.1,
#             )

#             content = chat_response.choices[0].message.content
#             chunk_data = json.loads(content.strip('` \n'))

#             for field in expected_fields:
#                 if field in chunk_data and chunk_data[field]:
#                     if field in single_value_fields:
#                         if not combined_response[field]:
#                             combined_response[field] = chunk_data[field]
#                     else:
#                         if combined_response[field]:
#                             combined_response[field] += f", {chunk_data[field]}"
#                         else:
#                             combined_response[field] = chunk_data[field]

#         except Exception as e:
#             print(f"⚠️ Error processing chunk {i+1}: {str(e)}")
#             continue

#     # Final cleanup
#     for field in combined_response:
#         if combined_response[field] is None:
#             combined_response[field] = "Null"

#     return combined_response  # or: return json.dumps(combined_response, indent=4)


# def gemini_extract(ocr_response):
#     system_prompt = (
#         "You are an assistant that fills structured JSON forms based on OCR text input. "
#         "You must return a single, valid JSON object with only the specified fields. "
#         "Do not include any explanation, markdown, or formatting like triple backticks. "
#         "If a field is missing or unknown, set its value to null."
#     )

#     user_prompt = (
#         f"This is OCR content extracted from a document:\n\n{ocr_response}\n\n"
#         "Please convert this into one structured JSON object with exactly the following fields:\n"
#         " - Title\n"
#         " - People and Organizations\n"
#         " - Summary\n"
#         " - Language\n"
#         " - Date\n"
#         " - Subject\n"
#         "⚠️ Output one valid JSON object only. No markdown, no backticks, no explanation. "
#         "If a field is missing, use null."
#     )

#     prompts = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]

#     chat_response = gemini_via_openai_client.chat.completions.create(
#         model="gemini-2.0-flash",
#         messages=prompts
#     )

#     content = chat_response.choices[0].message.content
#     print("🔍 Gemini raw response:\n", content)

#     def strip_code_fence(text):
#         lines = text.strip().splitlines()
#         if lines and lines[0].strip().startswith("```"):
#             lines = lines[1:]
#         if lines and lines[-1].strip().startswith("```"):
#             lines = lines[:-1]
#         return "\n".join(lines).strip()

#     cleaned = strip_code_fence(content)

#     try:
#         response_dict = json.loads(cleaned)
#     except json.JSONDecodeError as e:
#         print(f"❌ JSON parsing failed: {e}")
#         return {
#             "error": "Model returned invalid JSON",
#             "raw_output": cleaned,
#             "exception": str(e)
#         }

#     # Ensure all required fields exist and null-fill any missing ones
#     required_fields = [
#         "Collection(s)", "Published URL", "Title", "People and Organizations",
#         "Summary", "Language", "Publisher", "Date", "Source",
#         "Subject", "Format Genre", "Accessibility Summary"
#     ]
#     for field in required_fields:
#         if field not in response_dict or response_dict[field] is None:
#             response_dict[field] = "Null"

#     return response_dict
