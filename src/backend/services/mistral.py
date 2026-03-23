"""
The following file uses Mistral 3 OCR. Check the docs below for more info.

https://docs.mistral.ai/capabilities/document_ai/basic_ocr
"""

import time
from typing import Callable

import tiktoken
from mistralai import DocumentURLChunk, ImageURLChunk, Mistral
from mistralai.models import OCRResponse

from core.exceptions import ConfigurationError, OCRProcessingError


class MistralOCRProvider:
    def __init__(
        self,
        *,
        api_key: str ,
        model_name: str,
        max_tokens: int = 100000,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client: Mistral = Mistral(api_key=api_key)

    def extract(self, file_name: str, content: bytes) -> tuple[list[str], str]:
        client = self._get_client()
        try:
            # Mistral OCR 2 call (deprecated)
            uploaded_file = client.files.upload(
                file={
                    "file_name": file_name,
                    "content": content,
                },
                purpose="ocr",
            )
            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        except Exception as exc:
            raise OCRProcessingError("Failed to upload PDF to OCR provider") from exc

        pdf_response = self._process_with_retries(
            lambda: client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model=self.model_name,
                include_image_base64=True,
            ),
            "Failed to process PDF with OCR provider",
        )

        pages, markdown_display = self._get_combined_markdown(pdf_response)
        token_count = self._num_tokens(markdown_display)
        if token_count > self.max_tokens:
            raise OCRProcessingError(
                "OCR output exceeds the supported token limit",
                details={"token_count": token_count, "max_tokens": self.max_tokens},
            )

        return pages, markdown_display

    def _get_client(self) -> Mistral:
        if not self.api_key:
            raise ConfigurationError("MISTRAL_API_KEY is not configured")
        if self._client is None:
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def _process_with_retries(self, operation: Callable[[], OCRResponse], message: str) -> OCRResponse:
        wait_time = 1
        last_error: Exception | None = None
        for _ in range(5):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                time.sleep(wait_time)
                wait_time = wait_time * 2
        raise OCRProcessingError(message) from last_error

    def _num_tokens(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model("gpt-4-turbo")
        return len(encoding.encode(text))

    def _replace_images_in_markdown(self, markdown_str: str, images_dict: dict[str, str]) -> str:
        client = self._get_client()
        for image_name, base64_str in images_dict.items():
            _, encoded = base64_str.split(",", 1)
            base64_data_url = f"data:image/jpeg;base64,{encoded}"
            image_response = self._process_with_retries(
                lambda: client.ocr.process(
                    document=ImageURLChunk(image_url=base64_data_url),
                    model=self.model_name,
                ),
                "Failed to process embedded image with OCR provider",
            )
            image_ocr_markdown = image_response.pages[0].markdown
            markdown_str = markdown_str.replace(
                f"![{image_name}]({image_name})",
                f"![]({image_ocr_markdown})",
            )
        return markdown_str

    def _get_combined_markdown(self, ocr_response: OCRResponse) -> tuple[list[str], str]:
        markdowns: list[str] = []
        markdown_to_display: list[str] = []
        for page in ocr_response.pages:
            image_data = {image.id: image.image_base64 for image in page.images}
            markdowns.append(self._replace_images_in_markdown(page.markdown, image_data))
            markdown_to_display.append(page.markdown)
        return markdowns, "\n\n".join(markdown_to_display)
