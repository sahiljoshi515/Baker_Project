import json

from openai import OpenAI

from core.exceptions import ConfigurationError, MetadataExtractionError
from models.schemas import MetadataResponse


class OpenAIMetadataProvider:
    def __init__(self, *, api_key: str , model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._client: OpenAI | None = None

    def extract(self, ocr_response: str) -> MetadataResponse:
        client = self._get_client()
        expected_fields = [
            "title",
            "people_and_organizations",
            "description",
            "date",
            "subject",
        ]
        system_prompt = (
            "You are an assistant that fills structured JSON forms based on OCR text input. "
            "You must return a single, valid JSON object with only the specified fields. "
            "Do not include any explanation, markdown, or formatting like triple backticks. "
            f"If a field is missing or unknown, set its value to null. Fill in the fields {expected_fields}."
        )

        try:
            response = client.responses.parse(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": ocr_response,
                    },
                ],
                text_format=MetadataResponse,
            )

            event = response.output_parsed
        except Exception as exc:
            raise MetadataExtractionError("Failed to extract metadata with language model") from exc

        if not event:
            raise MetadataExtractionError("Language model returned an empty metadata response")

        # try:
        #     parsed = json.loads(content)
        # except json.JSONDecodeError as exc:
        #     raise MetadataExtractionError("Language model returned invalid JSON metadata") from exc

        # formatted_metadata = json.dumps(parsed, indent=2)
        return event

    def _get_client(self) -> OpenAI:
        if not self.api_key:
            raise ConfigurationError("OPENAI_API_KEY is not configured")
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client
