from typing import Literal

from pydantic import BaseModel, Field, field_validator

from book_records.constants import CUSTOM_ID_ILLEGAL
from llm_classify_request.constants import MODEL, SYSTEM_PROMPT


class AnthropicRequestMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class AnthropicRequestParams(BaseModel):
    model: str = MODEL
    max_tokens: int
    thinking: dict = {"type": "disabled"}
    output_config: dict = {"effort": "low"}
    system: str = SYSTEM_PROMPT
    messages: list[AnthropicRequestMessage]


class AnthropicRequest(BaseModel):
    custom_id: str = Field(max_length=64)
    params: AnthropicRequestParams

    @field_validator("custom_id")
    @classmethod
    def custom_id_must_match_anthropic_format(cls, value):
        if not value or CUSTOM_ID_ILLEGAL.search(value):
            raise ValueError(f"custom_id {value!r} must match ^[a-zA-Z0-9_-]{{1,64}}$")
        return value
