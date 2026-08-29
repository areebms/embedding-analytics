from typing import Literal

from pydantic import BaseModel, Field

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
