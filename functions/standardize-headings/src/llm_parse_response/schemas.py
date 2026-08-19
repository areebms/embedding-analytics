from pydantic import BaseModel


class AnthropicResponse(BaseModel):

    position: int
    semantic_block: str

