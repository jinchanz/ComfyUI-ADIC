from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class OpenAIImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str
    quality: Optional[str] = None
    background: Optional[str] = None
    n: Optional[int] = 1
    seed: Optional[int] = None
    size: Optional[str] = None


class OpenAIImageEditRequest(OpenAIImageGenerationRequest):
    pass


class OpenAIImageResponseItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class OpenAIImageGenerationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    created: Optional[int] = None
    data: List[OpenAIImageResponseItem] = Field(default_factory=list)
    success: Optional[bool] = None
    message: Optional[str] = None
    raw: Optional[Any] = None
