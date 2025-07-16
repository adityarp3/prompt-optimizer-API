from pydantic import BaseModel
from typing import Optional

class OptimizeRequest(BaseModel):
    prompt: str
    model: str = "gpt" # gpt default
    max_tokens: Optional[int] = None # none by default

class OptimizeResponse(BaseModel):
    raw_prompt: str
    optimized_prompt: str
    token_count: int
    compression_ratio: float
