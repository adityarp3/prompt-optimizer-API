from pydantic import BaseModel
from typing import Optional, List

class OptimizeRequest(BaseModel):
    prompt: str
    model: str = "gpt"  # gpt default
    level: str = "aggressive"  # light, medium, aggressive, chatbot
    optimizations: Optional[List[str]] = None  # custom optimization list
    max_tokens: Optional[int] = None  # for future use/validation

class OptimizeResponse(BaseModel):
    raw_prompt: str
    optimized_prompt: str
    original_tokens: int
    token_count: int  # optimized tokens
    compression_ratio: float
    applied_optimizations: List[str]