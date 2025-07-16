from fastapi import FastAPI, HTTPException
from .schemas import OptimizeRequest, OptimizeResponse
from .core.optimizer import PromptOptimizer

SUPPORTED_MODELS = ["gpt", "claude", "deepseek"]
SUPPORTED_LEVELS = ["light", "medium", "aggressive", "chatbot"]

app = FastAPI(title="prompt-optimizer-API", version="0.1")

@app.get("/")
def read_root():
    return {"message": "Welcome to prompt-optimizer-API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/optimize", response_model=OptimizeResponse)
def optimize_prompt(request: OptimizeRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if request.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model type. Must be one of: {SUPPORTED_MODELS}")

    if request.level not in SUPPORTED_LEVELS:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported optimization level. Must be one of: {SUPPORTED_LEVELS}")

    if request.max_tokens is not None and request.max_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be positive if specified.")

    optimizer = PromptOptimizer(request.model)
    result = optimizer.optimize(
        prompt=request.prompt,
        lvl=request.level,
        optim=request.optimizations
    )

    if request.max_tokens and result["optimized_tokens"] > request.max_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Optimized prompt ({result['optimized_tokens']} tokens) exceeds max_tokens ({request.max_tokens})"
        )

    return OptimizeResponse(
        raw_prompt=request.prompt,
        optimized_prompt=result["optimized_text"],
        original_tokens=result["original_tokens"],
        token_count=result["optimized_tokens"],
        compression_ratio=result["compression_ratio"],
        applied_optimizations=result["applied_optimizations"]
    )