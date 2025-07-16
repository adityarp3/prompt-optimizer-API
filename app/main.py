from fastapi import FastAPI, HTTPException
from .schemas import OptimizeRequest, OptimizeResponse
from .core .optimizer import PromptOptimizer

SUPPORTED_MODELS = ["gpt", "claude", "deepseek"]

app = FastAPI(title="prompt-optimizer API", version="0.1")

@app.get("/")
def read_root():
    return {"message": "Welcome to prompt-optimizer API"}

@app.get("/health")
def health_check():
    return {"status": "0"}

@app.post("/optimize", response_model=OptimizeResponse)
def optimize_prompt(request: OptimizeRequest):
    if not request.prompt.split():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if request.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model type.")

    optimizer = PromptOptimizer(request.model)
    result = optimizer.optimize(request.prompt, request.max_tokens)

    return OptimizeResponse(
        raw_prompt=request.prompt,
        optimized_prompt=result["optimized_text"],
        token_count=result["optimized_tokens"],
        compression_ratio=result["compression_ratio"]
    )