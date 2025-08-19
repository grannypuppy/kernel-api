import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming the script is run from the root of the kernel-api directory
# Adjust the path if necessary to correctly locate the 'src' and 'KernelBench' directories
try:
    from src.eval import eval_kernel_against_ref, KernelExecResult
    from src.dataset import construct_kernelbench_dataset
    from src.utils import read_file
except ImportError as e:
    logger.error(f"Failed to import KernelBench modules: {e}")
    logger.error("Please ensure that the script is run from the project root and PYTHONPATH is set correctly.")
    # As a fallback for containerized environment, add src to path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from src.eval import eval_kernel_against_ref, KernelExecResult
    from src.dataset import construct_kernelbench_dataset
    from src.utils import read_file


app = FastAPI(
    title="KernelBench API",
    description="An API to evaluate custom CUDA kernels against the KernelBench benchmark.",
    version="1.0.0"
)

# --- Pydantic Models for API Request and Response ---

class EvalParams(BaseModel):
    num_correct_trials: int = Field(3, description="Number of trials for correctness check.")
    num_perf_trials: int = Field(20, description="Number of trials for performance measurement.")
    measure_performance: bool = Field(True, description="Whether to measure performance.")

class EvaluateRequest(BaseModel):
    level: int = Field(..., description="The benchmark level.", example=1)
    problem_id: int = Field(..., description="The problem ID within the level.", example=1)
    custom_code: str = Field(..., description="The source code of the custom kernel to evaluate.", example="import torch\n\ndef kernel(a, b, c):\n    # ... kernel implementation ...")
    eval_params: Optional[EvalParams] = Field(None, description="Optional evaluation parameters.")

# The response will be the JSON representation of KernelExecResult, 
# which is already a Pydantic model, so we don't need a separate response model.

# --- Global State ---
# This dictionary will hold the dataset, loaded at startup.
kernel_bench_dataset: Optional[Dict[int, list[str]] = None
BASE_PATH = Path(__file__).parent

# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    """
    Load the KernelBench dataset on server startup.
    """
    global kernel_bench_dataset
    logger.info("Loading KernelBench dataset...")
    try:
        # Assuming the 'KernelBench' directory is at the same level as this app
        dataset_path = BASE_PATH / 'KernelBench'
        if not dataset_path.exists():
            raise FileNotFoundError(f"KernelBench dataset directory not found at {dataset_path}")
        kernel_bench_dataset = await asyncio.to_thread(construct_kernelbench_dataset, dataset_path)
        logger.info("KernelBench dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load KernelBench dataset: {e}")
        # We will let the server start, but endpoints will fail until the dataset is available.
        kernel_bench_dataset = None

# --- API Endpoints ---

@app.get("/", summary="Health Check", description="Check if the API server is running.")
async def read_root():
    return {"status": "ok", "message": "Welcome to KernelBench API!"}

@app.post("/evaluate", summary="Evaluate a Custom Kernel", response_model=KernelExecResult)
async def evaluate_kernel(request: EvaluateRequest):
    """
    Evaluates a given custom kernel code against a reference kernel from the KernelBench dataset.
    """
    if kernel_bench_dataset is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: KernelBench dataset is not loaded.")

    logger.info(f"Received evaluation request for level {request.level}, problem {request.problem_id}")

    # 1. Find the reference kernel
    try:
        level_data = kernel_bench_dataset.get(request.level)
        if not level_data:
            raise KeyError
        reference_kernel = level_data.get(request.problem_id)
        if not reference_kernel:
            raise KeyError
    except KeyError:
        logger.warning(f"Problem not found: level={request.level}, id={request.problem_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Problem not found for level {request.level} and problem_id {request.problem_id}"
        )

    # 2. Read the reference kernel source code
    try:
        reference_code_path = get_kernel_path(reference_kernel, BASE_PATH)
        original_model_src = await asyncio.to_thread(read_file, reference_code_path)
    except Exception as e:
        logger.error(f"Failed to read reference kernel file {reference_kernel.path}: {e}")
        raise HTTPException(status_code=500, detail="Could not read reference kernel file.")

    # 3. Prepare evaluation parameters
    eval_params = request.eval_params or EvalParams()
    
    # Check for GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Cannot run evaluation.")
        raise HTTPException(status_code=500, detail="CUDA not available on the server.")

    # 4. Run the evaluation
    try:
        logger.info("Starting kernel evaluation...")
        result = await asyncio.to_thread(
            eval_kernel_against_ref,
            original_model_src=original_model_src,
            custom_model_src=request.custom_code,
            num_correct_trials=eval_params.num_correct_trials,
            num_perf_trials=eval_params.num_perf_trials,
            measure_performance=eval_params.measure_performance
        )
        logger.info("Kernel evaluation finished.")
        return result
    except Exception as e:
        logger.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred during evaluation: {str(e)}")

if __name__ == "__main__":
    # This allows running the app directly for debugging purposes.
    # For production, it's better to use a process manager like Gunicorn with Uvicorn workers.
    uvicorn.run(app, host="0.0.0.0", port=8000)
