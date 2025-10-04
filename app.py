import asyncio
import logging
import os
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
    device: Optional[str] = Field(None, description="The CUDA device to run evaluation on (e.g., 'cuda:0'). Defaults to the current default device.", example="cuda:0")

# The response will be the JSON representation of KernelExecResult, 
# which is already a Pydantic model, so we don't need a separate response model.

# --- Global State ---
# This dictionary will hold the dataset, loaded at startup.
kernel_bench_dataset: Optional[Dict[int, list[str]]] = None
BASE_PATH = Path(__file__).parent

# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    """
    Load the KernelBench dataset and set GPU memory limits on server startup.
    """
    logger.info("Server starting up...")

    # Set per-process GPU memory fraction from environment variable
    try:
        if torch.cuda.is_available():
            gpu_memory_fraction_env = os.environ.get("GPU_MEMORY_FRACTION")
            if gpu_memory_fraction_env:
                try:
                    fraction = float(gpu_memory_fraction_env)
                    if 0.0 < fraction <= 1.0:
                        device_count = torch.cuda.device_count()
                        for i in range(device_count):
                            torch.cuda.set_per_process_memory_fraction(fraction, device=i)
                            logger.info(
                                f"GPU memory usage for device {i} limited to {fraction*100:.0f}% of its capacity."
                            )
                    else:
                        logger.warning(
                            f"GPU_MEMORY_FRACTION must be between 0.0 and 1.0, but got {fraction}. "
                            "Ignoring the setting."
                        )
                except ValueError:
                    logger.warning(
                        f"Could not parse GPU_MEMORY_FRACTION '{gpu_memory_fraction_env}' as a float. "
                        "Ignoring the setting."
                    )
        else:
            logger.warning("CUDA not available. Cannot set GPU memory fraction.")
    except Exception as e:
        logger.error(f"An error occurred while setting GPU memory fraction: {e}", exc_info=True)

    global kernel_bench_dataset
    logger.info("Loading KernelBench dataset...")
    loaded_data = {}
    try:
        for level in (1,2,3,4):
            logger.info(f"Loading level {level} dataset...")
            level_dataset = await asyncio.to_thread(construct_kernelbench_dataset, level)
            if level_dataset:
                loaded_data[level] = level_dataset
                logger.info(f"Level {level} dataset loaded with {len(level_dataset)} problems.")
            else:
                logger.warning(f"construct_kernelbench_dataset returned None for level {level}")
        kernel_bench_dataset = loaded_data
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
    level_problems = kernel_bench_dataset.get(request.level)
    if not level_problems:
        raise HTTPException(
            status_code=404,
            detail=f"Level {request.level} not found or is empty."
        )
    
    # Check if problem_id is within valid range
    if request.problem_id < 1 or request.problem_id > len(level_problems):
        raise HTTPException(
            status_code=404,
            detail=f"Problem ID {request.problem_id} is out of range. Level {request.level} has {len(level_problems)} problems (valid range: 1-{len(level_problems)})."
        )
    
    reference_code_path = level_problems[request.problem_id - 1]

    # 2. Read the reference kernel source code
    try:
        logger.info(f"Reading reference kernel from: {reference_code_path}")
        original_model_src = await asyncio.to_thread(read_file, reference_code_path)
    except Exception as e:
        logger.error(f"Failed to read reference kernel file {reference_code_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not read reference kernel file.")

    # 3. Prepare evaluation parameters
    eval_params = request.eval_params or EvalParams()
    
    # Check for GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Cannot run evaluation.")
        raise HTTPException(status_code=500, detail="CUDA not available on the server.")

    if request.device:
        try:
            device = torch.device(request.device)
            # PyTorch doesn't validate the device index on creation, so we must check it manually.
            if device.type != 'cuda':
                raise HTTPException(status_code=400, detail=f"Unsupported device type '{device.type}'. Only 'cuda' devices are supported.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise HTTPException(status_code=400, detail=f"Requested CUDA device index {device.index} is not available. Server has {torch.cuda.device_count()} devices.")
            
            logger.info(f"Evaluation will run on specified device: {request.device}")
        except RuntimeError as e:
            logger.error(f"Invalid device format '{request.device}': {e}")
            raise HTTPException(status_code=400, detail=f"Invalid device format: '{request.device}'. Use 'cuda' or 'cuda:N'.")
    else:
        device = torch.device("cuda:0")
        
    # 5. Run the evaluation
    try:
        logger.info("Starting kernel evaluation...")
        result = await asyncio.to_thread(
            eval_kernel_against_ref,
            original_model_src=original_model_src,
            custom_model_src=request.custom_code,
            num_correct_trials=eval_params.num_correct_trials,
            num_perf_trials=eval_params.num_perf_trials,
            measure_performance=eval_params.measure_performance,
            device=device
        )
        logger.info("Kernel evaluation finished.")
        return result
    except Exception as e:
        if "CUDA error" in str(e):
            metadata = {
                "cuda_error": f"CUDA Error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }  # log this for debugging as this usually signifies illegal memory access
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result
        else:
            metadata = {"other_error": f"error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        } # for debugging
            eval_result = KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result

if __name__ == "__main__":
    # This allows running the app directly for debugging purposes.
    # For production, it's better to use a process manager like Gunicorn with Uvicorn workers.
    uvicorn.run(app, host="0.0.0.0", port=8000)
