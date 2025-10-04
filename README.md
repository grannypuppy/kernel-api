# KernelBench API

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine.
- An NVIDIA GPU with the appropriate drivers installed.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to enable GPU access for Docker containers.

### 1. Pull the Docker Image

Pull the pre-built image from Docker Hub.

```bash
docker pull jiamin0630/kernelbench-api:latest
```

### 2. Run the Docker Container

Run the container, mapping a local port to the container's port 8000 and granting it access to the host's GPUs.

```bash
# Run the container in detached mode
docker run --gpus all -d -p 8000:8000 --name kernelbench-service jiamin0630/kernelbench-api:latest
```

**Command Breakdown**:
- `--gpus all`: Grants the container access to all available host GPUs. **This is required.**
- `-d`: Runs the container in the background (detached mode).
- `-p 8000:8000`: Maps port 8000 on the host to port 8000 in the container. You can change the host port (the first number) if needed, e.g., `-p 30000:8000`.
- `--name kernelbench-service`: Assigns a memorable name to the running container.

You can check the container logs to ensure it started correctly:
```bash
docker logs kernelbench-service
```

#### Advanced: Limiting GPU Memory Usage

You can control the maximum GPU memory allocated to the service by setting the `GPU_MEMORY_FRACTION` environment variable. This is useful for preventing the service from consuming all available GPU memory on a shared machine. The value should be a float between `0.0` and `1.0`.

For example, to limit the memory usage on all available GPUs to 50% of their capacity, use the `-e` flag:

```bash
docker run --gpus all -d -p 8000:8000 \
  --name kernelbench-service \
  -e GPU_MEMORY_FRACTION=0.5 \
  jiamin0630/kernelbench-api:latest
```

---

## API Reference

**Base URL**: `http://localhost:8000` (or your mapped host port)

### Health Check Endpoint

- **`GET /`**
  - **Description**: A health check endpoint to verify that the API service is running.
  - **Success Response (200 OK)**:
    ```json
    {
      "status": "ok",
      "message": "Welcome to KernelBench API!"
    }
    ```

### Evaluation Endpoint

- **`POST /evaluate`**
  - **Description**: The core endpoint for submitting a kernel for evaluation.
  - **Request Body**: `application/json`

    **Fields**:

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `level` | integer | **Yes** | The problem level in the KernelBench set. |
| `problem_id` | integer | **Yes** | The 1-based problem index within the level. |
| `custom_code` | string | **Yes** | A string containing the full Python source code to evaluate. |
| `eval_params` | object | No | Optional parameters to customize the evaluation. |
| `device` | string | No | The CUDA device to run on (e.g., `cuda:0`). Defaults to the current default device. |

**`eval_params` Object**:

| Field | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `num_correct_trials` | integer | Number of trials for the correctness check. | `3` |
| `num_perf_trials` | integer | Number of trials for the performance benchmark. | `20` |
| `measure_performance`| boolean | Whether to execute the performance test. | `True` |

  - **Success Response (200 OK)**: A JSON object containing the detailed evaluation results.
  - **Error Responses**:
    - `404`: The requested `level` or `problem_id` does not exist.
    - `503`: The server failed to load the benchmark dataset on startup.

---

## Usage Example

This example demonstrates how to evaluate a custom kernel for Level 1, Problem 2.

### 1. Create a Request File

First, create a JSON file named `request.json` with the evaluation details. The `custom_code` field must contain the full source code as a single, escaped string.

```json
{
  "level": 1,
  "problem_id": 2,
  "custom_code": "import torch\nfrom torch.utils.cpp_extension import load_inline\n\nmatrix_multiplication_source = '''\n#include <torch/extension.h>\n#include <cuda_runtime.h>\n\n__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K, int N) {\n    int row = blockIdx.y * blockDim.y + threadIdx.y;\n    int col = blockIdx.x * blockDim.x + threadIdx.x;\n\n    if (row < M && col < N) {\n        float sum = 0.0;\n        for (int i = 0; i < K; ++i) {\n            sum += A[row * K + i] * B[i * N + col];\n        }\n        C[row * N + col] = sum;\n    }\n}\n\ntorch::Tensor matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B) {\n    int M = A.size(0);\n    int K = A.size(1);\n    int N = B.size(1);\n\n    auto C = torch::zeros({M, N}, A.options());\n\n    const int block_size = 16;\n    dim3 blocks((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);\n    dim3 threads(block_size, block_size);\n\n    matrix_multiplication_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);\n\n    return C;\n}\n'''\n\nmatrix_multiplication_cpp_source = (\n    'torch::Tensor matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B);'\n)\n"
}
```

### 2. Send the Request using `curl`

Use a command-line tool like `curl` to send the POST request to the running service.

```bash
curl -X POST http://localhost:8000/evaluate \
-H "Content-Type: application/json" \
-d @request.json
```

### 3. Receive the Response

If successful, the API will return a JSON object with the evaluation results.

```json
{
  "compiled": true,
  "correctness": true,
  "metadata": {
    "hardware": "NVIDIA GeForce RTX 4090",
    "device": "cuda:0",
    "correctness_trials": "(3 / 3)"
  },
  "runtime": 0.000123,
  "runtime_stats": {
    "mean": 0.000123,
    "std": 1.2e-05,
    "min": 0.00011,
    "max": 0.00014,
    "num_trials": 20,
    "hardware": "NVIDIA GeForce RTX 4090",
    "device": "cuda:0"
  }
}
```
*Note: `runtime` values and `hardware` details will vary based on your system.*
