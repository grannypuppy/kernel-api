import torch
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

const int BLOCK_SIZE = 16;
const int TILE_SIZE = 16;

template <typename T>
__global__ void sgemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K) {
    
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    T sum = 0;
    
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        int a_col = tile * BLOCK_SIZE + threadIdx.x;
        int b_row = tile * BLOCK_SIZE + threadIdx.y;
        
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

template <typename T>
torch::Tensor sgemm_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    sgemm_kernel<T><<<grid, block>>>(A.data_ptr<T>(), B.data_ptr<T>(), C.data_ptr<T>(), M, N, K);
    
    return C;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");
    
    if (A.scalar_type() == torch::kFloat32) {
        return sgemm_cuda<float>(A, B);
    } else if (A.scalar_type() == torch::kFloat16) {
        return sgemm_cuda<__half>(A, B);
    } else if (A.scalar_type() == torch::kBFloat16) {
        return sgemm_cuda<__nv_bfloat16>(A, B);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_ext = load_inline(
    name="matmul_ext",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cuda_cflags=["-arch=sm_89", "-O3", "--use_fast_math"],
)

class ModelNew:
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul_ext.matmul_cuda(A.cuda(), B.cuda())

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N, device='cuda')
    B = torch.rand(N, N, device='cuda')
    return [A, B]

def get_init_inputs():
    return []