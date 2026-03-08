#include <cuda_runtime.h>
#include <stdio.h>

// RMSNorm kernel - naive version (你的原代码保持不变)
__global__ void rms_norm_naive(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ w,
    float eps,
    int D,
    int total_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens) return;

    float sum_sq = 0.0f;
    
    for (int d = 0; d < D; d += 4) {
        float4 x_vec = *reinterpret_cast<const float4*>(&x[idx * D + d]);
        sum_sq += x_vec.x * x_vec.x;
        sum_sq += x_vec.y * x_vec.y;
        sum_sq += x_vec.z * x_vec.z;
        sum_sq += x_vec.w * x_vec.w;
    }
    
    float rms = rsqrtf(sum_sq / D + eps);
    
    for (int d = 0; d < D; d += 4) {
        float4 x_vec = *reinterpret_cast<const float4*>(&x[idx * D + d]);
        float4 w_vec = *reinterpret_cast<const float4*>(&w[d]);
        
        float4 y_vec;
        y_vec.x = x_vec.x * rms * w_vec.x;
        y_vec.y = x_vec.y * rms * w_vec.y;
        y_vec.z = x_vec.z * rms * w_vec.z;
        y_vec.w = x_vec.w * rms * w_vec.w;
        
        *reinterpret_cast<float4*>(&y[idx * D + d]) = y_vec;
    }
}

// 导出给C++调用的函数（Windows平台）
extern "C" __declspec(dllexport) void launch_rms_norm(
    const float* x,
    float* y,
    const float* w,
    float eps,
    int batch_size,
    int seq_len,
    int dim
) {
    int total_tokens = batch_size * seq_len;
    int threads_per_block = 256;
    int blocks_per_grid = (total_tokens + threads_per_block - 1) / threads_per_block;
    
    rms_norm_naive<<<blocks_per_grid, threads_per_block>>>(x, y, w, eps, dim, total_tokens);
    cudaDeviceSynchronize();
}

// ==================== 修复后的共享内存版本 ====================

// 修复版本1：使用 block 归约的正确实现
__global__ void rms_norm_shared_fixed(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ w,
    float eps,
    int D,
    int total_tokens
) {
    // 每个 block 负责一个 token
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // 动态分配共享内存 - 注意这里不能有多个 extern __shared__ 声明
    extern __shared__ float shared_mem[];
    
    // 划分共享内存区域
    float* x_shared = shared_mem;                    // 存储当前token的x数据 [D]
    float* w_shared = &shared_mem[D];                 // 存储权重数据 [D]
    float* sum_sq_shared = &shared_mem[2 * D];        // 用于归约的共享内存 [block_size]
    
    // 1. 加载x数据到共享内存
    for (int i = tid; i < D; i += block_size) {
        x_shared[i] = x[token_idx * D + i];
    }
    
    // 2. 加载w数据到共享内存
    for (int i = tid; i < D; i += block_size) {
        w_shared[i] = w[i];
    }
    
    __syncthreads();
    
    // 3. 计算当前线程的局部平方和
    float partial_sum = 0.0f;
    for (int i = tid; i < D; i += block_size) {
        float val = x_shared[i];
        partial_sum += val * val;
    }
    
    // 4. 将局部和存入共享内存用于归约
    sum_sq_shared[tid] = partial_sum;
    __syncthreads();
    
    // 5. 树状归约（优化版本）
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_sq_shared[tid] += sum_sq_shared[tid + stride];
        }
        __syncthreads();
    }
    
    // 6. 计算rms值并广播给所有线程
    float rms;
    if (tid == 0) {
        float total_sum_sq = sum_sq_shared[0];
        rms = rsqrtf(total_sum_sq / D + eps);
    }
    
    // 广播rms值到所有线程（使用共享内存）
    if (tid == 0) {
        sum_sq_shared[0] = rms;  // 复用第一个位置存储rms
    }
    __syncthreads();
    rms = sum_sq_shared[0];
    
    // 7. 计算输出并写回全局内存
    for (int i = tid; i < D; i += block_size) {
        y[token_idx * D + i] = x_shared[i] * rms * w_shared[i];
    }
}

// 导出修复后的共享内存版本
extern "C" __declspec(dllexport) void launch_rms_norm_shared(
    const float* x,
    float* y,
    const float* w,
    float eps,
    int batch_size,
    int seq_len,
    int dim
) {
    int total_tokens = batch_size * seq_len;
    int threads_per_block = 256;
    int blocks_per_grid = total_tokens;  // 每个block处理一个token
    
    // 计算共享内存大小：x数据(dim) + w数据(dim) + 归约用数组(threads_per_block)
    size_t shared_mem_size = (dim + dim + threads_per_block) * sizeof(float);
    
    // 启动修复后的核函数
    rms_norm_shared_fixed<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        x, y, w, eps, dim, total_tokens);
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in launch_rms_norm_shared: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}

// ==================== 可选：更高效的版本 ====================

// 优化版本：使用warp级归约，更高效
__global__ void rms_norm_shared_optimized(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ w,
    float eps,
    int D,
    int total_tokens
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    extern __shared__ float shared_mem[];
    float* x_shared = shared_mem;
    float* w_shared = &shared_mem[D];
    
    // 加载数据到共享内存
    for (int i = tid; i < D; i += block_size) {
        x_shared[i] = x[token_idx * D + i];
    }
    for (int i = tid; i < D; i += block_size) {
        w_shared[i] = w[i];
    }
    __syncthreads();
    
    // 计算局部平方和
    float partial_sum = 0.0f;
    for (int i = tid; i < D; i += block_size) {
        float val = x_shared[i];
        partial_sum += val * val;
    }
    
    // warp级归约
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }
    
    // 每个warp的第一个线程将结果存入共享内存
    __shared__ float warp_sums[32];  // 最多32个warp
    if (lane_id == 0) {
        warp_sums[warp_id] = partial_sum;
    }
    __syncthreads();
    
    // 第二个归约阶段：将warp的结果归约
    float total_sum_sq = 0.0f;
    if (warp_id == 0) {
        total_sum_sq = warp_sums[lane_id];
        for (int offset = 16; offset > 0; offset >>= 1) {
            total_sum_sq += __shfl_down_sync(0xffffffff, total_sum_sq, offset);
        }
    }
    
    // 广播结果并计算rms
    __shared__ float rms_val;
    if (tid == 0) {
        rms_val = rsqrtf(total_sum_sq / D + eps);
    }
    __syncthreads();
    
    // 计算输出
    for (int i = tid; i < D; i += block_size) {
        y[token_idx * D + i] = x_shared[i] * rms_val * w_shared[i];
    }
}

// 导出优化版本（可选）
extern "C" __declspec(dllexport) void launch_rms_norm_shared_optimized(
    const float* x,
    float* y,
    const float* w,
    float eps,
    int batch_size,
    int seq_len,
    int dim
) {
    int total_tokens = batch_size * seq_len;
    int threads_per_block = 256;
    int blocks_per_grid = total_tokens;
    
    // 共享内存大小：x数据(dim) + w数据(dim)
    size_t shared_mem_size = (dim + dim) * sizeof(float);
    
    rms_norm_shared_optimized<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        x, y, w, eps, dim, total_tokens);
    
    cudaDeviceSynchronize();
}