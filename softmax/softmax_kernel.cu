#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#define WARP_SIZE 32

// Warp 级别归约逻辑
__device__ __forceinline__ void warp_reduce_online(float &local_max, float &local_sum) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float remote_max = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        float remote_sum = __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

        if (remote_max > local_max) {
            local_sum = local_sum * expf(local_max - remote_max) + remote_sum;
            local_max = remote_max;
        } else {
            local_sum += remote_sum * expf(remote_max - local_max);
        }
    }
}

__global__ void softmax_online_vectorized_kernel(const float* __restrict__ x, float* __restrict__ y, int D, int total_rows) {
    int row = blockIdx.x;
    if (row >= total_rows) return;

    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    // 1. 第一趟：计算 local max 和 sum (使用 float4 向量化)
    int vec_size = D / 4;
    const float4* x_vec_ptr = reinterpret_cast<const float4*>(x + row * D);
    
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 x_val = x_vec_ptr[i];
        float vals[4] = {x_val.x, x_val.y, x_val.z, x_val.w};
        for(int j = 0; j < 4; j++) {
            float val = vals[j];
            if (val > local_max) {
                local_sum = local_sum * expf(local_max - val) + 1.0f;
                local_max = val;
            } else {
                local_sum += expf(val - local_max);
            }
        }
    }

    // 2. Block 级别归约
    __shared__ float s_max[WARP_SIZE]; 
    __shared__ float s_sum[WARP_SIZE];
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    warp_reduce_online(local_max, local_sum);

    if (lane == 0) {
        s_max[wid] = local_max;
        s_sum[wid] = local_sum;
    }
    __syncthreads();

    local_max = (tid < (blockDim.x / WARP_SIZE)) ? s_max[lane] : -FLT_MAX;
    local_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
    if (wid == 0) warp_reduce_online(local_max, local_sum);

    __shared__ float m_final;
    __shared__ float d_final;
    if (tid == 0) {
        m_final = local_max;
        d_final = local_sum;
    }
    __syncthreads();

    // 3. 第二趟：写回结果 (使用 float4 向量化)
    float4* y_vec_ptr = reinterpret_cast<float4*>(y + row * D);
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 x_val = x_vec_ptr[i];
        float4 y_val;
        y_val.x = expf(x_val.x - m_final) / d_final;
        y_val.y = expf(x_val.y - m_final) / d_final;
        y_val.z = expf(x_val.z - m_final) / d_final;
        y_val.w = expf(x_val.w - m_final) / d_final;
        y_vec_ptr[i] = y_val;
    }
}

extern "C" __declspec(dllexport) void launch_softmax(const float* x, float* y, int D, int total_rows) {
    int threads_per_block = 256; 
    int blocks_per_grid = total_rows;
    softmax_online_vectorized_kernel<<<blocks_per_grid, threads_per_block>>>(x, y, D, total_rows);
    cudaDeviceSynchronize();
}