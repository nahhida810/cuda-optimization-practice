# 深度学习算子开发与优化进阶汇报

## 1. 核心目标

通过对 Transformer 架构中两个关键算子——**Softmax** 与 **RMSNorm** 的底层重构，探索算子在硬件层面上的访存优化（Memory Bound）与并行计算上限。

------

## 2. Softmax 算子：从原生组合到 Triton 融合

### 2.1 算子融合（Operator Fusion）

**痛点**：PyTorch 原生的 `x * scale` 和 `softmax` 是分开的 Kernel，会产生中间显存读写。

**优化方案**：在 Triton 内核中实现融合，做到“一次 Load，计算 Scale+Max+Sum，一次 Store”。

- **性能提升**：实测耗时由 **0.59ms** 降至 **0.30ms**，加速比达 **1.95x**。

### 2.2 超长序列的 Tiling 策略

针对列数超过硬件单次处理上限（如 65536 维度）的情况，手动实现了 **Tiling Softmax**：

- **双遍历逻辑**：第一遍循环块计算全局 $Max$ 与 $Sum$，第二遍循环计算归一化。
- **数值稳定性**：在 Tiling 过程中引入 $e^{(old\_max - new\_max)}$ 修正因子，保证了在线计算 $Sum$ 的精确度。

------

## 3. RMSNorm 算子：极致的 CUDA C++ 硬件压榨

这是本次学习中最体现底层调优的部分，涵盖了从 Naive 到 Shared Memory 的三次演进。

### 3.1 向量化访存优化 (Vectorization)

在 CUDA 代码中，通过将 `float*` 强制转换为 `float4*`，利用硬件的指令级并行：

- **原理**：一次内存指令读取 128 bit（4个 float），显著提升了带宽利用率。
- **代码体现**：`float4 x_vec = *reinterpret_cast<const float4*>(&x[idx * D + d]);`

### 3.2 共享内存与并行归约 (Reduction)

为了进一步提升性能，开发了 **Shared Memory 版本**，这是目前实测表现最好的实现。

- **分工协作**：一个 Block 负责一个 Token，所有线程协作加载数据到共享内存。
- **树状归约 (Tree Reduction)**：摒弃了简单的循环累加，在共享内存中通过 `stride >>= 1` 的方式进行对数级归约计算平方和。
- **实测数据表现**：
  - **加速比**：共享内存版本较 PyTorch 原生提升了 **2.80x**，较基础向量化版本提升了 **2.31x**。
  - **有效带宽**：实测达到 **81.8 GB/s**，在 RTX 4060 上表现优异。

------

## 4. 自动化调优与工程实践

### 4.1 Autotune 性能调优

在 Triton 实现中，利用 `@triton.autotune` 自动搜寻最佳配置：

- 通过对比 `num_warps` (4, 8, 16) 和 `num_stages`，在不同维度下自动适配最优并行度。

### 4.2 C-Python 跨语言通信与 Debug

在集成过程中解决了关键的工程难题：

- **ctypes 适配**：处理了 Windows 平台下 `__declspec(dllexport)` 的导出问题。
- **内存溢出处理**：针对 `OverflowError: int too long to convert` 报错，通过将 `argtypes` 修改为 `ctypes.c_uint64` 成功解决了 64 位显存地址的传递问题。

------

## 5. 实验数据汇总

| **任务**    | **实现方式**        | **耗时 (ms)** | **性能对比/加速比** |
| ----------- | ------------------- | ------------- | ------------------- |
| **Softmax** | PyTorch Native      | 0.59          | 1.00x               |
|             | **Triton Fused**    | **0.30**      | **1.95x**           |
| **RMSNorm** | PyTorch Native      | 0.99          | 1.00x               |
|             | CUDA 向量化         | 0.82          | 1.21x               |
|             | **CUDA Shared Mem** | **0.35**      | **2.80x**           |

------

## 6. 结论

1. **算子融合**是解决计算密集型模型中访存瓶颈的最有效手段。
2. **共享内存与向量化**是 CUDA 调优的“两把利剑”，能将原本受限于显存延迟的任务大幅加速。
3. **Triton** 提供了极高的开发效率，在大多数场景下能达到原生 CUDA 90% 以上的性能。