
# 1. transformer

## 1.1 代码
`LLMCompass_run.py`
```python
from design_space_exploration.dse import template_to_system, read_architecture_template  
 from software_model.transformer import TransformerBlockAutoRegressionTP, TransformerBlockInitComputationTP  
 from software_model.utils import DataType, data_type_dict  
 from software_model.utils import data_type_dict, Tensor  
 ​  
 ​  
 specs = read_architecture_template(r"configs\mi210.json")  
 system = template_to_system(specs)  
 ​  
 #定义序列长度和批次大小  
 seq_len = 12288  
 bs = 8  
 ​  
 model_auto_regression = TransformerBlockAutoRegressionTP(  
         d_model=12288,  
         n_heads=96,  
         device_count=1,  
         data_type=data_type_dict["fp16"],  
     )  
 _ = model_auto_regression(  
   Tensor([bs, 1, 12288], data_type_dict["fp16"]),  
   seq_len,  
 )  
 ​  
 print("Starting simulation...")  
 auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(  
   system, "heuristic-GPU"  
 )  
 print("Simulation completed!")  
 print(f"Simulated latency: {auto_regression_latency_simulated}")  
```
## 1.2 原理


# 2. init
```python
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
```
## 2.1 tensor

这段代码展示了一个多头注意力（**Multi-Head Attention**）和前馈神经网络（**Feed-Forward Network**）的实现，主要是针对分布式计算场景进行的张量并行化操作。它通过将模型参数分配到多个设备上，并为每个设备定义了矩阵乘法、转置、归一化等操作。

---

每个张量（Wq, Wk, Wv, W0, W1, W2）都被分割到多个设备上。假设有 device_count 个设备，每个设备只处理 d_model // device_count 个维度的计算。
```python
        # parameters per device   计算类型大小
        d = d_model# 模型维度
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
```

**Tensor**：用于表示一个张量（类似于 PyTorch 或 TensorFlow 中的张量）。张量是深度学习中的核心数据结构，通常用于存储多维数组以及相关的元数据。
```python
class Tensor:
    def __init__(
        self, shape: List, data_type=data_type_dict["fp16"]
    ) -> None:
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type
```
**size**：用于计算 list 的大小。如果是 list，则调用 size_of_list(list) 来计算列表中所有元素的乘积（即张量的总元素数）。这通常用于根据张量的形状来计算总的元素个数。如果不是 list（例如这是一个自定义的对象如类 Tensor），则调用该对象的 size() 方法。
```python
def size(list):# 如果输入是列表，返回列表中所有元素的乘积；如果输入是对象，返回对象的 size 属性
    if isinstance(list, List):     
        return size_of_list(list)
    else:
        return list.size

def closest_factors(n):# 寻找给定整数 n 的最接近的两个因数  比如传入12返回3,4
    x = int(n**0.5)
    while x >= 1:
        if n % x == 0:
            return x, n // x
        x -= 1
    return 0,0
```
## 2.2 multi-head attention（多头注意力机制）

**Q_proj, K_proj, V_proj**：这三个操作符是矩阵乘法，用于生成查询（Q）、键（K） 和 值（V） 向量。它们对应着 Wq, Wk, Wv 权重矩阵。
**Q_reshape, K_reshape, V_reshape**：这些操作符用于调整查询、键、值向量的形状，以适应多头注意力的计算需求。通常是将 d_model 划分为多个注意力头。
**Q_transpose, K_transpose, V_transpose**：这些操作符会对查询、键、值向量进行转置操作，方便后续进行矩阵乘法。特别是在计算注意力分数时，需要将键向量转置。
**K_concat, V_concat**：用于将来自不同设备上的键、值向量拼接在一起。这在张量并行中非常重要，尤其是在多设备并行计算时，拼接操作确保每个设备的计算结果能够组合在一起。
**Q_mul_K**：批量矩阵乘法操作符，用于计算注意力分数，即查询向量 Q 和键向量 K 的点积。 **A_softmax**：对注意力分数应用 Softmax 操作，生成注意力权重。 
**A_mul_V**：批量矩阵乘法操作符，用于将注意力权重与值向量 V 相乘，生成加权的值向量输出。 
**H_transpose, H_reshape**：用于对多头注意力的输出进行转置和重新调整形状，以便与后续层结合。 
**H_matmul0**：将多头注意力的输出通过 W0 进行线性变换。
**layer_norm0**：层归一化操作符，用于对多头注意力的输出进行归一化，帮助模型更快收敛并稳定训练过程。 
**allreduce_mha**：在分布式环境中，使用 AllReduce 操作将多个设备上的计算结果进行同步和合并。
```python
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
```

## 2.3 feed-forward network（前馈神经网络）

**H_matmul1**：这是前馈神经网络的第一层线性变换，使用权重 W1，将输入维度从 d_model 扩展到 4 * d_model。 
**H_gelu**：这是一个激活函数，通常是 GELU，它是前馈神经网络中的常见激活函数。 
**H_matmul2**：前馈网络的第二层线性变换，使用 W2，将维度从 4 * d_model 映射回 d_model。 **layer_norm1**：对前馈网络的输出进行层归一化。 
**allreduce_ffn**：类似于多头注意力中的 allreduce_mha，这是对前馈网络输出进行 AllReduce 操作，以同步各设备的结果。
```python
self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)
```
# 3. call

**b**：批次大小（batch_size），表示一次输入的样本数。
**d**：隐藏层维度（hidden_dimension），即输入张量的最后一维（d_model）。
**s**：序列长度（sequence_length），即输入张量的第二维。
**h**：多头注意力中的注意力头数量（n_heads）。 
**dev_cnt**：设备数量（device_count），表示并行计算的设备数（如多个 GPU）。
**d_h**：每个注意力头处理的维度大小，由 d_model 除以头的数量得到。
```python
    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, _, d = x.shape
        assert d == self.d_model
        s = seq_len
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
```
## 3.1 KV cache

**K_cache 和 V_cache**：KV 缓存，用于存储键（Key）和值（Value）向量。它们的形状分别为： **K_cache**：形状为 [b, h // dev_cnt, d_h, s]，表示每个设备缓存 h // dev_cnt 个头的键向量，每个键向量的维度为 d_h，序列长度为 s。
**V_cache**：形状为 [b, h // dev_cnt, s, d_h]，表示每个设备缓存 h // dev_cnt 个头的值向量，序列长度为 s，每个值向量的维度为 d_h。
```python
        # KV cache
        K_cache = Tensor([b, h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, h // dev_cnt, s, d_h], self.data_type)
```

## 3.2 multi-head attention
### 3.2.1 生成查询、键、值向量

**Q_proj, K_proj, V_proj**：通过矩阵乘法生成查询向量（Q）、键向量（K） 和 值向量（V），每个向量的形状为 [b, 1, d / dev_cnt]，表示批次大小 b，序列长度为 1（自回归生成），隐藏维度为 d / dev_cnt。
```python
        q = self.Q_proj(x, self.Wq)  # [b, 1, d / dev_cnt]
        assert q.shape == [b, 1, d // dev_cnt]
        k = self.K_proj(x, self.Wk)  # [b, 1, d / dev_cnt]
        v = self.V_proj(x, self.Wv)  # [b, 1, d / dev_cnt]
```
### 3.2.2 查询、键、值向量重塑和转置

**Q_reshape, K_reshape, V_reshape**：将查询、键、值向量的形状调整为 [b, 1, h // dev_cnt, d_h]，表示每个设备处理多个头的维度。
**Q_transpose, K_transpose, V_transpose**：进行转置操作，调整张量的维度以便后续矩阵运算。查询向量 q_T 的形状为 [b, h // dev_cnt, 1, d_h]，键向量 k_T 的形状为 [b, h // dev_cnt, d_h, 1]，值向量 v_T 的形状为 [b, h // dev_cnt, 1, d_h]。
```python
        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])
        q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert q_T.shape == [b, h // dev_cnt, 1, d_h]
        k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, 1]
        assert k_T.shape == [b, h // dev_cnt, d_h, 1]
        v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert v_T.shape == [b, h // dev_cnt, 1, d_h]
```
### 3.2.3 拼接缓存的键和值（KV 缓存）

**K_concat 和 V_concat**：将当前时间步的键、值向量 k_T 和 v_T 与缓存的键、值向量 K_cache 和 V_cache 拼接，生成新的键、值序列。K_T 的形状为 [b, h // dev_cnt, d_h, s + 1]，表示序列长度增加了 1，V_T 的形状为 [b, h // dev_cnt, s + 1, d_h]。
```python
        K_T = self.K_concat(K_cache, k_T, 3)  # [b, h / dev_cnt, d_h, s+1]
        assert K_T.shape == [b, h // dev_cnt, d_h, s + 1]
        V_T = self.V_concat(V_cache, v_T, 2)  # [b, h / dev_cnt, s+1, d_h]
        assert V_T.shape == [b, h // dev_cnt, s + 1, d_h]
```
### 3.2.4 计算注意力分数并应用 Softmax

**Q_mul_K**：通过查询向量 q_T 和键向量 K_T 的点积计算注意力分数，结果形状为 [b, h // dev_cnt, 1, s + 1]。 **A_softmax**：对注意力分数 a 应用 Softmax，生成归一化的注意力权重 a_prob。
```python
        a = self.Q_mul_K(q_T, K_T)  # [b, h / dev_cnt, 1, s+1]
        assert a.shape == [b, h // dev_cnt, 1, s + 1]
```
### 3.2.5 使用注意力权重和值向量生成注意力输出

**A_mul_V**：使用注意力权重 a_prob 对值向量 V_T 进行加权求和，生成多头注意力的输出 h0，形状为 [b, h // dev_cnt, 1, d_h]。
```python
        h0 = self.A_mul_V(a_prob, V_T)  #  [b, h / dev_cnt, 1, d_h]
```
### 3.2.6 重塑和线性变换

**H_transpose, H_reshape**：对注意力输出 h0 进行转置和重塑，最终变为 [b, 1, d // dev_cnt]。 **H_matmul0**：通过 W0 进行线性变换，将输出映射回原始维度 d，形状为 [b, 1, d]。
**layer_norm0**：对变换后的输出进行层归一化。
```python
        h0 = self.H_transpose(h0, [0, 2, 1, 3])  #  [b, 1, h / dev_cnt, d_h]
        assert h0.shape == [b, 1, h // dev_cnt, d_h]
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        assert h0.shape == [b, 1, d // dev_cnt]
        h0 = self.H_matmul0(h0, self.W0)  #  [b, 1, d]
        assert h0.shape == [b, 1, d]
        h0 = self.layer_norm0(h0)
        assert h0.shape == [b, 1, d]
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)
```
## 3.3 feed-forward network
```python
        # feed-forward network
        h1 = self.H_matmul1(h0, self.W1)  # [b, 1, 4 * d / dev_cnt]
        assert h1.shape == [b, 1, 4 * d // dev_cnt]
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2)  #  [b, 1, d]
        assert h2.shape == [b, 1, d]
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)
```
## 3.4 内存计算需求

**memory_requirement**：计算在内存中存储所有权重（Wq, Wk, Wv, W0, W1, W2）、缓存（K_cache, V_cache）所需的总字节数。
返回经过多头注意力机制和前馈神经网络处理后的最终输出 h2，形状为 [b, 1, d]。
```python
        assert h2.shape == [b, 1, d]
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
            + K_cache.size * K_cache.data_type.word_size
            + V_cache.size * V_cache.data_type.word_size
        )
        return h2
```

# 4. compile_and_simulate

## 4.1 matmul

### 4.1.1 simulating qkv

**计算最终的延迟时，分别计算三次矩阵乘法的延迟，并加上每次矩阵乘法的硬件开销，最后将它们乘以 3 得到总延迟。** **qkv_latency 是 Q、K、V 向量生成的总延迟。(所以乘3)**

```python
        # print("simulating qkv")
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
```

`hardware_model/compute_module.py`---`pcb.compute_module.overhead.matmul
```python
class Overhead:
    def __init__(self, matmul, softmax, layernorm, gelu):
        self.matmul = matmul
        self.softmax = softmax
        self.layernorm = layernorm
        self.gelu = gelu


overhead_dict = {
    "A100": Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5),
    "TPUv3": Overhead(11e-5, 30e-5, 14e-5, 10e-5),
    "MI210": Overhead(3.4e-5, 2.2e-5, 2.8e-5, 2.1e-5),
}
```
#### 4.1.1.1 compile_and_simulate
---
**实现了一个矩阵计算的编译和模拟函数，旨在找到最优的执行配置以最小化延迟。它通过计算 I/O 延迟和计算延迟来评估不同配置的性能。**

---

**初始化和提取矩阵**
```python
    def compile_and_simulate(
        self,
        pcb_module: Device,
        compile_mode: str = "exhaustive",
    ):
        min_cycle_count = 2**63 - 1
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        K = self.computational_graph.K
```

**计算工作集大小、i/o延时、计算延时，最后汇总**
**total_io_count**：计算总的 I/O 数据量。它是工作集大小乘以每个数据元素的字大小（word_size）。 ​ **io_latency**：根据硬件设备的带宽（pcb_module.io_module.bandwidth）计算 I/O 延迟。I/O 延迟是总数据量除以带宽得到的时间。
**total_flop_count**：计算矩阵乘法的总浮点运算次数。矩阵乘法的 FLOP 计算为 2 * M * N * K，因为每个矩阵乘法涉及一次乘法和一次加法。 ​
**compute_latency**：根据硬件的计算能力，计算执行这些 FLOP 所需的时间。 ​ **total_vector_flops_per_cycle**：表示硬件每个周期能够执行的总 FLOP 数。
​ **core_count**：硬件计算模块的核心数量。 
​ **clock_freq**：硬件核心的时钟频率。
**self.latency**：最后的延迟是计算延迟和 I/O 延迟中的较大者。因为总的延迟通常是由计算或 I/O 中的瓶颈决定的，所以取二者中的最大值。
**pcb_module.io_module.latency * 2**：这行被注释掉了，可能是为了考虑额外的 I/O 固定延迟或两次存取（读和写）的延迟。在某些情况下，为了更准确地模拟延迟，可能需要加上 I/O 模块的固定延迟。
```python
        if (M == 1 or N == 1) and (
            compile_mode == "heuristic-GPU"
            or compile_mode == "heuristic-our-throughput"
        ):
            working_set_size = M * K + N * K + M * N
            total_io_count = working_set_size * self.data_type.word_size
            io_latency = total_io_count / pcb_module.io_module.bandwidth
            total_flop_count = 2 * M * N * K
            compute_latency = (
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                / pcb_module.compute_module.core_count
                / pcb_module.compute_module.clock_freq
            )
            self.latency = max(
                compute_latency, io_latency
            )  # + pcb_module.io_module.latency * 2
            return self.latency
```
---
**在 GPU 上优化矩阵乘法操作的执行，通过启发式搜索的方法找到最优的块大小（tile size）和其他相关配置（如循环顺序、双缓冲机制等），以最小化执行周期数（cycle_count）。具体而言，它通过调整 L1 和 L2 缓存块大小、K 轴的分块因子、循环顺序、以及 systolic array 的 tiling 因子，逐步模拟不同的配置，最终找到最优配置。**

---

**L2缓存快大小选择和K轴因子选择。 K轴是矩阵乘法中的中间维度（通常是第一个矩阵的列数和第二个矩阵的行数）。如果 K 小于或等于 12288，则使用 [1, 2, 4, 8] 这四个分块因子。如果 K 大于 12288，则使用 [K // 1024, K // 2048, K // 4096, K // 8192]，即更大的分块因子，确保块大小适合硬件的缓存。**
```python
        elif compile_mode == "heuristic-GPU":
            i = 0
            for l2_tile_M in [64, 128, 256, 512, 1024, 2048]:
                for l2_tile_N in [l2_tile_M // 2, l2_tile_M, l2_tile_M * 2]:
                    if K <= 12288:
                        l2_K_tiling_factor_list = [1, 2, 4, 8]
                    else:
                        l2_K_tiling_factor_list = [
                            K // 1024,
                            K // 2048,
                            K // 4096,
                            K // 8192,
                        ]
```

**计算 L2 缓存块在 K 轴的大小和检查 L2 缓存块的工作集大小**
```python
                    for l2_K_tiling_factor in l2_K_tiling_factor_list:
                        l2_tile_K = ceil(
                            self.computational_graph.K / l2_K_tiling_factor
                        )
                        l2_tile_K = 2 ** floor(log2(l2_tile_K))
                        working_set_size = (
                            l2_tile_N * l2_tile_K
                            + l2_tile_M * l2_tile_K
                            + l2_tile_M * l2_tile_N
                        )
                        if (
                            working_set_size
                            > pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                        ):
                            continue
                        elif (
                            working_set_size
                            <= pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                            // 2
                        ):
                            is_l2_double_buffering = True
                        else:
                            is_l2_double_buffering = False

```

**L1 缓存块大小的选择和检查 L1 缓存块的工作集大小**
```python
                        for l1_tile_M in [32, 64, 128, 256]:
                            if l1_tile_M > min(l2_tile_M, l2_tile_N):
                                continue
                            l1_tile_N = l1_tile_M
                            for l1_K_tiling_factor in [1, 2, 4, 8, 16, 32]:
                                l1_tile_K = ceil(l2_tile_K / l1_K_tiling_factor)
                                if (
                                    l1_tile_M * l1_tile_N
                                    + l1_tile_N * l1_tile_K
                                    + l1_tile_M * l1_tile_K
                                    > pcb_module.compute_module.core.SRAM_size
                                    // self.data_type.word_size
                                    // 2
                                ):
                                    continue
```

**循环顺序和 tiling 因子选择**
```python
                                    continue
                                l2_loop_order = "knm"
                                l1_loop_order = "knm"
                                for (
                                    l0_M_tiling_factor,
                                    l0_N_tiling_factor,
                                    l0_K_tiling_factor,
                                ) in self.find_permutations(
                                    pcb_module.compute_module.core.systolic_array_count
                                ):
                                    i += 1
                                    start = time.time()
```

**映射和模拟。映射：创建一个 mapping 对象，包含当前的 L1、L2 块大小、双缓冲设置、循环顺序和 tiling 因子。模拟：调用 simulate 函数，模拟当前配置下的执行周期数（cycle_count）。
```python
                                    mapping = self.Mapping(
                                        l2_tile_M,
                                        l2_tile_N,
                                        l2_tile_K,
                                        is_l2_double_buffering,
                                        l1_tile_M,
                                        l1_tile_N,
                                        l1_tile_K,
                                        l2_loop_order,
                                        l1_loop_order,
                                        l0_M_tiling_factor,
                                        l0_N_tiling_factor,
                                        l0_K_tiling_factor,
                                    )
                                    cycle_count = self.simulate(
                                        self.computational_graph,
                                        mapping,
                                        pcb_module,
                                    )
```

**更新**
```python
 self.best_mapping = best_mapping
        # if self.best_mapping is not None:
        #     self.best_mapping.display()
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # self.best_mapping.display()
        return self.latency
```

#### 4.1.1.2 simulate
---
**实现了一个矩阵计算模拟器，它的功能是模拟特定硬件设备（如深度学习加速器）上执行矩阵乘法操作的周期数（cycle_count）。具体来说，矩阵乘法是通过一种分块策略（tiling strategy）进行的，代码通过模拟每个块（tile）的加载、计算和存储来估计矩阵乘法的总执行周期数。**

---
使用 Pandas 库的 read_csv() 函数从指定路径的 CSV 文件中读取查找表。文件名根据硬件的 Systolic Array 的高度和宽度动态生成。

加载并处理一个查找表（Look-up Table, LUT），该查找表用于记录不同硬件配置（如 Systolic Array 的高度、宽度）和矩阵大小下的执行周期数和利用率。通过将相关参数（如矩阵维度和硬件配置）设置为索引，后续的查找操作可以非常快速，进一步加速整个模拟过程。
```python
 if self.look_up_table is None:
            self.look_up_table = pd.read_csv(
                f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
                header=None,
                names=[
                    "M",
                    "N",
                    "K",
                    "ArrayHeight",
                    "ArrayWidth",
                    "Dataflow",
                    "cycle_count",
                    "util_rate",
                ],
            )
            self.look_up_table.drop_duplicates(
                inplace=True,
                subset=["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
            )
            # self.look_up_table.reset_index(drop=True, inplace=True)
            # self.look_up_table.to_csv(
            #     f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
            #     header=False,
            #     index=False,
            # )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
            )
        # print(self.look_up_table)
        # print(self.look_up_table.loc[(32, 16, 256, 16, 16, 'os'), "cycle_count"
        #                              ].item())
        # print('sdfsdfsdfsd')
        # exit()
```

**确保分块策略（tiling strategy）与硬件的L2缓存大小相匹配，防止分块后的数据量超过缓存容量。这段代码根据是否启用了L2双缓冲机制（double buffering）来进行不同的缓存大小检查。**
**计算 L2 缓存分块数与剩余大小**
```python
 if mapping.is_l2_double_buffering:
            assert (
                l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
                <= pcb_module.compute_module.l2_size // self.data_type.word_size // 2
            )
        else:
            assert (
                l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
                <= pcb_module.compute_module.l2_size // self.data_type.word_size
            )

        M_l2_t = M // l2_tile_M
        N_l2_t = N // l2_tile_N
        K_l2_t = K // l2_tile_K
        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K
```

根据分块策略（tiling strategy）将矩阵乘法的运算划分成多个L2 缓存块，并为每个块创建一个模拟器（L2TileSimulator），用于模拟每个块的加载、计算和存储延迟。代码首先初始化一个三维数组 l2_tiles，表示每个 L2 缓存块的模拟器实例。然后根据矩阵的维度和剩余部分，逐个填充这些缓存块。
```python
l2_tiles = np.empty(
            [ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        )
        # print('-'*20)
        # print(l2_tiles.shape)
        if M_l2_t * N_l2_t * K_l2_t != 0://填充完整的L2缓存块
            l2_tiles[:M_l2_t, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain != 0://处理剩余部分的 L2 缓存块
            l2_tiles[-1, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain != 0://N 维度的剩余块
            l2_tiles[:M_l2_t, -1, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if K_remain != 0://K 维度的剩余块
            l2_tiles[:M_l2_t, :N_l2_t, -1] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
//处理多维剩余块
        if M_remain * N_remain != 0://M 和 N 维度同时有剩余
            l2_tiles[-1, -1, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * K_remain != 0://M 和 K 维度同时有剩余
            l2_tiles[-1, :N_l2_t, -1] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * K_remain != 0://N 和 K 维度同时有剩余
            l2_tiles[:M_l2_t, -1, -1] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain * K_remain != 0://M、N 和 K 维度同时有剩余
            l2_tiles[-1, -1, -1] = self.L2TileSimulator(
                M_remain,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
```

遍历 L2 缓存块.生成遍历 L2 缓存块的循环顺序，具体的顺序由 mapping.l2_loop_order 决定.
获取当前和前一个 L2 缓存块.它们分别用于计算当前块的读取延迟和前一个块的计算和写回延迟。
```python
for m, n, k in self.generate_tile_loops(
    ceil(M / l2_tile_M),
    ceil(N / l2_tile_N),
    ceil(K / l2_tile_K),
    mapping.l2_loop_order,
):
    if m == 0 and n == 0 and k == 0:
        continue
l2_tile = l2_tiles[m, n, k]
previous_l2_tile = l2_tiles[previous_m, previous_n, previous_k]
```

当前 L2 缓存块的读取延迟（current_tile_read_cycle_count）根据当前块与前一个块的位置关系来计算.
额外的 M×N 读取：如果 k > 0 并且当前块和前一个块在 M 和 N 维度上不同（即 m != previous_m 或 n != previous_n），还需要增加 M×N 的读取延迟（l2_tile.M_N_io_cycle_count）。
```python
# current tile read latency
if m == previous_m and k == previous_k:
    current_tile_read_cycle_count = l2_tile.K_N_io_cycle_count
elif n == previous_n and k == previous_k:
    current_tile_read_cycle_count = l2_tile.M_K_io_cycle_count
else:
    current_tile_read_cycle_count = (
        l2_tile.M_K_io_cycle_count + l2_tile.K_N_io_cycle_count
    )
    
if k > 0 and not (m == previous_m and n == previous_n):
    current_tile_read_cycle_count += l2_tile.M_N_io_cycle_count
```

计算前一个块的计算延迟
 计算前一个块的写回延迟
```python
# previous tile compute latency
previous_tile_compute_cycle_count = previous_l2_tile.compute_cycle_count
if k > 0:
    previous_tile_compute_cycle_count += (
        previous_l2_tile.K_reduction_cycle_count
    )

# previous tile write latency
if m == previous_m and n == previous_n:
    previous_tile_write_cycle_count = 0
else:
    previous_tile_write_cycle_count = previous_l2_tile.M_N_io_cycle_count
```

根据是否启用双缓冲机制更新总周期数
```python
# read current tile, compute previous tile, write previous tile
if mapping.is_l2_double_buffering:  # pipelined
    total_cycle_count += (
        max(
            current_tile_read_cycle_count, previous_tile_compute_cycle_count
        )
        + previous_tile_write_cycle_count
    )
else:  # non-pipelined
    total_cycle_count += (
        current_tile_read_cycle_count
        + previous_tile_compute_cycle_count
        + previous_tile_write_cycle_count
    )
```

更新前一个块的索引
处理最后一个块的计算和写回
```python
previous_m = m
previous_n = n
previous_k = k

total_cycle_count += (
    l2_tiles[-1, -1, -1].M_N_io_cycle_count
    + l2_tiles[-1, -1, -1].compute_cycle_count
)
if previous_k > 0:
    total_cycle_count += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)
```

##### 4.1.1.2.1 L2TileSimulator


###### 4.1.1.2.1.1 simulate_l2_tile_io_cycle_count


###### 4.1.1.2.1.2 simulate_l2_tile_compute_cycle_count


##### 4.1.1.2.2 L1TileSimulat

###### 4.1.1.2.2.1 simulate_l1_tile_compute_cycle_count

###### 4.1.1.2.2.2 simulate_systolic_array_cycle_count




### 4.1.2 simulating q_mul_k

### 4.1.3 simulating a_mul_v

### 4.1.4 simulating h_matmul0

### 4.1.5 simulating h1_matmul1

### 4.1.6 simulating h2_matmul2

### 4.1.7 原理

#### 4.1.7.1 tiling

## 4.2 normalization

### 4.2.1 softmax_latency
---
```python
    softmax_latency = (
            self.A_softmax.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.softmax
        )
```
---
初始化
**self.computational_graph.data_type**：从设备的 vector_unit 中获取数据类型，并将其设置为当前计算图（computational_graph）的数据类型。这个数据类型决定了矩阵乘法的计算精度（如 float32, int8 等），并影响数据的存储和传输大小
**min_cycle_count**：初始化为正无穷大，用于存储模拟过程中找到的最小执行周期数（cycle_count）。
**best_mapping**：初始化为 None，用于存储找到的最优分块策略。
**M 和 N**：分别表示矩阵的行数和列数，来自计算图（computational_graph）。
**data_type**：矩阵数据类型，用于计算数据大小（如每个数据点的字节数）
```python
        self.computational_graph.data_type = pcb_module.compute_module.core.vector_unit.data_type
        min_cycle_count = float("inf")
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        data_type = self.computational_graph.data_type
```

计算 L2 缓存块的大小。
L2_size 是 L2 缓存的总大小，l2_tile_N 是 N 维度的分块大小，word_size 是数据类型的字节大小。确保 M 维度的分块大小不会超过 L2 缓存的容量。
```python
l2_tile_N = N
l2_tile_M = (
    pcb_module.compute_module.l2_size // (l2_tile_N * data_type.word_size)
)
l2_tile_M = min(l2_tile_M, M)
is_l2_double_buffering = False
```

**这段代码的作用是遍历不同的 L1 缓存块（Tile）划分策略，并找到在给定硬件配置（pcb_module）下执行时间最短的分块策略（mapping）。它通过模拟每种分块策略的执行周期数（cycle count），并记录最优的结果。**
```python
for l1_N_tiling_factor in [1, 2, 4, 8, 16, 32]:
#调整 L1 缓存块在N维度和M维度上的分块大小的因子
            l1_tile_N = ceil(l2_tile_N / l1_N_tiling_factor)
            for l1_tile_M in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for is_l1_double_buffering in [True, False]:
                    if is_l1_double_buffering:
	                    # l1_tile_M * l1_tile_N * data_type.word_size：这是当前分块的大小（字节数）
	                    if (
                            l1_tile_M * l1_tile_N * data_type.word_size
                            > pcb_module.compute_module.core.SRAM_size // 2
                        ):
                            continue
                    else:
                        if (
                            l1_tile_M * l1_tile_N * data_type.word_size
                            > pcb_module.compute_module.core.SRAM_size
                        ):
                            continue
                    mapping = self.Mapping(#构造映射
                        l2_tile_M,
                        l2_tile_N,
                        is_l2_double_buffering,
                        l1_tile_M,
                        l1_tile_N,
                        is_l1_double_buffering,
                    )
                    cycle_count = self.simulate(#模拟当前分块策略的执行周期数
                        self.computational_graph, mapping, pcb_module
                    )
                    if cycle_count < min_cycle_count:
                        min_cycle_count = cycle_count
                        best_mapping = mapping
```

计算最佳延时
```python
        self.best_mapping = best_mapping#记录最优的分块映射策略
        self.best_cycle_count = min_cycle_count#记录最优分块策略下的最小执行周期数
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq#根据最小执行周期数和硬件的时钟频率计算最优延迟（latency）
        self.latency = self.best_latency
```

#### 4.2.1.1 simulate

目的是模拟一个矩阵运算（如矩阵乘法）在特定硬件设备（pcb_module）上的执行过程。主要是通过L2 缓存分块（L2 tiling）策略，将矩阵划分为多个 L2 缓存块（L2 tiles），并模拟每个块的读取、计算和写回操作，最终计算出整个运算所需的总周期数（total_cycle_count）。
```python
def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
    ) -> int:
        M = computational_graph.M
        N = computational_graph.N
        data_type = computational_graph.data_type
        l2_tile_M = mapping.l2_tile_M

        if mapping.is_l2_double_buffering:
            assert (
                l2_tile_M * N * data_type.word_size * 2
                <= pcb_module.compute_module.l2_size
            )
        else:
            assert (
                l2_tile_M * N * data_type.word_size <= pcb_module.compute_module.l2_size
            )
		# 计算完整的 L2 缓存块数和剩余部分
        M_l2_t = M // l2_tile_M
        M_remain = M % l2_tile_M

		# 初始化 L2 缓存块的模拟器数组
		# l2_tiles：这是一个用来存储每个 L2 缓存块模拟器实例的数组。数组的大小为 ceil(M / l2_tile_M)，即 M 维度上 L2 块的数量。        
		
		l2_tiles = np.empty([ceil(M / l2_tile_M)], dtype=self.L2TileSimulator)
		
		# 填充完整的 L2 缓存块
        if M_l2_t != 0:
            l2_tiles[:M_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N,
                data_type,
                mapping,
                pcb_module,
            )
		
		# 填充剩余的 L2 缓存块        
		if M_remain != 0:
            l2_tiles[-1] = self.L2TileSimulator(
                M_remain,
                N,
                data_type,
                mapping,
                pcb_module,
            )

		# 计算总周期数
        total_cycle_count = 0
        l2_tile_count = ceil(M / l2_tile_M)
        for m in range(l2_tile_count):
            total_cycle_count += l2_tiles[m].read_cycle_count# 读取延时
            total_cycle_count += l2_tiles[m].compute_cycle_count# 计算延时
            total_cycle_count += l2_tiles[m].write_cycle_count# 写回延时
        return total_cycle_count```

##### 4.2.1.1.1 L2TileSimulator

定义了一个名为 L2TileSimulator 的类，它用于模拟 L2 缓存块（Tile） 在特定硬件设备（pcb_module）上的执行周期数。它的作用是估算在处理矩阵运算时，每个 L2 缓存块的读取延迟、写入延迟和计算延迟，从而用于整个矩阵运算性能的评估。
```python
class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            self.M = M
            self.N = N
            self.read_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.write_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
```

simulate_l2_tile_io_cycle_count 的作用是模拟 L2 缓存块的 I/O 操作所需的时钟周期数。具体来说，**它估算了将一个大小为 M × N 的矩阵块从内存加载到 L2 缓存（或从 L2 缓存写回到内存）所需的时钟周期数。**计算的依据是矩阵块的大小、数据类型和硬件的 I/O 带宽以及时钟频率。
```python
def simulate_l2_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, chiplet_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (
	                # 硬件 I/O 模块的带宽，表示每秒可以传输的字节数
                    chiplet_module.io_module.bandwidth
                    # 硬件计算模块的时钟频率，表示每秒的时钟周期数
                    / chiplet_module.compute_module.clock_freq
                )
            )
            ```

simulate_l2_tile_compute_cycle_count 函数的目的是模拟在执行 Softmax 操作 时，L2 缓存块的总计算周期数（compute cycle count）。该函数通过将矩阵划分为多个 L1 缓存块，然后对每个 L1 缓存块进行读取、计算和写回操作的模拟，从而估算整个 L2 缓存块的计算延迟。
```python
def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = Softmax.L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                data_type,
                mapping,
                pcb_module,
            )
            # 计算 L1 缓存块的数量
            l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N) 
            #计算 L1 缓存块的总周期数
            l1_tile_cycle_count = (
                l1_tile.read_cycle_count
                + l1_tile.write_cycle_count
                + l1_tile.compute_cycle_count
            )
            # 
            total_cycle_count = (
	            # 表示 L1 缓存块的数量除以核心数，得到需要调度的批次数（即每次并行调度多少 L1 缓存块）。加 1 是为了处理不完全分块的情况，确保所有块都能被调度
                ceil(l1_tile_count / pcb_module.compute_module.core_count) + 1
            ) * (
	            # Softmax 操作通常需要在 N 维度上进行归约操作，这里使用 log2 来模拟归约操作的复杂度。l1_tile.reduction_cycle_count 表示归约操作的延迟
                l1_tile_cycle_count
                + log2(ceil(N / l1_tile_N)) * l1_tile.reduction_cycle_count
            )
            return total_cycle_count
```

##### 4.2.1.1.2 L1TileSimulator

L1TileSimulator 用于模拟一个 L1 缓存块（Tile） 在执行 Softmax 操作时的行为。它主要估算每个 L1 缓存块的 读取周期数、计算周期数、写入周期数 以及 归约周期数，并依赖于具体的硬件配置（pcb_module）来进行计算模拟。通过这些模拟，可以评估每个 L1 缓存块在不同硬件上的执行延迟，从而优化矩阵操作的性能。

---
`M * N * (self.flops_per_exp + 2)`
softmax的计算步骤是
1. 计算每个元素的指数（这里self.flops_per_exp代表了执行这个步骤所需的的flops
2. 对所有指数值进行求和
3. 归一化每个元素，每个指数除以总和
**这里加2就是补充一次加法和一次除法**
![[Pasted image 20240926172125.png]]

`/ pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle`
表示硬件的向量单元在每个时钟周期内能够执行的总 FLOPs

`M * N * data_type.word_size * 2`
Softmax 操作中的 数据传输延迟 模拟. **乘2是因为归约操作通常需要两次数据传输**。
1. 读取指数值：从 L2 缓存读取指数值进行求和。
2. 写回归一化后的结果：将归一化的结果写回 L2 缓存。

`/ (pcb_module.compute_module.l2_bandwidth_per_cycle / pcb_module.compute_module.core_count)`
 cb_module.compute_module.l2_bandwidth_per_cycle：表示硬件的 L2 缓存每个时钟周期的带宽，即在一个时钟周期内可以从 L2 缓存传输多少字节的数据。
pcb_module.compute_module.core_count：表示每个核心的带宽。通常带宽会在所有核心之间共享，因此需要将总带宽除以核心数。

```python
class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            self.M = M
            self.N = N
            self.flops_per_exp = (
	            # 表示硬件的向量单元每秒可以完成的指数运算次数（FLOPs）
                pcb_module.compute_module.core.vector_unit.flops_per_exp
            )
            self.read_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.write_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            # L1 缓存块的归约周期数
            self.reduction_cycle_count = (
                M
                * N
                * (self.flops_per_exp + 2)
                /                                pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                + M
                * N
                * data_type.word_size
                * 2
                / (pcb_module.compute_module.l2_bandwidth_per_cycle/pcb_module.compute_module.core_count)
            )```

simulate_l1_tile_io_cycle_count 是一个用于估算 L1 缓存块（Tile） 进行 I/O 操作（读取或写回） 所需的时钟周期数的函数。该函数通过根据矩阵的大小（M × N）、数据类型的字节大小以及硬件的 L2 缓存带宽 来计算传输这些数据所需的周期数。

I/O 操作通常指的是 从 L2 缓存读取数据到 L1 缓存 或 将 L1 缓存中的数据写回到 L2 缓存。
```python
 def simulate_l1_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, pcb_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (pcb_module.compute_module.l2_bandwidth_per_cycle)
            )```

simulate_l1_tile_compute_cycle_count 函数用于模拟在 L1 缓存块 上执行 Softmax 计算所需的 计算周期数。该函数通过计算 Softmax 操作的 浮点运算次数（FLOPs），并结合硬件的 FLOPs/周期（即每个周期能够执行的向量浮点操作数），估算出完成 Softmax 计算的 时钟周期数。
```python
def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            # online softmax
            total_flop_count = M * N * (self.flops_per_exp * 3 + 7)
            return ceil(
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )
```


### 4.2.2 layernorm_latency


## 4.3 gelu_latency


## 4.4 allreduce_latency












这两个函数分别模拟了 **Softmax** 和 **LayerNorm** 操作在 L2 缓存块上的计算周期数（**cycle count**）。尽管它们都遵循类似的分块（tiling）策略并计算总的执行周期数，但在每个操作的具体实现上有一些关键的不同，导致它们的周期数计算方式不一样。这是因为 **Softmax** 和 **LayerNorm** 的计算逻辑不同，涉及的数据传输、计算模式以及归约（reduction）方式也不同。

接下来，我们分别解释它们的区别。

---

### 1. **Softmax 的模拟**

#### 函数结构

```python
def simulate_l2_tile_compute_cycle_count(
    self,
    M: int,
    N: int,
    data_type: DataType,
    mapping: "Softmax.Mapping",
    pcb_module: Device,
):
    l1_tile_M = mapping.l1_tile_M
    l1_tile_N = mapping.l1_tile_N

    l1_tile = Softmax.L1TileSimulator(
        l1_tile_M,
        l1_tile_N,
        data_type,
        mapping,
        pcb_module,
    )
    l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
    l1_tile_cycle_count = (
        l1_tile.read_cycle_count
        + l1_tile.write_cycle_count
        + l1_tile.compute_cycle_count
    )
    total_cycle_count = (
        ceil(l1_tile_count / pcb_module.compute_module.core_count) + 1
    ) * (
        l1_tile_cycle_count
        + log2(ceil(N / l1_tile_N)) * l1_tile.reduction_cycle_count
    )
    return total_cycle_count
```

#### 1.1 **Softmax 的操作特点**

- Softmax 是一种归一化函数，通常用于将输入转换为概率分布。它的执行流程通常包括：
  1. **指数计算**：对每个输入元素计算其指数。
  2. **归约（Reduction）**：对每行或每列的指数结果求和。
  3. **归一化**：将每个元素的指数值除以总和，得到概率分布。

由于 Softmax 涉及一个 **归约操作**（求和），这会影响到计算周期数的估计。

#### 1.2 **L1 缓存块的模拟**

- **`l1_tile`** 是一个 **L1TileSimulator** 实例，用于模拟一个 L1 缓存块的行为。它负责计算当前块的读取、计算和写回延迟（`read_cycle_count`, `compute_cycle_count`, `write_cycle_count`）。

#### 1.3 **计算 L1 缓存块的数量**

```python
l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
```

- L1 缓存块的数量由矩阵的维度 `M` 和 `N` 以及分块大小 `l1_tile_M` 和 `l1_tile_N` 决定，计算公式为：
  \[
  \text{l1\_tile\_count} = \lceil M / l1\_tile\_M \rceil \times \lceil N / l1\_tile\_N \rceil
  \]

#### 1.4 **计算每个 L1 缓存块的总周期数**

```python
l1_tile_cycle_count = (
    l1_tile.read_cycle_count
    + l1_tile.write_cycle_count
    + l1_tile.compute_cycle_count
)
```

- 每个 L1 缓存块的总周期数由三部分组成：
  - **`read_cycle_count`**：读取数据的延迟。
  - **`compute_cycle_count`**：计算 Softmax 的延迟。
  - **`write_cycle_count`**：将结果写回的延迟。

#### 1.5 **计算总的 L2 缓存块的周期数**

```python
total_cycle_count = (
    ceil(l1_tile_count / pcb_module.compute_module.core_count) + 1
) * (
    l1_tile_cycle_count
    + log2(ceil(N / l1_tile_N)) * l1_tile.reduction_cycle_count
)
```

- **总周期数计算公式**：
  - **调度核心周期**：计算 L1 缓存块的数量除以硬件的核心数（`pcb_module.compute_module.core_count`），得到需要调度的批次数（batch）。加 1 是为了处理不完全分块的情况。
  - **L1 缓存块周期**：每个批次的周期数等于 L1 缓存块的总周期数。
  - **归约的计算复杂度**：Softmax 需要在最后一维（`N` 维）上进行归约操作，因此增加了一个额外的归约延迟，使用 `log2` 模拟归约的复杂度。

---

### 2. **LayerNorm 的模拟**

#### 函数结构

```python
def simulate_l2_tile_compute_cycle_count(
    self,
    M: int,
    N: int,
    data_type: DataType,
    mapping: "LayerNorm.Mapping",
    pcb_module: Device,
):
    l1_tile_M = mapping.l1_tile_M
    l1_tile_N = mapping.l1_tile_N

    l1_tile = LayerNorm.L1TileSimulator(
        l1_tile_M,
        l1_tile_N,
        data_type,
        mapping,
        pcb_module,
    )
    l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
    l1_tile_cycle_count = (
        l1_tile.read_cycle_count * 3
        + l1_tile.write_cycle_count
        + l1_tile.compute_cycle_count
    )
    total_cycle_count = (
        ceil(l1_tile_count / pcb_module.compute_module.core_count)
    ) * (
        l1_tile_cycle_count
        + (ceil(N / l1_tile_N) - 1) * l1_tile.reduction_cycle_count
    )
    return total_cycle_count
```

#### 2.1 **LayerNorm 的操作特点**

- **LayerNorm** 是一种常见的归一化技术，通过对输入的每一层（通常是最后一个维度）进行标准化。与 Softmax 类似，它也有一个归约操作，但 LayerNorm 需要计算均值和方差，并进行归一化。
  - **LayerNorm 的计算步骤**：
    1. **计算均值**。
    2. **计算方差**。
    3. **归一化**。
  - 由于 LayerNorm 需要读取数据多次（至少 3 次——一次用于均值计算，一次用于方差计算，一次用于归一化），它的 I/O 开销比 Softmax 更大。

#### 2.2 **L1 缓存块的模拟**

- **`l1_tile`** 是一个 **LayerNorm.L1TileSimulator** 实例，用于模拟每个 L1 缓存块的行为。

#### 2.3 **计算 L1 缓存块的数量**

```python
l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
```

- L1 缓存块的数量与 Softmax 相同，仍然是通过 `M` 和 `N` 维度的分块大小计算得到。

#### 2.4 **计算每个 L1 缓存块的总周期数**

```python
l1_tile_cycle_count = (
    l1_tile.read_cycle_count * 3
    + l1_tile.write_cycle_count
    + l1_tile.compute_cycle_count
)
```

- **I/O 开销更大**：与 Softmax 不同，LayerNorm 的读取操作需要执行 3 次（一次用于均值计算、一次用于方差计算、一次用于标准化），因此 `read_cycle_count` 需要乘以 3。
  - **`read_cycle_count * 3`**：读取数据的延迟（因为读取操作会发生 3 次）。
  - **`write_cycle_count`**：写回数据的延迟。
  - **`compute_cycle_count`**：计算 LayerNorm 的延迟。

#### 2.5 **计算总的 L2 缓存块的周期数**

```python
total_cycle_count = (
    ceil(l1_tile_count / pcb_module.compute_module.core_count)
) * (
    l1_tile_cycle_count
    + (ceil(N / l1_tile_N) - 1) * l1_tile.reduction_cycle_count
)
```

- **总周期数计算公式**：
  - 与 Softmax 类似，LayerNorm 也需要计算归约操作的延迟，但归约操作的复杂度不同于 Softmax。
  - 计算归约延迟时，LayerNorm 使用：
    \[
    (\lceil N / l1\_tile\_N \rceil - 1) \times l1\_tile.reduction\_cycle\_count
    \]
  - **读取操作更多**：由于 LayerNorm 的读取延迟更大（`read_cycle_count * 3`），其总的 I/O 开销比 Softmax 要大。

---

### 3. **Softmax 与 LayerNorm 模拟的主要区别**

1. **读取操作的次数不同**：
   - **Softmax**：每个 L1 缓存块只需要一次读取操作。
   - **LayerNorm**：每个 L1 缓存块需要 3 次读取操作（一次用于均值计算、一次用于方差计算、一次用于归一化），因此读取延迟更大。

2. **归约操作的复杂度不同**：
   - **Softmax**：归约操作的复杂度用 `log2` 来近似，表示指数求和操作。
   - **LayerNorm**：归约操作的复杂度较低，因此直接用 `ceil(N / l1_tile_N) - 1` 来表示。

3. **总的 I/O 和计算开销不同**：
   - **LayerNorm** 的总开销更大，主要是因为它涉及多次读取和更复杂的计算步骤，因此其周期数计算公式中的 `read_cycle_count` 乘以 3。

---

### 4. **总结**

- **Softmax** 的计算开销主要来自于一次读取、一次写回和一次归约（`log2` 复杂度）的计算。
- **LayerNorm** 的计算开销更大，涉及 3 次读取操作、一次写回和一次较简单的归约操作，但其 I/O 延迟显著增加。

这两种操作的模拟周期数计算方式不同，主要是由于它们在计算逻辑上的差异导致的不同的 I/O 和计算开销。






