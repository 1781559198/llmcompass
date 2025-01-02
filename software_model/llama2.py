from software_model.operators import (
    Operator,
    Reshape,
    Concat,
    Transpose,
)
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.silu import SiLU
from software_model.rmsnorm import RMSNorm
from software_model.rope import RoPE
from software_model.casual_mask import CausalMask
from software_model.add import Add
from software_model.element_wise_multiply import ElementWiseMultiply

from software_model.utils import Tensor, DataType
from software_model.communication_primitives import AllReduceMultiPCB
from math import ceil
from typing import List
from hardware_model.system import System

import torch

class Llama2BlockInitComputationTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        # d_model: 模型的隐藏层维度，也就是输入张量的特征维度
        # n_heads: 注意力机制中的头数（多头注意力机制）
        # device_count: 分布式计算中的设备数量，通常表示并行计算的设备数量
        # data_type: 数据类型（DataType），用于定义张量的数据类型，如 float32 或 float16
        super().__init__(0, 0, 0, 0, data_type)# 调用父类函数Operator
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        # parameters per device
        d = d_model
        # 分别初始化了多头自注意力机制中的 查询（Query）、键（Key） 和 值（Value） 投影矩阵
        # d：输入维度（即 d_model），表示模型的隐藏层维度。
        # d // device_count：表示该设备上负责的权重部分。假设有 device_count 个设备，模型的权重被划分成 device_count 份，每个设备负责一部分
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
        # operators per device
        # multi-head attention
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
        self.causal_mask = CausalMask(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.add = Add(data_type)
        self.rmsnorm0 = RMSNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        self.RoPE = RoPE(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_silu = SiLU(data_type)
        self.element_wise_multiply = ElementWiseMultiply(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.H_matmul3 = Matmul(data_type)
        self.rmsnorm1 = RMSNorm(data_type)
        self.rope = RoPE(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, s, d = X.shape
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        # multi-head attention


        # Matmul:Q_K_V：通过矩阵乘法将输入 X 投影为查询（Q）、键（K）、值（V）
        Q = self.Q_proj(X, self.Wq)  # [b, s, d / dev_cnt]
        assert Q.shape == [b, s, d // dev_cnt]
        K = self.K_proj(X, self.Wk)  # [b, s, d / dev_cnt]
        V = self.V_proj(X, self.Wv)  # [b, s, d / dev_cnt]
        Q = self.rope(Q, self.Wq)
        K = self.rope(K, self.Wk)   
        Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
        K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
        V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])

        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
        assert Q_T.shape == [b, h // dev_cnt, s, d_h]
        K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, s]
        assert K_T.shape == [b, h // dev_cnt, d_h, s]
        V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
        assert V_T.shape == [b, h // dev_cnt, s, d_h] 

        # Matmul:Q_mul_K：将查询矩阵 Q_T 和键矩阵 K_T 相乘，生成注意力权重
        A = self.Q_mul_K(Q_T, K_T)  # [b, h / dev_cnt, s, s]
        assert A.shape == [b, h // dev_cnt, s, s]
        # Softmax：对注意力权重矩阵 A 应用 softmax，生成归一化的注意力权重
        A = self.causal_mask(A)
        assert A.shape == [b, h // dev_cnt, s, s]
        A_prob = self.A_softmax(A)
        # Matmul:A_mul_V：将归一化的注意力权重与值矩阵 V_T 相乘，生成注意力机制的输出
        H = self.A_mul_V(A_prob, V_T)  #  [b, h / dev_cnt, s, d_h]
        assert H.shape == [b, h // dev_cnt, s, d_h]
        H = self.H_transpose(H, [0, 2, 1, 3])  #  [b, s, h / dev_cnt, d_h]
        assert H.shape == [b, s, h // dev_cnt, d_h]
        H = self.H_reshape(H, [b, s, d // dev_cnt])
        assert H.shape == [b, s, d // dev_cnt]
        # Matmul:Wo_proj：多头自注意力的输出投影，用于将多头注意力的输出重新投影到原始的隐藏层维度
        H0 = self.H_matmul0(H, self.W0)  #  [b, s, d]
        assert H0.shape == [b, s, d]
        # LayerNorm - MHA：多头注意力机制后的层归一化操作
        H0 = self.add(H0, X)
        assert H0.shape == [b, s, d]

        H0 = self.rmsnorm0(H0)
        assert H0.shape == [b, s, d]
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)


        # feed-forward network

        # Matmul:W1_proj：前馈神经网络的第一层，全连接层 W1 将输入的隐藏层维度扩展到 4 倍。对应代码中的 self.H_matmul1
        H1 = self.H_matmul1(H0, self.W1)  # [b, s, 4 * d / dev_cnt]
        assert H1.shape == [b, s, 4 * d // dev_cnt]

        # gate, SiLU   
        H2 = self.H_matmul2(H0, self.W1)  # [b, 1, 4 * d / dev_cnt]
        assert H2.shape == [b, 1, 4 * d // dev_cnt]
        H3 = self.H_silu(H2)   

        H4 = self.element_wise_multiply(H1, H3)  # [b, 1, 4 * d / dev_cnt]

        H5 = self.H_matmul3(H4, self.W2)  #  [b, 1, d]
        assert H5.shape == [b, 1, d]

        # Residual connection
        H6 = self.add(H0, H5)
        assert H6.shape == [b, 1, d]
        
        H6 = self.rmsnorm1(H6)
        if dev_cnt > 1:
            H6 = self.allreduce_ffn(H6)
        assert H6.shape == [b, 1, d]

        return H6

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        casual_mask_latency = (
            self.causal_mask.roofline_model(device)
            + device.compute_module.overhead.casual_mask
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + casual_mask_latency
        )

        rope_latency = (
            self.rope.roofline_model(device)
            + device.compute_module.overhead.rope
        )
        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        rmsnorm_latency = (
            self.rmsnorm0.roofline_model(device)
            + device.compute_module.overhead.rmsnorm
        )
        add_latency = (
            self.add.roofline_model(device)
            + device.compute_module.overhead.add
        )
        normlization_total_latency = softmax_latency + rmsnorm_latency * 3 + add_latency * 3

        # silu
        silu_latency = (
            self.H_silu.roofline_model(device)
            + device.compute_module.overhead.silu
        )
        element_wise_multiply_latency = (
            self.element_wise_multiply.roofline_model(device)
            + device.compute_module.overhead.element_wise_multiply
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        '''
        print("Roofline breakdown:")
        print(  
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{rope_latency}\n{softmax_latency}\n{add_latency}\n{add_latency}\n{add_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{silu_latency}\n{element_wise_multiply_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        self.roofline_log =  f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{rope_latency}\n{softmax_latency}\n{add_latency}\n{add_latency}\n{add_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{silu_latency}\n{element_wise_multiply_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{silu_latency}\n{element_wise_multiply_latency}\n{allreduce_total_latency}\n"
        )'''
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + silu_latency
            + element_wise_multiply_latency
            + allreduce_total_latency
        )
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str):
        device = system.device
        interconnect = system.interconnect

        # matmul
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        casual_mask_latency = (
            self.causal_mask.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.casual_mask
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + casual_mask_latency
        )

        rope_latency = (
            self.rope.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.rope
        )
        # normalization
        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )
        rmsnorm_latency = (
            self.rmsnorm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.rmsnorm
        )
        add_latency = (
            self.add.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.add
        )   
        normlization_total_latency = softmax_latency + rmsnorm_latency * 3 + add_latency * 3

        # silu
        silu_latency = (
            self.H_silu.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.silu
        )
        element_wise_multiply_latency = (
            self.element_wise_multiply.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.element_wise_multiply
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + silu_latency
            + element_wise_multiply_latency
            + allreduce_total_latency
        )
        self.simluate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {casual_mask_latency},{rope_latency}, {softmax_latency}, {add_latency}, {add_latency}, {add_latency}, {rmsnorm_latency},{rmsnorm_latency}, {rmsnorm_latency}, {silu_latency}, {element_wise_multiply_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency

    def run_on_gpu(self):
        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
        ) * 3
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )
        rope_latency = (
            self.rope.run_on_gpu()  # - self.rope.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + rope_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        rmsnorm_latency = (
            self.rmsnorm0.run_on_gpu()
            - self.rmsnorm0.gpu_kernel_launch_overhead()
        )
        add_latency = ( 
            self.add.run_on_gpu()
            - self.add.gpu_kernel_launch_overhead()
        )
        normlization_total_latency = softmax_latency + rmsnorm_latency * 3 + add_latency * 3

        # silu
        silu_latency = (
            self.H_silu.run_on_gpu()  # - self.H_silu.gpu_kernel_launch_overhead()
        )
        element_wise_multiply_latency = (
            self.element_wise_multiply.run_on_gpu()
            - self.element_wise_multiply.gpu_kernel_launch_overhead()
        )

        # allreduce 
        allreduce_total_latency = 0

        # others

        # print
        '''
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{rope_latency}\n{softmax_latency}\n{add_latency}\n{add_latency}\n{add_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{silu_latency}\n{element_wise_multiply_latency}\n{allreduce_total_latency}\n"
        )
        print("total:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{rope_latency}\n{softmax_latency}\n{add_latency}\n{add_latency}\n{add_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{silu_latency}\n{element_wise_multiply_latency}\n{allreduce_total_latency}\n"
        )'''
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + silu_latency
            + element_wise_multiply_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu


class Llama2BlockAutoRegressionTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        # parameters per device   计算类型大小
        d = d_model# 模型维度
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
        # operators per device
        # # multi-head attention

        # matmul 用于矩阵乘法操作，通常在计算查询（Q）、键（K）、值（V）时使用
        # reshape 用于调整张量的形状，以便后续的计算
        # transpone 用于转置张量的维度，以便进行适当的矩阵乘法
        # batchedmatmul 用于批量矩阵乘法，通常用于计算注意力权重等
        # softmax 用于批量矩阵乘法，通常用于计算注意力权重等
        # layernorm 用于层归一化，帮助模型收敛
        # allreducemultipcb 用于层归一化，帮助模型收敛
        # gelu 用于激活函数，常用于前馈神经网络中
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
        self.causal_mask = CausalMask(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.add = Add(data_type)
        self.rmsnorm0 = RMSNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        self.RoPE = RoPE(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_silu = SiLU(data_type)
        self.element_wise_multiply = ElementWiseMultiply(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.H_matmul3 = Matmul(data_type)
        self.rmsnorm1 = RMSNorm(data_type)
        self.rope = RoPE(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        # b: batch size 批次大小    8
        # s: sequence length 序列长度 122888
        # d: hidden dimension 隐藏维度
        # d_h: dimension per head 
        b, _, d = x.shape 
        assert d == self.d_model # 12288
        s = seq_len
        h = self.n_heads# 多头注意力的头数 96
        dev_cnt = self.device_count# 设备数量 1
        d_h = d // h# 每个头的维度 128
        
        # KV cache
        K_cache = Tensor([b, h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, h // dev_cnt, s, d_h], self.data_type)

        x = self.rmsnorm0(x)# rmsnorm

        # multi-head attention
        # 通过线性变换生成qkv向量
        q = self.Q_proj(x, self.Wq)  # [b, 1, d / dev_cnt]
        assert q.shape == [b, 1, d // dev_cnt]
        k = self.K_proj(x, self.Wk)  # [b, 1, d / dev_cnt]
        v = self.V_proj(x, self.Wv)  # [b, 1, d / dev_cnt]

        # RoPE旋转向量
        q = self.rope(q, self.Wq)
        k = self.rope(k, self.Wk)

        # 对qkv重新调整状态，适应多头注意力机制
        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])# 批次大小，序列长度，每个设备上的头数，每个头的维度
        k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])

        # 对qkv进行维度转置操作
        q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert q_T.shape == [b, h // dev_cnt, 1, d_h]
        k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, 1]
        assert k_T.shape == [b, h // dev_cnt, d_h, 1]
        v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert v_T.shape == [b, h // dev_cnt, 1, d_h]

        K_T = self.K_concat(K_cache, k_T, 3)  # [b, h / dev_cnt, d_h, s+1]
        assert K_T.shape == [b, h // dev_cnt, d_h, s + 1]
        V_T = self.V_concat(V_cache, v_T, 2)  # [b, h / dev_cnt, s+1, d_h]
        assert V_T.shape == [b, h // dev_cnt, s + 1, d_h]

        a = self.Q_mul_K(q_T, K_T)  # [b, h / dev_cnt, 1, s+1]
        assert a.shape == [b, h // dev_cnt, 1, s + 1]
        a = self.causal_mask(a)  # 应用因果掩码
        assert a.shape == [b, h // dev_cnt, 1, s + 1]
        a_prob = self.A_softmax(a)

        # 多头注意力
        h0 = self.A_mul_V(a_prob, V_T)  #  [b, h / dev_cnt, 1, d_h]
        assert h0.shape == [b, h // dev_cnt, 1, d_h]
        h0 = self.H_transpose(h0, [0, 2, 1, 3])  #  [b, 1, h / dev_cnt, d_h]
        assert h0.shape == [b, 1, h // dev_cnt, d_h]
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        assert h0.shape == [b, 1, d // dev_cnt]
        h0 = self.H_matmul0(h0, self.W0)  #  [b, 1, d]
        assert h0.shape == [b, 1, d]

        h0 = self.add(h0, x)
        assert h0.shape == [b, 1, d]

        h0 = self.rmsnorm0(h0)
        assert h0.shape == [b, 1, d]
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)


        # feed-forward network
        # up
        h1 = self.H_matmul1(h0, self.W1)  # [b, 1, 4 * d / dev_cnt]
        assert h1.shape == [b, 1, 4 * d // dev_cnt]
        
        # gate, SiLU   
        h2 = self.H_matmul2(h0, self.W1)  # [b, 1, 4 * d / dev_cnt]
        assert h2.shape == [b, 1, 4 * d // dev_cnt]
        h3 = self.H_silu(h2)   

        h4 = self.element_wise_multiply(h1, h3)  # [b, 1, 4 * d / dev_cnt]

        h5 = self.H_matmul3(h4, self.W2)  #  [b, 1, d]
        assert h5.shape == [b, 1, d]

        # Residual connection
        h6 = self.add(h0, h5)
        assert h6.shape == [b, 1, d]
        
        h6 = self.rmsnorm1(h6)
        if dev_cnt > 1:
            h6 = self.allreduce_ffn(h6)

        assert h6.shape == [b, 1, d]
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
        return h6

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        casual_mask_latency = (
            self.causal_mask.roofline_model(device)
            + device.compute_module.overhead.casual_mask
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + casual_mask_latency
        )

        rope_latency = (
            self.rope.roofline_model(device)
            + device.compute_module.overhead.rope
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        rmsnorm_latency = (
            self.rmsnorm0.roofline_model(device)
            + device.compute_module.overhead.rmsnorm
        )
        add_latency = (
            self.add.roofline_model(device)
            + device.compute_module.overhead.add
        )

        normlization_total_latency = softmax_latency + rmsnorm_latency * 3 + add_latency * 3

        # silu  
        silu_latency = (
            self.H_silu.roofline_model(device) 
            + device.compute_module.overhead.silu
        )
        element_wise_multiply_latency = (
            self.element_wise_multiply.roofline_model(device)
            + device.compute_module.overhead.element_wise_multiply
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        '''
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{rope_latency}\n{softmax_latency}\n{add_latency}\n{add_latency}\n{add_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{silu_latency}\n{element_wise_multiply_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{silu_latency}\n{allreduce_total_latency}\n"
        )'''
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + silu_latency
            + allreduce_total_latency
        )
        # print(f'memory requirement: {self.memory_requirement/1e9*96}GB')
        self.roofline_log =  f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{rope_latency}\n{softmax_latency}\n{add_latency}\n{add_latency}\n{add_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{silu_latency}\n{element_wise_multiply_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str):
        pcb = system.device
        interconnect = system.interconnect

        # matmul
        # print("simulating qkv")
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating q_mul_k")
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating a_mul_v")
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        casual_mask_latency = (
            self.causal_mask.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.casual_mask
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + casual_mask_latency
        )

        rope_latency = (
            self.rope.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.rope
        )

        # normalization   执行时间加硬件开销
        softmax_latency = (
            self.A_softmax.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.softmax
        )
        rmsnorm_latency = (
            self.rmsnorm1.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.rmsnorm   # 自定义
        )
        add_latency = (
            self.add.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.add
        )

        normlization_total_latency = softmax_latency + rmsnorm_latency * 3 + add_latency * 3

        # silu
        silu_latency = (
            self.H_silu.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.silu # 自定义
        )
        element_wise_multiply_latency = (
            self.element_wise_multiply.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.element_wise_multiply
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + silu_latency
            + element_wise_multiply_latency
            + allreduce_total_latency
        )
        self.simluate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {casual_mask_latency},{rope_latency}, {softmax_latency}, {add_latency}, {add_latency}, {add_latency}, {rmsnorm_latency},{rmsnorm_latency}, {rmsnorm_latency}, {silu_latency}, {element_wise_multiply_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency

    def run_on_gpu(self):
        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
        ) * 3
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )
        rope_latency = (
            self.rope.run_on_gpu()  # - self.rope.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + rope_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        rmsnorm_latency = (
            self.rmsnorm0.run_on_gpu()
            - self.rmsnorm0.gpu_kernel_launch_overhead()
        )
        add_latency = ( 
            self.add.run_on_gpu()
            - self.add.gpu_kernel_launch_overhead()
        )
        normlization_total_latency = softmax_latency + rmsnorm_latency * 3 + add_latency * 3

        # silu
        silu_latency = (
            self.H_silu.run_on_gpu()  # - self.H_silu.gpu_kernel_launch_overhead()
        )
        element_wise_multiply_latency = (
            self.element_wise_multiply.run_on_gpu()
            - self.element_wise_multiply.gpu_kernel_launch_overhead()
        )
        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{rope_latency}\n{softmax_latency}\n{add_latency}\n{add_latency}\n{add_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{silu_latency}\n{element_wise_multiply_latency}\n{allreduce_total_latency}\n"
        )
        print("total:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{rope_latency}\n{softmax_latency}\n{add_latency}\n{add_latency}\n{add_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{rmsnorm_latency}\n{silu_latency}\n{element_wise_multiply_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + silu_latency
            + element_wise_multiply_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu


class LLMInitComputationTP:
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        device_count,
    ) -> None:
        pass
