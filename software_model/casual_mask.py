from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil, log2, log
import time
import statistics
import numpy as np
import torch


# Define the causal mask function for GPU execution
@torch.compile
def causal_mask_gpu(input: torch.Tensor) -> torch.Tensor:
    seq_len = input.shape[-1]
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device='cuda')).unsqueeze(0)
    return input * causal_mask


class CausalMask(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None  # Shape of the input tensor
        self.M = None      # Total number of elements involved
        self.N = None      # Sequence length
        self.B = None      # Batch size

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type, "Data types must match."
        assert len(input.shape) >= 2, "Input tensor must have at least 2 dimensions (batch_size, seq_length)."
        self.shape = input.shape
        self.B = input.shape[0]
        self.N = input.shape[-1]
        self.M = size(input.shape)  # Total number of elements in the input tensor

        # Set up the computational graph
        self.computational_graph = self.ComputationalGraph(self.B, self.N, self.data_type)

        # Return the input tensor (the actual masking operation would modify the input in-place or return a new tensor)
        return input

    def roofline_model(self, pcb_module: Device):
        # Adjust data type based on the device's capabilities
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        B = self.B
        N = self.N
        data_type = self.computational_graph.data_type

        # Calculate total IO count (reads and writes)
        total_io_count = B * N * N * data_type.word_size * 2  # Read input tensor and write output tensor
        io_latency = (
            total_io_count / min(pcb_module.io_module.bandwidth,
                                 pcb_module.compute_module.l2_bandwidth_per_cycle
                                 * pcb_module.compute_module.clock_freq)
        )

        # Calculate total flop count (element-wise multiplication with the mask)
        total_flop_count = B * N * N  # One multiplication per element
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )
        self.roofline_latency = max(compute_latency, io_latency)
        return self.roofline_latency

    def print_latency(self):
        print(f"Input shape: {self.shape}, Latency on GPU: {self.latency_on_gpu * 1e6:.2f} μs")

    class ComputationalGraph:
        def __init__(self, B: int, N: int, data_type: DataType):
            self.B = B  # Batch size
            self.N = N  # Sequence length
            self.data_type = data_type

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        # Adjust data type based on the device's capabilities
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        parallelism = (# parallelism 表示设备在一个时钟周期中可以并行处理的总数据元素数量
            pcb_module.compute_module.core_count
            * pcb_module.compute_module.core.vector_unit.vector_width
            * pcb_module.compute_module.core.vector_unit.vector_count
        )
        total_elements = self.B * self.N * self.N
        M = ceil(total_elements / parallelism) * parallelism# 对齐后的总元素
        data_type = self.computational_graph.data_type

        # Calculate total IO count (reads and writes)
        total_io_count = M * data_type.word_size * 2  # 读取输入张量和写入输出张量
        io_latency = (
            total_io_count / pcb_module.io_module.bandwidth
            + total_io_count
            / pcb_module.compute_module.l2_bandwidth_per_cycle
            / pcb_module.compute_module.clock_freq
        )

        # Calculate total flop count (element-wise multiplication)
        total_flop_count = M  # One multiplication per element
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )

        return max(compute_latency, io_latency)

    def run_on_gpu(self):
        assert self.shape is not None, "Input shape must be defined."
        input = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []

        # Warmup iterations
        for _ in range(3):
            _ = causal_mask_gpu(input)
            torch.cuda.synchronize()

        # Timed iterations
        for _ in range(self.iterations):
            start = time.time()
            output = causal_mask_gpu(input)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape == input.shape, "Output shape must match input shape."
            latencies.append(end - start)

        self.latency_on_gpu = statistics.median(latencies)
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = causal_mask_gpu(a)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # Print or return the kernel launch overhead
        print("GPU kernel launch overhead (median): {:.6f} ms".format(avg_overhead * 1e3))
        # print(latencies)
        return avg_overhead