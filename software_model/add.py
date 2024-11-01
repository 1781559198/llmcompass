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

# Define the addition function for GPU execution
@torch.compile
def add_gpu(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    return input1 + input2

class Add(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None  # Shape of the input tensors

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        assert self.data_type == input1.data_type == input2.data_type, "Data types must match."
        assert input1.shape == input2.shape, "Input tensors must have the same shape."
        self.shape = input1.shape
        self.M = size(self.shape)  # Total number of elements
        self.computational_graph = self.ComputationalGraph(self.M, self.data_type)
        return Tensor(shape=self.shape, data_type=self.data_type)

    def roofline_model(self, pcb_module: Device):
        # Set data type based on the device's capabilities
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        M = self.M
        data_type = self.computational_graph.data_type

        # Calculate total IO count (reads and writes)
        total_io_count = M * 3 * data_type.word_size  # Read input1, input2 and write output
        io_latency = (
            total_io_count / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle * pcb_module.compute_module.clock_freq
            )
        )

        # Calculate total flop count (one addition per element)
        total_flop_count = M  # One addition per element
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
        def __init__(self, M: int, data_type: DataType):
            self.M = M
            self.data_type = data_type

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        # Set data type based on the device's capabilities
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        parallelism = (
            pcb_module.compute_module.core_count
            * pcb_module.compute_module.core.vector_unit.vector_width
            * pcb_module.compute_module.core.vector_unit.vector_count
        )
        M = ceil(self.computational_graph.M / parallelism) * parallelism
        data_type = self.computational_graph.data_type

        # Calculate total IO count (reads and writes)
        total_io_count = M * 3 * data_type.word_size  # Read input1, input2 and write output
        io_latency = (
            total_io_count / pcb_module.io_module.bandwidth
            + total_io_count
            / pcb_module.compute_module.l2_bandwidth_per_cycle
            / pcb_module.compute_module.clock_freq
        )

        # Calculate total flop count (one addition per element)
        total_flop_count = M  # One addition per element
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )

        return max(compute_latency, io_latency)

    def run_on_gpu(self):
        assert self.shape is not None, "Input shape must be defined."
        input1 = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        input2 = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []

        # Warmup iterations
        for _ in range(3):
            _ = add_gpu(input1, input2)
            torch.cuda.synchronize()

        # Timed iterations
        for _ in range(self.iterations):
            start = time.time()
            output = add_gpu(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape == self.shape, "Output shape must match input shape."
            latencies.append(end - start)

        self.latency_on_gpu = statistics.median(latencies)
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            b = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = add_gpu(a, b)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # Print or return the kernel launch overhead
        print("GPU kernel launch overhead (median): {:.6f} ms".format(avg_overhead * 1e3))
        # print(latencies)
        return avg_overhead