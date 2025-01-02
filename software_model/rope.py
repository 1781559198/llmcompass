from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil, log2
import time
import statistics
import numpy as np
import torch
import math


@torch.compile
def rope_gpu(input: torch.Tensor, sin_emb: torch.Tensor, cos_emb: torch.Tensor) -> torch.Tensor:
    return (input * cos_emb) + (rotate_half(input) * sin_emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


class RoPE(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor, position: int) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:-1])
        self.N = input.shape[-1]
        self.position = position
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type, self.position
        )
        return input

    def roofline_model(self, pcb_module: Device):
        # 计算 I/O 次数和 FLOP 次数
        # RoPE 需要计算 sin 和 cos，以及元素级别的乘法和加法
        self.io_count = self.M * self.N * self.data_type.word_size * 3  # 输入 x, sin_emb, cos_emb
        self.flop_count = self.M * self.N * 6  # sin, cos, multiply, multiply, add, add
        self.roofline_latency = max(
            self.io_count
            / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle
                * pcb_module.compute_module.clock_freq,
            ),
            self.flop_count / pcb_module.compute_module.total_vector_flops,
        )
        return self.roofline_latency

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu * 1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, N: int, data_type: DataType, position: int):
            self.M = M
            self.N = N
            self.data_type = data_type
            self.position = position

    class Mapping:
        def __init__(  # 表示和管理内存映射配置
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            l1_tile_M: int,
            l1_tile_N: int,
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N

        def display(self):
            print("-" * 20)
            print(
                f"l2_tile_M: {self.l2_tile_M}, l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}"
            )

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        min_cycle_count = float("inf")
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        data_type = self.computational_graph.data_type
        position = self.computational_graph.position

        # 确定 L2 切片因子
        l2_tile_N = N
        l2_tile_M = (
            pcb_module.compute_module.l2_size // (l2_tile_N * data_type.word_size) // 2
        )
        l2_tile_M = min(l2_tile_M, M)

        if compile_mode in ["heuristic-GPU", "heuristic-our-throughput", "yizhu-g100"]:
            l1_tile_N = N
            l1_tile_M = (
                pcb_module.compute_module.core.SRAM_size
                // (l1_tile_N * data_type.word_size)
                // 2
            )
            while l1_tile_M < pcb_module.compute_module.core.vector_unit.vector_count:
                l1_tile_N = l1_tile_N // 2
                l1_tile_M = (
                    pcb_module.compute_module.core.SRAM_size
                    // (l1_tile_N * data_type.word_size)
                    // 2
                )
            l1_tile_M = min(l1_tile_M, l2_tile_M)
        elif compile_mode == "heuristic-TPU":
            l1_tile_N = N
            l1_tile_M = pcb_module.compute_module.core.SRAM_size // (
                2 * l1_tile_N * data_type.word_size
            )
            l1_tile_M = min(l1_tile_M, M)
        else:
            # 默认配置，您可以根据需要调整
            l1_tile_N = ceil(N / 2)
            l1_tile_M = ceil(M / 2)

        mapping = self.Mapping(
            l2_tile_M,
            l2_tile_N,
            l1_tile_M,
            l1_tile_N,
        )
        cycle_count = self.simulate(self.computational_graph, mapping, pcb_module)
        if cycle_count < min_cycle_count:
            min_cycle_count = cycle_count
            best_mapping = mapping
        self.best_mapping = best_mapping
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # self.best_mapping.display()
        return self.latency

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

        M_l2_t = M // l2_tile_M
        M_remain = M % l2_tile_M

        l2_tiles = np.empty([ceil(M / l2_tile_M)], dtype=object)

        if M_l2_t != 0:
            for i in range(M_l2_t):
                l2_tiles[i] = self.L2TileSimulator(
                    l2_tile_M,
                    N,
                    data_type,
                    mapping,
                    pcb_module,
                )
        if M_remain != 0:
            l2_tiles[-1] = self.L2TileSimulator(
                M_remain,
                N,
                data_type,
                mapping,
                pcb_module,
            )

        total_cycle_count = 0
        l2_tile_count = ceil(M / l2_tile_M)
        for m in range(l2_tile_count):
            total_cycle_count += l2_tiles[m].read_cycle_count
            total_cycle_count += l2_tiles[m].compute_cycle_count
            total_cycle_count += l2_tiles[m].write_cycle_count
        return total_cycle_count

    class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "RoPE.Mapping",
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

        def simulate_l2_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, chiplet_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (
                    chiplet_module.io_module.bandwidth
                    / chiplet_module.compute_module.clock_freq  # 硬件时钟频率
                )
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "RoPE.Mapping",
            pcb_module: Device,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = RoPE.L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                data_type,
                mapping,
                pcb_module,
            )
            l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
            l1_tile_cycle_count = (
                l1_tile.read_cycle_count * 2  # 读取 x 和 sin/cos embedding
                + l1_tile.write_cycle_count
                + l1_tile.compute_cycle_count
            )
            total_cycle_count = (
                ceil(l1_tile_count / pcb_module.compute_module.core_count)
            ) * (
                l1_tile_cycle_count
            )
            return total_cycle_count

    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "RoPE.Mapping",
            pcb_module: Device,
        ):
            self.M = M
            self.N = N
            self.read_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.write_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.reduction_cycle_count = (
                0# rope函数没有规约操作
            )

        def simulate_l1_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, pcb_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (pcb_module.compute_module.l2_bandwidth_per_cycle)
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "RoPE.Mapping",
            pcb_module: Device,
        ):
            # RoPE 主要涉及 sin、cos 计算和元素级别的乘加操作
            flops_per_element = 4  # sin, cos, multiply, add
            total_flop_count = M * N * flops_per_element
            return ceil(
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )

    def run_on_gpu(self):
        assert self.shape is not None
        input = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        # 生成 sin 和 cos 的位置编码
        position = self.position
        dim = self.N
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
        sinusoid = torch.einsum("i,j->ij", torch.arange(position, device="cuda").float(), inv_freq)
        sin_emb = torch.sin(sinusoid).unsqueeze(0).expand_as(input)
        cos_emb = torch.cos(sinusoid).unsqueeze(0).expand_as(input)

        latencies = []

        # 预热
        for _ in range(3):
            _ = rope_gpu(input, sin_emb, cos_emb)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = rope_gpu(input, sin_emb, cos_emb)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape == input.shape
            latencies.append(end - start)
        self.latency_on_gpu = statistics.median(latencies)
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        import torch

        latencies = []
        a = torch.randn(1, 1, 1, device="cuda")
        sin_emb = torch.sin(torch.tensor([[0.0]], device="cuda"))
        cos_emb = torch.cos(torch.tensor([[0.0]], device="cuda"))
        for _ in range(50):
            start = time.time()
            c = rope_gpu(a, sin_emb, cos_emb)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print(latencies)
        return avg_overhead