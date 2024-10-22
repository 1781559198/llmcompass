from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil, log2
import torch
import time
import statistics
import numpy as np


class CausalMask(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:-1])  # 批次维度的乘积
        self.N = input.shape[-1]         # 序列长度
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type
        )

        # mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)*float('-inf')
        # input = input + mask
        return input

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu * 1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, N: int, data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

    class Mapping:
        def __init__(
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            is_l2_double_buffering: bool,
            l1_tile_M: int,
            l1_tile_N: int,
            is_l1_double_buffering: bool = False,
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N
            self.is_l1_double_buffering = is_l1_double_buffering

        def display(self):
            print("-" * 20)
            print(
                f"l2_tile_M: {self.l2_tile_M}, is_l2_double_buffering: {self.is_l2_double_buffering}, "
                f"l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, "
                f"is_l1_double_buffering: {self.is_l1_double_buffering}"
            )

    def roofline_model(self, pcb_module: Device):
        self.io_count = self.M * self.N * self.data_type.word_size * 2  # 读取和写入
        self.flop_count = self.M * self.N  # 每个元素一次操作

        io_bandwidth = min(
            pcb_module.io_module.bandwidth,
            pcb_module.compute_module.l2_bandwidth_per_cycle * pcb_module.compute_module.clock_freq,
        )
        compute_throughput = pcb_module.compute_module.total_vector_flops

        io_latency = self.io_count / io_bandwidth
        compute_latency = self.flop_count / compute_throughput

        self.roofline_latency = max(io_latency, compute_latency)
        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode=None):
        self.computational_graph.data_type = pcb_module.compute_module.core.vector_unit.data_type
        min_cycle_count = float("inf")
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        data_type = self.computational_graph.data_type

        l2_tile_N = N
        l2_tile_M = (
            pcb_module.compute_module.l2_size // (l2_tile_N * data_type.word_size)
        )
        l2_tile_M = min(l2_tile_M, M)
        is_l2_double_buffering = False

        for l1_N_tiling_factor in [1, 2, 4, 8, 16, 32]:
            l1_tile_N = ceil(l2_tile_N / l1_N_tiling_factor)
            for l1_tile_M in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for is_l1_double_buffering in [True, False]:
                    l1_sram_requirement = l1_tile_M * l1_tile_N * data_type.word_size
                    l1_sram_limit = (
                        pcb_module.compute_module.core.SRAM_size // 2
                        if is_l1_double_buffering
                        else pcb_module.compute_module.core.SRAM_size
                    )
                    if l1_sram_requirement > l1_sram_limit:
                        continue
                    mapping = self.Mapping(
                        l2_tile_M,
                        l2_tile_N,
                        is_l2_double_buffering,
                        l1_tile_M,
                        l1_tile_N,
                        is_l1_double_buffering,
                    )
                    cycle_count = self.simulate(
                        self.computational_graph, mapping, pcb_module
                    )
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

        l2_sram_requirement = l2_tile_M * N * data_type.word_size
        l2_sram_limit = (
            pcb_module.compute_module.l2_size // 2
            if mapping.is_l2_double_buffering
            else pcb_module.compute_module.l2_size
        )
        if l2_sram_requirement > l2_sram_limit:
            return float('inf')  # 由于SRAM大小限制，映射无效

        M_l2_t = M // l2_tile_M
        M_remain = M % l2_tile_M

        l2_tiles = []

        if M_l2_t != 0:
            for _ in range(M_l2_t):
                l2_tiles.append(self.L2TileSimulator(
                    l2_tile_M,
                    N,
                    data_type,
                    mapping,
                    pcb_module,
                ))
        if M_remain != 0:
            l2_tiles.append(self.L2TileSimulator(
                M_remain,
                N,
                data_type,
                mapping,
                pcb_module,
            ))

        total_cycle_count = 0
        l2_tile_count = len(l2_tiles)
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
            mapping: "CausalMask.Mapping",
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
            self, M: int, N: int, data_type: DataType, pcb_module: Device
        ):
            bandwidth_per_cycle = (
                pcb_module.io_module.bandwidth / pcb_module.compute_module.clock_freq
            )
            return ceil(
                M * N * data_type.word_size / bandwidth_per_cycle
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "CausalMask.Mapping",
            pcb_module: Device,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = CausalMask.L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                data_type,
                mapping,
                pcb_module,
            )
            l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
            l1_tile_cycle_count = (
                l1_tile.read_cycle_count
                + l1_tile.compute_cycle_count
                + l1_tile.write_cycle_count
            )
            total_cycle_count = ceil(
                l1_tile_count / pcb_module.compute_module.core_count
            ) * l1_tile_cycle_count
            return total_cycle_count

    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "CausalMask.Mapping",
            pcb_module: Device,
        ):
            self.M = M
            self.N = N
            self.read_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, data_type, pcb_module
            )
            self.write_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )

        def simulate_l1_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, pcb_module: Device
        ):
            bandwidth_per_cycle = pcb_module.compute_module.l2_bandwidth_per_cycle
            return ceil(M * N * data_type.word_size / bandwidth_per_cycle)

        def simulate_l1_tile_compute_cycle_count(
            self, M: int, N: int, data_type: DataType, pcb_module: Device
        ):
            total_flop_count = M * N  # 每个元素一次操作
            flops_per_cycle = pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            return ceil(total_flop_count / flops_per_cycle)

    def run_on_gpu(self):
        assert self.shape is not None
        input_tensor = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []
        # 预热
        for _ in range(3):
            _ = self._causal_mask_gpu(input_tensor)
            torch.cuda.synchronize()
        # 测量
        for _ in range(self.iterations):
            start = time.time()
            output = self._causal_mask_gpu(input_tensor)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape == input_tensor.shape
            latencies.append(end - start)
        self.latency_on_gpu = statistics.median(latencies)
        return self.latency_on_gpu

    @staticmethod
    @torch.compile
    def _causal_mask_gpu(input_tensor: torch.Tensor) -> torch.Tensor:
        seq_len = input_tensor.size(-1)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=input_tensor.device), diagonal=1
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(input_tensor.size(0), -1, -1)
        input_tensor = input_tensor.masked_fill(causal_mask, float('-inf'))
        return input_tensor

    @staticmethod
    def gpu_kernel_launch_overhead():
        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            causal_mask = torch.triu(torch.ones(size, size, device='cuda'), diagonal=1).bool()
            c = a.masked_fill(causal_mask, float('-inf'))
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print('GPU内核启动开销：', avg_overhead * 1e6, '微秒')
        print(latencies)
        return avg_overhead