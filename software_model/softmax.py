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


class Softmax(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:-1])# 最后一维之前的维度的乘积
        self.N = input.shape[-1]# 最后一维的大小
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type
        )
        return input

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu*1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, N: int, data_type: DataType):
            self.M = M# 最后一维之前的维度的乘积
            self.N = N# 最后一维的大小
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
                f"l2_tile_M: {self.l2_tile_M}, is_l2_double_buffering: {self.is_l2_double_buffering}, l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, is_l1_double_buffering: {self.is_l1_double_buffering}"
            )
    
    def roofline_model(self, pcb_module: Device):
        self.io_count = self.M * self.N * self.data_type.word_size * 3
        self.flop_count = self.M * self.N * (pcb_module.compute_module.core.vector_unit.flops_per_exp * 3 + 7)
        self.roofline_latency=max(self.io_count/min(pcb_module.io_module.bandwidth, pcb_module.compute_module.l2_bandwidth_per_cycle*pcb_module.compute_module.clock_freq), self.flop_count/pcb_module.compute_module.total_vector_flops)
        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode=None):
        self.computational_graph.data_type = pcb_module.compute_module.core.vector_unit.data_type
        min_cycle_count = float("inf")
        best_mapping = None
        M = self.computational_graph.M# 最后一维之前的维度的乘积
        N = self.computational_graph.N# 最后一维的大小
        data_type = self.computational_graph.data_type
        l2_tile_N = N
        l2_tile_M = (# global_buffer_size//(l2_tile_N*data_type.word_size)每行的数据大小
            pcb_module.compute_module.l2_size // (l2_tile_N * data_type.word_size)
        )
        l2_tile_M = min(l2_tile_M, M)# 确保不超出硬件的限制
        is_l2_double_buffering = False
        for l1_N_tiling_factor in [1, 2, 4, 8, 16, 32]:
            l1_tile_N = ceil(l2_tile_N / l1_N_tiling_factor)# 每个l1_tile的列数并且向上取整
            for l1_tile_M in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for is_l1_double_buffering in [True, False]:
                    if is_l1_double_buffering:
                        if (# 确保分块的大小小于SRAM的大小
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
                    mapping = self.Mapping(# 记录映射信息
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
        #self.best_mapping.display()
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

        if mapping.is_l2_double_buffering:# 检查
            assert (
                l2_tile_M * N * data_type.word_size * 2
                <= pcb_module.compute_module.l2_size
            )
        else:
            assert (
                l2_tile_M * N * data_type.word_size <= pcb_module.compute_module.l2_size
            )

        M_l2_t = M // l2_tile_M
        M_remain = M % l2_tile_M

        l2_tiles = np.empty([ceil(M / l2_tile_M)], dtype=self.L2TileSimulator)# (shape, dtype)

        if M_l2_t != 0:
            l2_tiles[:M_l2_t] = self.L2TileSimulator(
                l2_tile_M,# 每个l2_tile的行数，compile_and_simulate那里遍历的结果
                N,# 每个l2_tile的列数
                data_type,
                mapping,
                pcb_module,
            )
        if M_remain != 0:
            l2_tiles[-1] = self.L2TileSimulator(
                M_remain,# 最后一个l2_tile的行数
                N,# 最后一个l2_tile的列数
                data_type,
                mapping,
                pcb_module,
            )

        total_cycle_count = 0
        l2_tile_count = ceil(M / l2_tile_M)
        for m in range(l2_tile_count):# 计算处理整个矩阵所需的总时钟周期数
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

        def simulate_l2_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, chiplet_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (# 一个时钟内可以传输的字节数
                    chiplet_module.io_module.bandwidth# I/O模块的总带宽
                    / chiplet_module.compute_module.clock_freq# 计算模块的时钟频率
                )
            )

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
            total_cycle_count = (# l1的分块数/硬件核心数
                ceil(l1_tile_count / pcb_module.compute_module.core_count) + 1# l1总分块数/core +1是增加了一次同步操作？
            ) * (
                l1_tile_cycle_count# l1的基础执行次数
                + log2(ceil(N / l1_tile_N)) * l1_tile.reduction_cycle_count# 规约操作的执行次数
            )
            return total_cycle_count


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
            self.reduction_cycle_count = (# 规约操作的周期数
                M# 规约操作的浮点运算时钟周期数
                * N
                * (self.flops_per_exp + 2)
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                + 
                M# 数据传输需要的周期数（规约操作需要传递数据）
                * N
                * data_type.word_size
                * 2
                / (pcb_module.compute_module.l2_bandwidth_per_cycle/pcb_module.compute_module.core_count)
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
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            # online softmax
            total_flop_count = M * N * (self.flops_per_exp * 3 + 7)
            return ceil(# 返回周期数
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )

    def run_on_gpu(self):
        assert self.shape is not None
        input = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []
        # warmup
        for _ in range(3):
            _ = torch.softmax(input, dim=-1)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = torch.softmax(input, dim=-1)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape == input.shape
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
            c = torch.softmax(a, dim=-1)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        print(latencies)
        return avg_overhead