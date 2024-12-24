import json, re
from hardware_model.compute_module import (
    VectorUnit,
    SystolicArray,
    Core,
    ComputeModule,
    overhead_dict,
)
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule
from hardware_model.device import Device
from hardware_model.interconnect import LinkModule, InterConnectModule, TopologyType
from hardware_model.system import System
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
from math import ceil

def read_architecture_template(file_path):
    with open(file_path, "r") as f:
        arch_specs = json.load(f)
    # 读取布尔
    return arch_specs


def template_to_system(arch_specs):
    device_specs = arch_specs["device"]
    compute_chiplet_specs = device_specs["compute_chiplet"]# 芯片组
    io_specs = device_specs["io"]
    io_3d_dram_specs = device_specs.get("io_3d_dram", None)# 3D DRAM
    core_specs = compute_chiplet_specs["core"]
    sublane_count = core_specs["sublane_count"]
    # vector unit
    vector_unit_specs = core_specs["vector_unit"]
    vector_unit = VectorUnit(
        sublane_count # 子通道
        * vector_unit_specs["vector_width"]
        * vector_unit_specs["flop_per_cycle"], # 计算每个周期的总浮点运算数量（FLOPs），即 total_vector_flops_per_cycle
        int(re.search(r"(\d+)", vector_unit_specs["data_type"]).group(1)) // 8, # 通过正则表达式从数据类型字符串中提取数据类型的位数，将其转换为字节数
        35, # 指数运算需要的flops
        vector_unit_specs["vector_width"],
        sublane_count,
    )
    # systolic array
    systolic_array_specs = core_specs["systolic_array"]
    systolic_array = SystolicArray(
        systolic_array_specs["array_height"],
        systolic_array_specs["array_width"],
        systolic_array_specs["mac_per_cycle"],
        int(re.search(r"(\d+)", systolic_array_specs["data_type"]).group(1)) // 8,
        int(re.search(r"(\d+)", systolic_array_specs["data_type"]).group(1)) // 8,
        sublane_count *systolic_array_specs["array_height"] * systolic_array_specs["array_width"] * systolic_array_specs["mac_per_cycle"] * 2
    )
    # core
    
    core = Core(
        vector_unit,
        systolic_array,
        1 if core_specs.get("single_tpe", False) else sublane_count,
        # systolic_array_count,
        core_specs["SRAM_KB"] * 1024,
    )

    # compute module
    compute_module = ComputeModule(
        core,
        compute_chiplet_specs["core_count"] * device_specs["compute_chiplet_count"],
        device_specs["frequency_Hz"],
        io_specs["global_buffer_MB"] * 1024 * 1024,
        io_specs["global_buffer_bandwidth_per_cycle_byte"],
        0 if io_3d_dram_specs is None else (io_3d_dram_specs["global_buffer_MB"] * 1024 * 1024),
        0 if io_3d_dram_specs is None else io_3d_dram_specs["global_buffer_bandwidth_per_cycle_byte"],
        overhead_dict["A100"],
    )

    # self.core = core
    # self.core_count = core_count
    # self.clock_freq = clock_freq
    # self.l2_size = int(l2_size)  # global buffer
    # self.l2_bandwidth_per_cycle = l2_bandwidth_per_cycle  # Byte/clock
    # self.total_vector_flops_per_cycle = ( 
    #     core.vector_unit.total_vector_flops_per_cycle * core_count # 总向量flops
    # )
    # self.total_vector_flops = self.total_vector_flops_per_cycle * clock_freq
    # self.total_systolic_array_flops = ( # 总矩阵乘法flops计算
    #     core_count # 核心数量
    #     * core.systolic_array_count # 每个核心中矩阵乘法单元
    #     * core.systolic_array.mac_per_cycle # 每个核心中矩阵乘法单元
    #     * 2
    #     * core.systolic_array.array_height # Systolic Array 的维度，表示矩阵乘法的规模
    #     * core.systolic_array.array_width 
    #     * clock_freq # 时钟频率
    # )
    # self.overhead = overhead

    # io module
    io_module = IOModule(
        io_specs["memory_channel_active_count"] # 活跃的内存通道数量
        * io_specs["pin_count_per_channel"] # 每个内存通道的引脚数量
        * io_specs["bandwidth_per_pin_bit"] # 每个引脚每秒可以传输的比特数
        // 8, # 字节转换
        1e-6,
    )

    # io_3d_dram
    if io_3d_dram_specs:
        io_3d_dram = IOModule(
            io_3d_dram_specs["memory_channel_active_count"]
            * io_3d_dram_specs["pin_count_per_channel"]
            * io_3d_dram_specs["bandwidth_per_pin_bit"]
            // 8,
            1e-6,
        )
    else:
        io_3d_dram = None  # 或者设置一个默认值

    # memory module
    memory_module = MemoryModule(
        device_specs["memory"]["total_capacity_GB"] * 1024 * 1024 * 1024
    )
    # device
    device = Device(compute_module, io_module, memory_module, io_3d_dram)
    # interconnect
    interconnect_specs = arch_specs["interconnect"]
    link_specs = interconnect_specs["link"]
    link_module = LinkModule(
        link_specs["bandwidth_per_direction_byte"], # 每个方向的通信带宽，以字节为单位
        link_specs["bandwidth_both_directions_byte"], # 同时在两个方向上通信的总带宽（双向带宽）
        link_specs["latency_second"], # 通信的延迟，以秒为单位
        link_specs["flit_size_byte"], # 最小的传输单位（flit）的大小，以字节为单位
        link_specs["max_payload_size_byte"], # 一次传输的最大有效载荷大小
        link_specs["header_size_byte"], # 数据包的头部大小，以字节为单位
    )
    interconnect_module = InterConnectModule(
        arch_specs["device_count"],
        TopologyType.FC
        if interconnect_specs["topology"] == "FC"
        else TopologyType.RING,
        link_module,
        interconnect_specs["link_count_per_device"],
    )

    # system
    system = System(device, interconnect_module)

    return system


def test_template_to_system():
    arch_specs, is_yizhu_g100 = read_architecture_template("configs/template.json")
    A100_system = template_to_system(arch_specs, is_yizhu_g100)
    bs = 8
    s = 2048
    model = TransformerBlockInitComputationTP(
        d_model=12288,
        n_heads=96,
        device_count=4,
        data_type=data_type_dict["fp16"],
    )
    _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
    model.roofline_model(A100_system)


def find_cheapest_design(
    d_model,
    n_heads,
    n_layers,
    batch_size,
    input_seq_length,
    init_latency,
    output_seq_length,
    auto_regression_latency,
    model_type
):
    i = 0
    smallest_total_area_mm2 = float('inf')
    best_arch_specs = None
    arch_specs = read_architecture_template("configs/template.json")
    
    # Example usage of model_type
    if model_type == "transformer":
        model_init = TransformerBlockInitComputationTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=4,
            data_type=data_type_dict["fp16"],
        )
        model_auto_regression = TransformerBlockAutoRegressionTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=4,
            data_type=data_type_dict["fp16"],
        )
    elif model_type == "llama2":
        # Initialize llama2 model
        pass
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Continue with the rest of the function logic
    _ = model_init(Tensor([batch_size, input_seq_length, model_init.d_model], data_type_dict["fp16"]))
    _ = model_auto_regression(Tensor([batch_size, 1, model_init.d_model], data_type_dict["fp16"]), input_seq_length + output_seq_length)
    
    # ... rest of the function logic ...


if __name__ == "__main__":
    # test_template_to_system()
    find_cheapest_design(12288, 96, 96, 8, 2048, 5, 1024, 0.1, "transformer")


