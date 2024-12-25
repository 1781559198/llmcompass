import argparse
import json, re
import datetime
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
from software_model.llama2 import(
    Llama2BlockInitComputationTP,
    Llama2BlockAutoRegressionTP,
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
    interconnect_module = None

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


def find_cheapest_design(# 搜索计算硬件架构的最佳设计
    d_model,
    n_heads,
    n_layers,
    batch_size,
    input_seq_length,
    init_latency,
    output_seq_length,
    auto_regression_latency,
    config_path,
    model_type
    
):
    i=0
    smallest_total_area_mm2=float('inf')
    best_arch_specs=None
    arch_specs = read_architecture_template(config_path)
    if model_type == 'transformer':
        model_init = TransformerBlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=1,
                data_type=data_type_dict["fp16"],
            )
        model_auto_regression = TransformerBlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=1,
                data_type=data_type_dict["fp16"],)
    elif model_type == 'llama2':
        model_init = Llama2BlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=1,
                data_type=data_type_dict["fp16"],
            )
        model_auto_regression = Llama2BlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=1,
                data_type=data_type_dict["fp16"],)
        if model_type == 'transformer':
            _ = model_init(Tensor([batch_size, input_seq_length, model_init.d_model], data_type_dict["fp16"]))
            _ = model_auto_regression(Tensor([batch_size, 1, model_init.d_model],data_type_dict["fp16"]), input_seq_length+output_seq_length)
        elif model_type == 'llama2':
            _ = model_init(Tensor([batch_size, 1, model_init.d_model], data_type_dict["fp16"]))
            _ = model_auto_regression(Tensor([batch_size, 1, model_init.d_model],data_type_dict["fp16"]), input_seq_length+output_seq_length)
        # device
        for core_count in [32, 64, 128, 256]:
            arch_specs["device"]["compute_chiplet"]["core_count"] = core_count
            # core
            for sublane_count in [1, 2, 4, 8]:
                arch_specs["device"]["compute_chiplet"]["core"][
                    "sublane_count"
                ] = sublane_count
                # systolic array
                for array_height in [16, 32, 64, 128]:
                    arch_specs["device"]["compute_chiplet"]["core"][
                        "systolic_array"
                    ]["array_height"] = array_height
                    arch_specs["device"]["compute_chiplet"]["core"][
                        "systolic_array"
                    ]["array_width"] = array_height
                    # vector unit
                    for vector_width in [16, 32, 64, 128]:
                        arch_specs["device"]["compute_chiplet"]["core"][
                            "vector_unit"
                        ]["vector_width"] = vector_width
                        for SRAM_KB in [64, 128, 256, 512, 1024]:
                            arch_specs["device"]["compute_chiplet"]["core"][
                                "SRAM_KB"
                            ] = SRAM_KB
                            # global buffer
                            for total_global_buffer_MB in [
                                80,
                                160,
                                240,
                                320,
                                400,
                                480,
                                640,
                                800,
                                960,
                            ]:
                                global_buffer_MB = total_global_buffer_MB
                                global_buffer_bandwidth_per_cycle_byte = (
                                    5120 * global_buffer_MB // 40
                                )
                                arch_specs["device"]["io"][
                                    "global_buffer_MB"
                                ] = global_buffer_MB
                                arch_specs["device"]["io"][
                                    "global_buffer_bandwidth_per_cycle_byte"
                                ] = global_buffer_bandwidth_per_cycle_byte
                                # memory
                                memory_capacity_requirement_GB = ceil(model_auto_regression.memory_requirement*n_layers/1e9/16)*16
                                # print(f"memory_capacity_requirement_GB={model_auto_regression.memory_requirement*n_layers/1e9}")
                                # exit()
                                for memory_protocol in [
                                    "yizhu_SRAM",
                                    "yizhu_3D_DRAM"
                                ]:
                                    arch_specs['device']['memory_protocol']=memory_protocol
                                    if memory_protocol == "yizhu_SRAM":
                                        channel_count_list = [16, 24, 32]
                                        pin_count_per_channel=512
                                        bandwidth_per_pin_bit=1e9
                                    if memory_protocol == "yizhu_3D_DRAM":
                                        channel_count_list = [16, 24, 32]
                                        pin_count_per_channel=512
                                        bandwidth_per_pin_bit=1e9
                                    for channel_count in channel_count_list:
                                        arch_specs['device']['memory']['total_capacity_GB'] = memory_capacity_requirement_GB
                                        arch_specs['device']['io']['memory_channel_active_count'] = channel_count
                                        arch_specs['device']['io']['memory_channel_physical_count'] = channel_count
                                        arch_specs['device']['io']['pin_count_per_channel'] = pin_count_per_channel
                                        arch_specs['device']['io']['bandwidth_per_pin_bit'] = bandwidth_per_pin_bit

                                        arch_specs['device']['io_3d_dram']['memory_channel_active_count'] = channel_count
                                        arch_specs['device']['io_3d_dram']['memory_channel_physical_count'] = channel_count
                                        arch_specs['device']['io_3d_dram']['pin_count_per_channel'] = pin_count_per_channel
                                        arch_specs['device']['io_3d_dram']['bandwidth_per_pin_bit'] = bandwidth_per_pin_bit

                                        
                                        total_area_mm2=calc_compute_chiplet_area_mm2(arch_specs)+calc_io_die_area_mm2(arch_specs)
                                        # print(f"channel_count={arch_specs['device']['io']['memory_channel_active_count']},total area={total_area_mm2}")
                                        if total_area_mm2>900:
                                            continue
                                        system=template_to_system(arch_specs)
                                        init_roofline_latency=model_init.roofline_model(system)*n_layers
                                        if init_roofline_latency>init_latency:
                                            continue
                    
                                        auto_regression_roofline_latency=model_auto_regression.roofline_model(system)*n_layers
                                        if auto_regression_roofline_latency>auto_regression_latency:
                                            continue
                                        auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(system, 'heuristic-GPU')
                                        if auto_regression_latency_simulated>auto_regression_latency:
                                            continue
                                        init_latency_simulated = model_init.compile_and_simulate(system, 'heuristic-GPU')
                                        if init_latency_simulated>init_latency:
                                            continue
                                        if total_area_mm2<smallest_total_area_mm2:
                                            smallest_total_area_mm2=total_area_mm2
                                            best_arch_specs=arch_specs
                                            best_arch_specs['area_per_device_mm2']=total_area_mm2
                                            # print(f"best_arch_specs={best_arch_specs}")
                                            # print(f"smallest_total_area_mm2={smallest_total_area_mm2}")
                                        i=i+1
                                        if i%100==0:
                                            print(f'i={i}')
    print(f'number of potential designs={i}')
    with open("configs/best_arch_specs.json", "w") as f:
        json.dump(best_arch_specs, f, indent=4)
                                            

if __name__ == "__main__":
    # test_template_to_system()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_type", type=str, choices=['g100', 'ga100', 'ga102_template', 'generate_template', 'G100', 'GA100', 'latency_design', 'mi210', 'template'], 
                       default='g100',
                       help="Choose config type: g100 or ga100")
    parser.add_argument("--model_type", type=str, choices=['transformer', 'llama2'],
                    default='transformer',
                    help="Choose model type: transformer or llama2")
    args = parser.parse_args()
    
    # 根据选择设置配置路径
    config_path = {
        'g100': "configs/G100.json",
        'ga100': "configs/GA100.json",
        'ga102_template': "configs/GA102_template.json",
        'generate_template': "configs/generate_template.json",
        'G100': "configs/G100.json",
        'GA100': "configs/GA100.json",
        'latency_design': "configs/latency_design.json",
        'mi210': "configs/mi210.json",
        'template': "configs/template.json",

    }[args.config_type]
    
    
    find_cheapest_design(12288, 96, 96, 8, 2048, 5, 1024, 0.1, config_path, args.model_type)



