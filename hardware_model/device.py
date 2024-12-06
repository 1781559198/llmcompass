from hardware_model.compute_module import ComputeModule, compute_module_dict
from hardware_model.io_module import IOModule, IO_module_dict
from hardware_model.memory_module import MemoryModule, memory_module_dict


class Device:
    def __init__(
        self,
        compute_module: ComputeModule,
        io_module: IOModule,
        memory_module: MemoryModule,
        is_yizhu_g100: bool,
    ) -> None:
        self.compute_module = compute_module
        self.io_module = io_module
        self.memory_module = memory_module
        self.is_yizhu_g100 = is_yizhu_g100

device_dict = {
    "A100_80GB_fp16": Device(
        compute_module_dict["A100_fp16"],
        IO_module_dict["A100"],
        memory_module_dict["A100_80GB"],
        False,
    ),
    "TPUv3": Device(
        compute_module_dict["TPUv3_bf16"],
        IO_module_dict["TPUv3"],
        memory_module_dict["TPUv3"],
        False,
    ),
    "MI210": Device(
        compute_module_dict["MI210_fp16"],
        IO_module_dict["MI210"],
        memory_module_dict["MI210"],
        False,
    ),
    "TPUv3_new": Device(
        compute_module_dict["TPUv3_new"],
        IO_module_dict["TPUv3"],
        memory_module_dict["TPUv3"],
        False,
    ),
}
