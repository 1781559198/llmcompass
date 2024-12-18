from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.llama2 import Llama2BlockAutoRegressionTP, Llama2BlockInitComputationTP
from software_model.utils import DataType, data_type_dict
from software_model.utils import data_type_dict, Tensor


specs = read_architecture_template(r"configs\G100.json")
system = template_to_system(specs)

#定义序列长度和批次大小
seq_len = 12288
bs = 8

model_auto_regression = Llama2BlockAutoRegressionTP(
        d_model=12288,
        n_heads=96,
        device_count=1,
        data_type=data_type_dict["fp16"],
    )
_ = model_auto_regression(
	Tensor([bs, 1, 12288], data_type_dict["fp16"]),
	seq_len,
)

print("Starting simulation...")
auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
	system, "heuristic-TPU"
)

# auto_regression_latency_simulated = model_auto_regression.run_on_gpu(
# )

print("Simulation completed!")
print(f"Simulated latency: {auto_regression_latency_simulated}")
