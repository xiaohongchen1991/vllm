import json

with open("/home/dev/sean/vllm/vllm/kernels/helion/configs/static_scaled_fp8_quant/nvidia_h100.json", "r") as f:
    data = json.load(f)

# sort only top-level keys
sorted_data = dict(sorted(data.items()))

with open("output.json", "w") as f:
    json.dump(sorted_data, f, indent=2)
