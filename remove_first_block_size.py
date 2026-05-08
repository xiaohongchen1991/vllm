import json
from pathlib import Path

# Input JSON file
json_path = Path("vllm/kernels/helion/configs/silu_and_mul_dynamic_per_token_quant/nvidia_h100.json")

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Remove the first value from each block_sizes list
for key, value in data.items():
    if (
        isinstance(value, dict)
        and "block_sizes" in value
        and isinstance(value["block_sizes"], list)
        and len(value["block_sizes"]) > 0
    ):
        value["block_sizes"] = value["block_sizes"][1:]

# Save updated JSON
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Updated {json_path}")
