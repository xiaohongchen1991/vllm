python - <<'PY'
import torch
from triton import runtime

cache = runtime.driver.active.get_empty_cache_for_benchmark()
runtime.driver.active.clear_cache(cache)
torch.cuda.synchronize()

PY
