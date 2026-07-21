# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for Helion kernel management."""

import regex as re
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def get_gpu_name(device_id: int | None = None) -> str:
    if device_id is None:
        logger.warning_once(
            "get_gpu_name() called without device_id, defaulting to 0. "
            "This may return the wrong device name in multi-node setups."
        )
        device_id = 0
    return current_platform.get_device_name(device_id)


def canonicalize_gpu_name(name: str) -> str:
    """
    Canonicalize GPU name for use as a platform identifier.

    Converts to lowercase and replaces separators with underscores, keeping
    any variant suffix (form factor, memory size, memory type, etc.).
    e.g., "NVIDIA H100 80GB HBM3" -> "nvidia_h100_80gb_hbm3"
          "NVIDIA A100-SXM4-80GB" -> "nvidia_a100_sxm4_80gb"
          "AMD Instinct MI300X"   -> "amd_instinct_mi300x"
    """
    if not name or not name.strip():
        raise ValueError("GPU name cannot be empty")
    return re.sub(r"[\s/-]+", "_", name.lower())


def get_canonical_gpu_name(device_id: int | None = None) -> str:
    return canonicalize_gpu_name(get_gpu_name(device_id))


def get_fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()


def get_int8_min_max() -> tuple[int, int]:
    qtype_traits = torch.iinfo(torch.int8)
    return qtype_traits.min, qtype_traits.max


def get_int8_min_scaling_factor() -> float:
    return torch.finfo(torch.float32).eps
