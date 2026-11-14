"""Check the four-card recipe before allocating training workers."""

import shutil

from .common import WORLD_SIZE


def preflight(stage, work):
    import torch

    if torch.cuda.device_count() != WORLD_SIZE:
        raise ValueError("Expose exactly four 80GB CUDA GPUs for this recipe.")
    if any(torch.cuda.get_device_properties(i).total_memory < 75 * 2**30 for i in range(WORLD_SIZE)):
        raise ValueError("This 8B recipe requires four 80GB-class CUDA GPUs.")
    for i in range(WORLD_SIZE):
        with torch.cuda.device(i):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("All four assigned GPUs must support BF16.")
            free, _ = torch.cuda.mem_get_info(i)
            if free < 68 * 2**30:
                raise ValueError("The four GPUs must be free; use a separate renderer or free the assigned cards.")
    # Native FSDP stores one FP32 actor and FP32 Adam states in each full
    # checkpoint (about 90 GiB for 8B). Allow for temp/atomic replacement.
    folder = work / ("route_sft" if stage == "route-sft" else stage)
    if not (folder / "contract.json").is_file() and shutil.disk_usage(work).free < 320 * 2**30:
        raise ValueError("Allow at least 320 GiB free in WORK for checkpoints; dataset and renderer are separate.")
