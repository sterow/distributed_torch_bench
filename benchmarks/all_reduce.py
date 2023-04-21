import os
import time

import torch

backend="gloo" if torch.cuda.nccl.version() is None or torch.cuda.device_count() == 0 else "nccl"
torch.distributed.init_process_group(backend=backend, init_method="env://")
assert torch.distributed.is_initialized()
print(f"torch.distributed initialized, backend: {backend}, rank {torch.distributed.get_rank()}, size {torch.distributed.get_world_size()}")

local_rank = int(os.environ["LOCAL_RANK"])

n = 1024 * 1024 * 1024

x = torch.randn(n).cuda(local_rank)

if torch.distributed.get_rank() == 0:
    for k in os.environ:
        if k.startswith("NCCL"):
            print(k, "=", os.environ[k])


for _ in range(10):
    t = time.time()
    torch.distributed.all_reduce(x)
    torch.cuda.synchronize()
    if torch.distributed.get_rank() == 0:
        print("{:.0f} MB: {:.1f} GB/s".format(n / 1024 / 1024, n * 4 / (time.time() - t) / 1024 / 1024 / 1024))
