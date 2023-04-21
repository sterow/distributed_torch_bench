import os
import time

import torch

def all_reduce(x):
    torch.distributed.all_reduce(x)
    torch.cuda.synchronize()

if "TORCH_DIST_BACKEND" in os.environ:
    backend = os.environ["TORCH_DIST_BACKEND"]
else:
    backend = "gloo" if torch.cuda.nccl.version() is None or torch.cuda.device_count() == 0 else "nccl"
print(f"Initializing torch.distributed with backend \"{backend}\"")
torch.distributed.init_process_group(backend=backend, init_method="env://")
assert torch.distributed.is_initialized()
print(f"torch.distributed initialized, backend: \"{backend}\", rank {torch.distributed.get_rank()}, size {torch.distributed.get_world_size()}", flush = True)

local_rank = int(os.environ["LOCAL_RANK"])
is_master = torch.distributed.get_rank() == 0

n = 1024 * 1024 * 1024
outer_loop = 10
inner_loop = 1

x = torch.randn(n).cuda(local_rank)

warming_up = True

if is_master:
    for k in os.environ:
        if k.startswith("NCCL"):
            print(k, "=", os.environ[k], flush = True)
    print("Message size: {:.0f} MB, loop {} * {}".format(n / 1024 / 1024, outer_loop, inner_loop), flush = True)
    print("Warming up ...", flush = True)

for _ in range(outer_loop + 1):
    t = time.time()
    for _ in range(inner_loop):
        all_reduce(x)
    if is_master:
        if warming_up:
            print("Begin testing ...", flush = True)
            warming_up = False
        else:
            print("{:.1f} GB/s".format(n * 4 * inner_loop / (time.time() - t) / 1024 / 1024 / 1024), flush = True)
