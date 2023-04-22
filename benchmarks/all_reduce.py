import os
import time
from datetime import datetime

import torch

def print_nccl_env_vars():
    for k in os.environ:
        if k.startswith("NCCL"):
            print(k, "=", os.environ[k])

def all_reduce(x):
    torch.distributed.all_reduce(x)
    torch.cuda.synchronize()

if "TORCH_DIST_BACKEND" in os.environ:
    backend = os.environ["TORCH_DISTRIBUTED_BACKEND"]
else:
    backend = "gloo" if torch.cuda.nccl.version() is None or torch.cuda.device_count() == 0 else "nccl"

torch.distributed.init_process_group(backend=backend, init_method="env://")
assert torch.distributed.is_initialized()
print(f"torch.distributed initialized, backend: \"{backend}\", size {torch.distributed.get_world_size()}, rank {torch.distributed.get_rank()}")

is_master = torch.distributed.get_rank() == 0

local_rank = int(os.environ["LOCAL_RANK"])
if local_rank == 0: print_nccl_env_vars()

n = 3 * 1024 * 1024 * 1024
outer_loop = 1000
inner_loop = 10

print("Generating test message, size: {:.0f} MB, test will loop {} * {} times".format(n / 1024 / 1024, outer_loop, inner_loop))
x = torch.randn(n).cuda(local_rank)

warming_up = True
if is_master : print("Warming up ...")

for _ in range(outer_loop + 1):
    t_begin = time.time()
    for _ in range(inner_loop):
        all_reduce(x)
    t_end = time.time()
    if warming_up:
        warming_up = False
        if is_master: print("Start testing now ...")
    if is_master:
        str_date_time = datetime.fromtimestamp(t_end).strftime("%m-%d %H:%M:%S")
        print("{} - {:.1f} GB/s".format(str_date_time, n * 4 * inner_loop / (t_end - t_begin) / 1024 / 1024 / 1024))

if is_master: print("Testing has ended.")