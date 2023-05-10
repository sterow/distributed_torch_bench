import os
import torch

global myrank
global world_size
global is_master
global local_rank

torch.distributed.init_process_group(backend="nccl", init_method="env://")

assert torch.distributed.is_initialized()

myrank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", default="0"))


# NOTICE: without this line, the P2P send and receive will hangs with NCCL backend!
torch.cuda.set_device(local_rank)

tensor = torch.zeros(4 * 1024 * 1024 * 1024).cuda(local_rank)
torch.distributed.all_reduce(tensor)

for _ in range(1000):
    if myrank // 2 == 0:
        r = torch.distributed.isend(tensor, myrank + 2)
    else:
        r = torch.distributed.recv(tensor, myrank - 2)
    torch.cuda.synchronize()
    r.wait()