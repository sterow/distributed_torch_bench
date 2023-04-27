import argparse
import os
import time
import torch
from datetime import datetime


def all_reduce(x):
    torch.distributed.all_reduce(x)
    torch.cuda.synchronize()


is_master = None
local_rank = None


def print_nccl_env_vars():
    for k in os.environ:
        if k.startswith("NCCL"):
            print(k, "=", os.environ[k])


def init_torch_distributed():
    global is_master
    global local_rank

    if "TORCH_DISTRIBUTED_BACKEND" in os.environ:
        backend = os.environ["TORCH_DISTRIBUTED_BACKEND"]
    elif torch.cuda.nccl.version() is None or torch.cuda.device_count() == 0:
        backend = "gloo"
    else:
        backend = "nccl"

    torch.distributed.init_process_group(backend=backend, init_method="env://")

    assert torch.distributed.is_initialized()
    print(f"torch.distributed initialized, backend: \"{backend}\", " +
          f"size {torch.distributed.get_world_size()}, " +
          f"rank {torch.distributed.get_rank()}")

    is_master = torch.distributed.get_rank() == 0

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print_nccl_env_vars()


def run_collective_loop(collective,
                        tensor_size=1024 * 1024 * 1024,
                        loop=1000,
                        inner_loop=10):
    print("Generating test message, size: {:.0f} MB, {} will loop {} * {} times"
          .format(tensor_size / 1024 / 1024,
                  collective.__name__, loop, inner_loop))

    x = torch.zeros(tensor_size).cuda(local_rank)

    warming_up = True
    if is_master:
        print("Warming up ...")

    for _ in range(loop + 1):
        t_begin = time.time()
        for _ in range(inner_loop):
            collective(x)
        t_end = time.time()
        if is_master:
            str_date_time = datetime.fromtimestamp(t_end) \
                .strftime("%m-%d %H:%M:%S")
            print("{} - {:.1f} GB/s".format(str_date_time,
                  tensor_size * 4 * inner_loop / (t_end - t_begin) /
                  (1024 * 1024 * 1024)))

        if warming_up:
            warming_up = False
            if is_master:
                print("Start testing now ...")

    if is_master:
        print("Testing has ended.")


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collective",
                    dest="collective",
                    choices=["send", "recv", "broadcast", "all_reduce",
                             "reduce", "all_gather", "gather", "scatter",
                             "reduce_scatter", "all_to_all", "barrier"],
                    default="all_reduce",
                    help="Collective function name",
                    type=str)
parser.add_argument("-s", "--size",
                    dest="size",
                    default=16 * 1024 * 1024 * 1024,
                    help="Tensor size",
                    type=int)
parser.add_argument("-n", "--loop",
                    dest="loop",
                    default=1000,
                    help="Loop count",
                    type=int)

args = parser.parse_args()

init_torch_distributed()
run_collective_loop(all_reduce, tensor_size=args.size, loop=args.loop)
