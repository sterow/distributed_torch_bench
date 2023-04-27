import argparse
import os
import time
import torch
from abc import abstractmethod
from datetime import datetime


class CollectiveOp:
    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def __init__(self, tensor_size, world_size):
        self.tensor_size = tensor_size
        self.world_size = world_size


class Reduce(CollectiveOp):
    def prepare():
        pass  # TODO

    def run(x):
        torch.distributed.reduce(x, dst=0)


class AllReduce(CollectiveOp):
    def prepare():
        pass  # TODO

    def run(x):
        torch.distributed.all_reduce(x)


class Gather(CollectiveOp):
    def prepare():
        pass  # TODO

    def run(x, x_list):
        torch.distributed.gather(x, gather_list=x_list)


class AllGather(CollectiveOp):
    def prepare():
        pass  # TODO

    def run(x, x_list):
        torch.distributed.all_gather(x_list, x)


class Scatter(CollectiveOp):
    def prepare():
        pass  # TODO

    def run(x, x_list):
        torch.distributed.scatter(x, scatter_list=x_list)


class ReduceScatter(CollectiveOp):
    def prepare():
        pass  # TODO

    def run(x, x_list):
        torch.distributed.reduce_scatter(x, input_list=x_list)


class AllToAll(CollectiveOp):
    def prepare():
        pass  # TODO

    def run(out_list, in_list):
        torch.distributed.all_to_all(out_list, in_list)


class Broadcast(CollectiveOp):
    def prepare():
        pass  # TODO

    def run(x):
        torch.distributed.broadcast(x, src=0)


class Barrier(CollectiveOp):
    def prepare():
        pass  # TODO

    def run():
        torch.distributed.barrier()


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
            collective.run()
            torch.cuda.synchronize()

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

match args.collective:
    case "send":
        pass
    case "recv":
        pass
    case "broadcast":
        op = Broadcast(args.size, torch.distributed.get_world_size())
    case "all_reduce":
        op = AllReduce(args.size, torch.distributed.get_world_size())
    case "reduce":
        op = Reduce(args.size, torch.distributed.get_world_size())
    case "all_gather":
        op = AllGather(args.size, torch.distributed.get_world_size())
    case "gather":
        op = Gather(args.size, torch.distributed.get_world_size())
    case "scatter":
        op = Scatter(args.size, torch.distributed.get_world_size())
    case "reduce_scatter":
        op = ReduceScatter(args.size, torch.distributed.get_world_size())
    case "all_to_all":
        op = AllToAll(args.size, torch.distributed.get_world_size())
    case "barrier":
        op = Barrier(args.size, torch.distributed.get_world_size())

op.prepare()
run_collective_loop(op, tensor_size=args.size, loop=args.loop)
