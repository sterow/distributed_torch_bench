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
    def run(self, target_rank=0):
        pass

    def __init__(self, data_size, world_size):
        self.data_size = data_size
        self.world_size = world_size


class SendRecv(CollectiveOp):
    def prepare(self):
        self.data_size /= 2
        print("Generating send and recv tensor, size: {:.0f} GB"
              .format(self.data_size / 1024 / 1024 / 1024))
        self.x = torch.zeros(self.data_size / 4).cuda(local_rank)
        self.y = torch.zeros(self.data_size / 4).cuda(local_rank)

    def run(self, target_rank=0):
        recv_req = torch.distributed.irecv(self.y)
        torch.distributed.send(self.x, dst=target_rank)
        recv_req.wait()


class Reduce(CollectiveOp):
    def prepare(self):
        pass  # TODO

    def run(self, target_rank=0):
        torch.distributed.reduce(self.x, dst=target_rank)


class AllReduce(CollectiveOp):
    def prepare(self):
        print("Generating all_reduce tensor, size: {:.0f} GB"
              .format(self.data_size / 1024 / 1024 / 1024))
        self.x = torch.zeros(self.data_size / 4).cuda(local_rank)

    def run(self, target_rank=0):
        torch.distributed.all_reduce(self.x)


class Gather(CollectiveOp):
    def prepare(self):
        pass  # TODO

    def run(self, target_rank=0):
        torch.distributed.gather(self.x, gather_list=self.x_list)


class AllGather(CollectiveOp):
    def prepare(self):
        pass  # TODO

    def run(self, x_list):
        torch.distributed.all_gather(self.x_list, self.x)


class Scatter(CollectiveOp):
    def prepare(self):
        pass  # TODO

    def run(self, target_rank=0):
        torch.distributed.scatter(self.x, scatter_list=self.x_list)


class ReduceScatter(CollectiveOp):
    def prepare(self):
        pass  # TODO

    def run(self, target_rank=0):
        torch.distributed.reduce_scatter(self.x, input_list=self.x_list)


class AllToAll(CollectiveOp):
    def prepare(self):
        pass  # TODO

    def run(self, target_rank=0):
        torch.distributed.all_to_all(self.out_list, self.in_list)


class Broadcast(CollectiveOp):
    def prepare(self):
        pass  # TODO

    def run(self, target_rank=0):
        torch.distributed.broadcast(self.x, src=target_rank)


class Barrier(CollectiveOp):
    def prepare(self):
        pass  # TODO

    def run(self, target_rank=0):
        torch.distributed.barrier()


world_size = None
is_master = None
local_rank = None


def print_nccl_env_vars():
    for k in os.environ:
        if k.startswith("NCCL"):
            print(k, "=", os.environ[k])


def init_torch_distributed():
    global world_size
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
          f"size {world_size}, " +
          f"rank {torch.distributed.get_rank()}")

    world_size = torch.distributed.get_world_size()
    is_master = torch.distributed.get_rank() == 0
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print_nccl_env_vars()


def run_collective_loop(collective,
                        loop=1000,
                        inner_loop=10):
    print("{} will loop {} * {} times"
          .format(collective.__name__, loop, inner_loop))

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
                  collective.data_size * inner_loop / (t_end - t_begin) /
                  (1024 * 1024 * 1024)))

        if warming_up:
            warming_up = False
            if is_master:
                print("Start testing now ...")

    if is_master:
        print("Testing has ended.")


def run_p2p_loop(collective,
                 loop=1000,
                 inner_loop=10):

    myrank = torch.distributed.get_rank()

    for _ in range(loop):

        for rank in range(myrank + 1, myrank + world_size):
            rank = rank % world_size

            t_begin = time.time()

            for _ in range(inner_loop):
                collective.run(rank)
                torch.cuda.synchronize()

            t_end = time.time()

            if is_master:
                str_date_time = datetime.fromtimestamp(t_end) \
                    .strftime("%m-%d %H:%M:%S")
                print("{} - {:.1f} GB/s".format(str_date_time,
                    collective.data_size * inner_loop / (t_end - t_begin) /
                    (1024 * 1024 * 1024)))


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collective",
                    dest="collective",
                    choices=["send_recv", "broadcast", "all_reduce",
                             "reduce", "all_gather", "gather", "scatter",
                             "reduce_scatter", "all_to_all", "barrier"],
                    default="all_reduce",
                    help="Collective function name",
                    type=str)
parser.add_argument("-s", "--size",
                    dest="size",
                    default=64 * 1024 * 1024 * 1024,
                    help="GPU memory size to use in bytes",
                    type=int)
parser.add_argument("-n", "--loop",
                    dest="loop",
                    default=1000,
                    help="Loop count",
                    type=int)

args = parser.parse_args()

init_torch_distributed()

match args.collective:
    case "send_receive":
        op = SendRecv(args.size, world_size)
    case "broadcast":
        op = Broadcast(args.size, world_size)
    case "all_reduce":
        op = AllReduce(args.size, world_size)
    case "reduce":
        op = Reduce(args.size, world_size)
    case "all_gather":
        op = AllGather(args.size, world_size)
    case "gather":
        op = Gather(args.size, world_size)
    case "scatter":
        op = Scatter(args.size, world_size)
    case "reduce_scatter":
        op = ReduceScatter(args.size, world_size)
    case "all_to_all":
        op = AllToAll(args.size, world_size)
    case "barrier":
        op = Barrier(args.size, world_size)

op.prepare()

if "send_receive" == args.collective:
    run_p2p_loop(op, loop=args.loop)
else:
    run_collective_loop(op, loop=args.loop)
