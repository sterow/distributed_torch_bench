import argparse
import logging
import os
import time
import torch
from abc import abstractmethod


class CollectiveOp:
    @abstractmethod
    def run(self):
        pass

    def prepare(self, tensor_size, local_rank):
        self.tensor_size = tensor_size
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank

    def prepare_one_tensor(self, op_name):
        logging.info("Generating {} message, size: {:.0f} MB"
                     .format(op_name, self.tensor_size / 1024 / 1024))
        return torch.zeros(self.tensor_size).cuda(self.local_rank)

    def prepare_list_of_tensors(self, op_name):
        logging.info("Generating {} {} messages, size: {:.0f} MB"
                     .format(self.world_size, op_name,
                             self.tensor_size / 1024 / 1024))

        x_list = []
        for _ in range(self.world_size):
            x_list += torch.zeros(self.tensor_size).cuda(self.local_rank)

        return x_list

    def prepare_tensors(self, op_name, one_and_all=False):
        total_tensor_size = self.tensor_size * (1 + self.world_size if one_and_all else 1)
        mem_info = torch.cuda.mem_get_info(self.local_rank)
        if total_tensor_size > mem_info[0]:
            logging.warn("Total tensors size {} bigger than GPU avaliable memory {}!"
                         .format(total_tensor_size, mem_info[0]))
        self.x = CollectiveOp.prepare_one_tensor(self, op_name)
        if one_and_all and self.rank == 0:
            self.x_list = CollectiveOp.prepare_list_of_tensors(self, op_name + "_list")


class Reduce(CollectiveOp):
    def prepare(self, tensor_size, local_rank):
        CollectiveOp.prepare(tensor_size, local_rank)
        CollectiveOp.prepare_tensors(self, "reduce")

    def run(self):
        torch.distributed.reduce(self.x, dst=0)


class AllReduce(CollectiveOp):
    def prepare(self, tensor_size, local_rank):
        CollectiveOp.prepare(tensor_size, local_rank)
        CollectiveOp.prepare_tensors(self, "all_reduce")

    def run(self):
        torch.distributed.all_reduce(self.x)


class Gather(CollectiveOp):
    def prepare(self, tensor_size, local_rank):
        CollectiveOp.prepare(tensor_size, local_rank)
        CollectiveOp.prepare_tensors(self, "gather", True)

    def run(self):
        torch.distributed.gather(self.x, gather_list=self.x_list)


class AllGather(CollectiveOp):
    def prepare(self, tensor_size, local_rank):
        CollectiveOp.prepare(tensor_size, local_rank)
        CollectiveOp.prepare_tensors(self, "all_gather", True)

    def run(self):
        torch.distributed.all_gather(self.x_list, self.x)


class Scatter(CollectiveOp):
    def prepare(self, tensor_size, local_rank):
        CollectiveOp.prepare(tensor_size, local_rank)
        CollectiveOp.prepare_tensors(self, "scatter", True)

    def run(self):
        torch.distributed.scatter(self.x, scatter_list=self.x_list)


class ReduceScatter(CollectiveOp):
    def prepare(self, tensor_size, local_rank):
        CollectiveOp.prepare(tensor_size, local_rank)
        CollectiveOp.prepare_tensors(self, "reduce_scatter", True)

    def run(self):
        torch.distributed.reduce_scatter(self.x, input_list=self.x_list)


class Broadcast(CollectiveOp):
    def prepare(self, tensor_size, local_rank):
        CollectiveOp.prepare(tensor_size, local_rank)
        CollectiveOp.prepare_tensors(self, "broadcast")

    def run(self):
        torch.distributed.broadcast(self.x, src=0)


class AllToAll(CollectiveOp):
    def prepare(self, tensor_size, local_rank):
        CollectiveOp.prepare(tensor_size, local_rank)
        self.output = CollectiveOp.prepare_one_tensor(self, "all_to_all_single[output]")
        self.input = CollectiveOp.prepare_one_tensor(self, "all_to_all_single[input]")

    def run(self):
        torch.distributed.all_to_all_single(self.output, self.input)


class Barrier(CollectiveOp):
    def prepare(self, tensor_size, local_rank):
        pass

    def run():
        torch.distributed.barrier()


class SendReceive(CollectiveOp):
    def prepare(self, tensor_size, local_rank):
        CollectiveOp.prepare(tensor_size, local_rank)
        self.output = CollectiveOp.prepare_one_tensor(self, "recv")
        self.input = CollectiveOp.prepare_one_tensor(self, "send")

    def run(self, target_rank):
        r = torch.distributed.irecv(self.output)
        torch.distributed.send(self.input, target_rank)
        r.wait()

is_master = None
local_rank = None


def print_nccl_env_vars():
    for k in os.environ:
        if k.startswith("NCCL"):
            logging.info(k, "=", os.environ[k])


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
    logging.info(f"torch.distributed initialized, backend: \"{backend}\", " +
                 f"size {torch.distributed.get_world_size()}, " +
                 f"rank {torch.distributed.get_rank()}")

    is_master = torch.distributed.get_rank() == 0

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print_nccl_env_vars()


def run_collective_loop(collective, tensor_size, loop=1000, inner_loop=10):
    logging.info("{} will loop {} * {} times"
                 .format(collective.__name__, loop, inner_loop))

    warming_up = True
    if is_master:
        logging.info("Warming up ...")

    for _ in range(loop + 1):
        t_begin = time.time()

        for _ in range(inner_loop):
            collective.run()
            torch.cuda.synchronize()

        t_end = time.time()

        if is_master:
            logging.info("{:.1f} GB/s".format(
                         tensor_size * 4 * inner_loop / (t_end - t_begin) /
                         (1024 * 1024 * 1024)))

        if warming_up:
            warming_up = False
            if is_master:
                logging.info("Start testing now ...")

    if is_master:
        logging.info("Testing has ended.")


def run_p2p_loop(send_recv, tensor_size, stride=1, loop=1000, inner_loop=10):
    myrank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    for _ in range(loop + 1):
        times = []
        for delta in range(1, world_size, stride):
            target_rank = (myrank + delta) % world_size
            t = time.time()
            for _ in range(inner_loop):
                send_recv.run(target_rank)
                torch.cuda.synchronize()
            t = time.time() - t
            times += t
        if myrank > 0:
            torch.distributed.gather_object(times)
        else:
            # I'm master, gather all times from all ranks
            all_times = [0] * (len(times) * world_size)
            torch.distributed.gather_object(times, all_times)
            # Find straggers


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%m-%d %H:%M:%S')

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
    case "send_recv":
        op = SendReceive()
    case "broadcast":
        op = Broadcast()
    case "all_reduce":
        op = AllReduce()
    case "reduce":
        op = Reduce()
    case "all_gather":
        op = AllGather()
    case "gather":
        op = Gather()
    case "scatter":
        op = Scatter()
    case "reduce_scatter":
        op = ReduceScatter()
    case "all_to_all":
        op = AllToAll()
    case "barrier":
        op = Barrier()

op.prepare(args.size, local_rank)

if args.collective == "send_receive":
    run_p2p_loop(op, tensor_size=args.size, loop=args.loop)
else:
    run_collective_loop(op, tensor_size=args.size, loop=args.loop)
