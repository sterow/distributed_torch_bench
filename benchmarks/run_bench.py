import argparse
import bisect
import logging
import os
import sys
import time
import torch
from abc import abstractmethod, ABC


def to_readable_number(n, sep=" "):
    if n >= 1024 * 1024 * 1024 * 1024:
        unit = "T"
        n /= 1024 * 1024 * 1024 * 1024
    elif n >= 1024 * 1024 * 1024:
        unit = "G"
        n /= 1024 * 1024 * 1024
    elif n >= 1024 * 1024:
        unit = "M"
        n /= 1024 * 1024
    elif n >= 1024:
        unit = "K"
        n /= 1024
    else:
        unit = ""
    format = ("{:.0f}" if n >= 10 else "{:.1f}") + sep + unit
    return format.format(n)


def readable_number_to_int(s):
    unit = s[-1:]
    if unit >= '0' and unit <= '9' or unit == '.':
        return int(s)
    s = s[:-1]
    if unit == 'K' or unit == 'k':
        unit = 1024
    elif unit == 'M' or unit == 'm':
        unit = 1024 * 1024
    elif unit == 'G' or unit == 'g':
        unit = 1024 * 1024 * 1024
    elif unit == 'T' or unit == 't':
        unit = 1024 * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Invalid unit {unit}, must be one of (K, M or G)")
    return int(float(s) * unit)


class CollectiveOp(ABC):
    display_name = None

    @abstractmethod
    def run(self):
        pass

    def prepare(self, tensor_bytes, local_rank):
        self.tensor_bytes = tensor_bytes
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank

    def prepare_one_tensor(self, tensor_name=None):
        logging.info("Generating {} message, size: {}B"
                     .format(tensor_name or self.display_name,
                             to_readable_number(self.tensor_bytes)))
        tensor = torch.zeros(self.tensor_bytes // 4)
        if torch.cuda.device_count() > 0:
            tensor = tensor.cuda(self.local_rank)
        return tensor

    def prepare_list_of_tensors(self, tensor_list_name=None):
        logging.info("Generating {} {} messages, size: {}B each"
                     .format(self.world_size, tensor_list_name or self.display_name,
                             to_readable_number(self.tensor_bytes)))
        x_list = []
        for _ in range(self.world_size):
            tensor = torch.zeros(self.tensor_bytes // 4)
            if torch.cuda.device_count() > 0:
                tensor = tensor.cuda(self.local_rank)
            x_list.append(tensor)

        return x_list

    def prepare_tensors(self, gather_or_scatter=False, with_all=False):
        if torch.cuda.device_count() > 0:
            total_tensor_bytes = self.tensor_bytes * \
                                 (1 + self.world_size if gather_or_scatter else 1)
            mem_info = torch.cuda.mem_get_info(self.local_rank)
            if total_tensor_bytes > mem_info[0]:
                logging.warn("Total tensors size {}B bigger than GPU avaliable memory {}B!"
                             .format(to_readable_number(total_tensor_bytes),
                                     to_readable_number(mem_info[0])))

        self.x = self.prepare_one_tensor()
        if gather_or_scatter:
            self.x_list = self.prepare_list_of_tensors() if with_all or self.rank == 0 else None


class Reduce(CollectiveOp):
    def prepare(self, tensor_bytes, local_rank):
        self.display_name = "reduce"
        super().prepare(tensor_bytes, local_rank)
        super().prepare_tensors()

    def run(self):
        torch.distributed.reduce(self.x, dst=0)


class AllReduce(CollectiveOp):
    def prepare(self, tensor_bytes, local_rank):
        self.display_name = "all_reduce"
        super().prepare(tensor_bytes, local_rank)
        super().prepare_tensors()

    def run(self):
        torch.distributed.all_reduce(self.x)


class Gather(CollectiveOp):
    def prepare(self, tensor_bytes, local_rank):
        self.display_name = "gather"
        super().prepare(tensor_bytes, local_rank)
        super().prepare_tensors(gather_or_scatter=True)

    def run(self):
        torch.distributed.gather(self.x, gather_list=self.x_list)


class AllGather(CollectiveOp):
    def prepare(self, tensor_bytes, local_rank):
        self.display_name = "all_gather"
        super().prepare(tensor_bytes, local_rank)
        super().prepare_tensors(gather_or_scatter=True, with_all=True)

    def run(self):
        torch.distributed.all_gather(self.x_list, self.x)


class Scatter(CollectiveOp):
    def prepare(self, tensor_bytes, local_rank):
        self.display_name = "scatter"
        super().prepare(tensor_bytes, local_rank)
        super().prepare_tensors(gather_or_scatter=True)

    def run(self):
        torch.distributed.scatter(self.x, scatter_list=self.x_list)


class ReduceScatter(CollectiveOp):
    def prepare(self, tensor_bytes, local_rank):
        self.display_name = "reduce_scatter"
        super().prepare(tensor_bytes, local_rank)
        super().prepare_tensors(gather_or_scatter=True, with_all=True)

    def run(self):
        torch.distributed.reduce_scatter(self.x, input_list=self.x_list)


class Broadcast(CollectiveOp):
    def prepare(self, tensor_bytes, local_rank):
        self.display_name = "broadcast"
        super().prepare(tensor_bytes, local_rank)
        super().prepare_tensors()

    def run(self):
        torch.distributed.broadcast(self.x, src=0)


class AllToAll(CollectiveOp):
    def prepare(self, tensor_bytes, local_rank):
        self.display_name = "all_to_all_single"
        super().prepare(tensor_bytes, local_rank)
        self.output = super().prepare_one_tensor("all_to_all_single[output]")
        self.input = super().prepare_one_tensor("all_to_all_single[input]")

    def run(self):
        torch.distributed.all_to_all_single(self.output, self.input)


class Barrier(CollectiveOp):
    def prepare(self, tensor_bytes, local_rank):
        self.display_name = "barrier"

    def run(self):
        torch.distributed.barrier()


class SendReceive(CollectiveOp):
    def prepare(self, tensor_bytes, local_rank):
        self.display_name = "send_recv"
        super().prepare(tensor_bytes, local_rank)
        self.recv = CollectiveOp.prepare_one_tensor(self, "recv")
        self.send = CollectiveOp.prepare_one_tensor(self, "send")

    def run(self, target_rank):
        r = torch.distributed.irecv(self.recv)
        torch.distributed.send(self.send, target_rank)
        r.wait()


is_master = None
local_rank = None


def print_nccl_env_vars():
    for k in os.environ:
        if k.startswith("NCCL"):
            logging.info("%s=%s", k, os.environ[k])


def init_torch_distributed():
    global world_size
    global is_master
    global local_rank

    if "TORCH_DISTRIBUTED_BACKEND" in os.environ:
        backend = os.environ["TORCH_DISTRIBUTED_BACKEND"]
    elif torch.cuda.device_count() == 0 or torch.cuda.nccl.version() is None:
        backend = "gloo"
    else:
        backend = "nccl"

    torch.distributed.init_process_group(backend=backend, init_method="env://")

    assert torch.distributed.is_initialized()
    logging.info(f"torch.distributed initialized, backend: \"{backend}\", " +
                 f"size {torch.distributed.get_world_size()}, " +
                 f"rank {torch.distributed.get_rank()}")

    world_size = torch.distributed.get_world_size()
    is_master = torch.distributed.get_rank() == 0
    local_rank = int(os.environ.get("LOCAL_RANK", default="0"))
    if local_rank == 0:
        print_nccl_env_vars()


def run_collective_loop(collective, tensor_bytes, loop=1000, inner_loop=10):
    logging.info("{} will loop {} * {} times"
                 .format(collective.display_name, loop, inner_loop))

    warming_up = True
    if is_master:
        logging.info("Warming up ...")

    for _ in range(loop + 1):

        t_begin = time.time()

        for _ in range(inner_loop):
            collective.run()
            if torch.cuda.device_count() > 0:
                torch.cuda.synchronize()

        t_end = time.time()

        if is_master:
            logging.info(f"{to_readable_number(tensor_bytes * inner_loop / (t_end - t_begin))}B/s")

        if warming_up:
            warming_up = False
            if is_master:
                logging.info("Start testing now ...")

    if is_master:
        logging.info("Testing has ended.")


def run_p2p_loop(send_recv, tensor_bytes, stride=1, loop=1000, inner_loop=10, lag_limitation=0.5):
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
            times += (t, myrank, target_rank)
        if myrank > 0:
            torch.distributed.gather_object(times)
        else:
            # I'm master, gather all times from all ranks
            n = (len(times) * world_size)
            all_times = [(0.0, 0, 0)] * n
            torch.distributed.gather_object(times, all_times)
            # Find straggers 30% larger than 50 percentile median
            all_times.sort(all_times, key=lambda x: x[0])
            median_pos = int(n * 0.5)
            median = all_times[median_pos]
            upper_bound = median[0] * (1 + lag_limitation)
            stragglers_pos = bisect.bisect_left(all_times, upper_bound,
                                                lo=median_pos, key=lambda x: x[0])

            logging.info("P2P test: median latency is {}s, median bandwidth is {}B/s".format(
                         median, to_readable_number(tensor_bytes * inner_loop / median)))

            if stragglers_pos == n:
                logging.info("No stragglers found within 50%% lag limitations")
            else:
                logging.warn("{:%} ({}/{}) as stragglers found within {:%} lag limitation".format(
                             (n-stragglers_pos)/n, n-stragglers_pos, n, lag_limitation))
                print_count = min(n-stragglers_pos, 10)
                logging.info(f"The largest {print_count} stragglers are:")
                logging.info("Stragger latency    From     To")
                for i in range(n - print_count, n):
                    s = all_times[i]
                    logging.info("{:.0f >16}    {: >8}    {: >8}".format(s[i][0], s[i][1], s[i][2]))


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%m-%d %H:%M:%S',
                    stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collective",
                    dest="collective",
                    choices=["send_recv", "broadcast", "all_reduce",
                             "reduce", "all_gather", "gather", "scatter",
                             "reduce_scatter", "all_to_all", "barrier"],
                    required=True,
                    help="Collective function name",
                    type=str)
parser.add_argument("-b", "--bytes",
                    dest="tensor_bytes",
                    default=64 * 1024 * 1024 * 1024,
                    help="Tensor size in bytes, can ends with K, M, or G",
                    type=readable_number_to_int)
parser.add_argument("-n", "--loop",
                    dest="loop",
                    default=1000,
                    help="Loop count",
                    type=int)
parser.add_argument("-s", "--stride",
                    dest="stride",
                    default=1,
                    help="P2P loop rank stride",
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

op.prepare(args.tensor_bytes, local_rank)

if args.collective == "send_receive":
    run_p2p_loop(op, tensor_bytes=args.tensor_bytes, stride=args.stride, loop=args.loop)
else:
    run_collective_loop(op, tensor_bytes=args.tensor_bytes, loop=args.loop)
