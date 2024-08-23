from __future__ import annotations
from itertools import chain
from multiprocessing.connection import PipeConnection
import random
from multiprocessing import Manager, Process, Pipe
from prelude import *
from mlp import MLP

type Data = list[tuple[arr, arr]]

def _optimize_async(conn: PipeConnection, opt: Optimizer, mlp: MLP, data: Data) -> None:
    opt_mlp = opt.optimize(mlp, data)
    conn.send(opt_mlp)


@dataclass(frozen=True, kw_only=True)
class Optimizer:
    epochs: int = 1000
    attempts: int = 25
    batch_size: int | None = None
    mut_rate: float = 0.1
    mut_scale: float = 0.01

    def _random_batch(self, data: Data) -> Data:
        if self.batch_size is None:
            return data

        return random.sample(data, k=self.batch_size)        

    @staticmethod
    def test(mlp: MLP, batch: Data) -> float:
        return sum(sum((y - mlp.forward(x))**2) for x, y in batch)

    def optimize(self, mlp: MLP, data: Data) -> MLP:
        best = self._single_step_optimize(mlp, data)

        for _ in range(1, self.epochs):
            best = self._single_step_optimize(best[0], data)
        
        return best[0]
    
    def optimize_async(self, mlp: MLP, data: Data) -> OptimizerJob:
        return OptimizerJob(self, mlp, data)
    
    def _single_step_optimize(self, mlp: MLP, data: Data) -> tuple[MLP, float]:
        batch = self._random_batch(data)
        candidates = chain((mlp,), (mlp.mutate(self.mut_scale, self.mut_rate) for _ in range(self.attempts)))
        results = ((c, self.test(c, batch)) for c in candidates)
        return min(results, key=lambda t: t[1])


@dataclass(init=False)
class OptimizerJob:
    recv: PipeConnection
    proc: Process

    def __init__(self, opt: Optimizer, mlp: MLP, data: Data) -> None:
        self.recv, send = Pipe(duplex=False)
        self.proc = Process(target=_optimize_async, args=(send, opt, mlp, data))
        self.proc.start()

    def wait_done(self) -> MLP:
        self.proc.join()
        return self.recv.recv() 