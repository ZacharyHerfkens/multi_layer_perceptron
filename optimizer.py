from __future__ import annotations
from itertools import chain
import random
from typing import Callable, Protocol
from prelude import *


class Evaluate[T](Protocol):
    def eval(self, candidate: T, data: Data) -> float: ...


class Mutate[T](Protocol):
    def mutate(self, item: T) -> T: ...


class DataProvider(Protocol):
    def next_batch(self) -> Data: ...


class ListData:
    def __init__(self, data: Data, batch_size: int | None = None) -> None:
        self.data = data
        self.batch_size = batch_size
    
    def next_batch(self) -> Data:
        if not self.batch_size:
            return self.data
        return random.sample(self.data, k=self.batch_size)


def optimize[
    I
](
    init: I,
    data_provider: DataProvider,
    mutator: Mutate[I],
    evaluator: Evaluate[I],
    epochs: int = 1000,
    samples: int = 25,
    on_epoch_complete: Callable[[int, I, float], None] | None = None,
) -> I:
    def step(seed: I) -> tuple[I, float]:
        batch = data_provider.next_batch()
        candidates = chain(
            (seed,),
            (mutator.mutate(seed) for _ in range(samples)),
        )
        tests = ((c, evaluator.eval(c, batch)) for c in candidates)
        return min(tests, key=lambda t: t[1])

    best = init
    for epoch in range(epochs):
        best, cost = step(best)
        if on_epoch_complete:
            on_epoch_complete(epoch, best, cost)

    return best
