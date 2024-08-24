from __future__ import annotations
from itertools import chain
import random
from typing import Callable
from prelude import *
from mlp import MLP

type Data = list[tuple[arr, arr]]


@dataclass(frozen=True, kw_only=True)
class OptimizerSettings:
    epochs: int = 1000
    batch_size: int | None = None
    samples: int = 25
    mut_rate: float = 0.5
    mut_scale: float = 0.1
    on_epoch_complete: Callable[[int, MLP, float], None] | None = None


def test(candidate: MLP, batch: Data) -> float:
    return sum(sum((y - candidate.forward(x)) ** 2) for x, y in batch)


def optimize(settings: OptimizerSettings, init: MLP, data: Data) -> MLP:
    def random_batch() -> Data:
        if settings.batch_size is None:
            return data
        return random.sample(data, settings.batch_size)

    def step(seed: MLP) -> tuple[MLP, float]:
        batch = random_batch()
        candidates = chain(
            (seed,),
            (
                seed.mutate(settings.mut_scale, settings.mut_rate)
                for _ in range(settings.samples)
            ),
        )
        tests = ((c, test(c, batch)) for c in candidates)
        return min(tests, key=lambda t: t[1])

    best = init
    for epoch in range(settings.epochs):
        best, cost = step(best)
        if settings.on_epoch_complete:
            settings.on_epoch_complete(epoch, best, cost)

    return best
