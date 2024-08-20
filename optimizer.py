from __future__ import annotations
from itertools import chain
import random
from prelude import *
from mlp import MLP

@dataclass(frozen=True, kw_only=True)
class Optimizer:
    data: list[tuple[arr, arr]] = field(kw_only=False)
    epochs: int = 1000
    attempts: int = 25
    chunk_size: int | None = None
    mut_rate: float = 0.1
    mut_scale: float = 0.01

    def _new_chunk(self) -> list[tuple[arr, arr]]:
        if self.chunk_size is None:
            return self.data

        return random.sample(self.data, k=self.chunk_size)        

    @staticmethod
    def _test_chunk(mlp: MLP, chunk: list[tuple[arr, arr]]) -> float:
        return sum(sum((y - mlp.forward(x))**2) for x, y in chunk)

    def optimize(self, mlp: MLP) -> MLP:
        chunk = self._new_chunk()
        best = (mlp, self._test_chunk(mlp, chunk))

        for _ in range(self.epochs):
            candidates = (best[0].mutate(self.mut_scale, self.mut_rate) for _ in range(self.attempts))
            results = ((c, self._test_chunk(c, chunk)) for c in candidates)
            best = min(*results, best, key=lambda t: t[1])
            chunk = self._new_chunk()
        
        return best[0]
    
    def test(self, mlp: MLP) -> float:
        return self._test_chunk(mlp, self.data)