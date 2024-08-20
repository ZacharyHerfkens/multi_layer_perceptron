from __future__ import annotations
import random
from prelude import *
from mlp import MLP

@dataclass(frozen=True, kw_only=True)
class Optimizer:
    data: list[tuple[arr, arr]] = field(kw_only=False)
    epochs: int = 1000
    attempts: int = 25
    batch_size: int | None = None
    mut_rate: float = 0.1
    mut_scale: float = 0.01

    def _random_batch(self) -> list[tuple[arr, arr]]:
        if self.batch_size is None:
            return self.data

        return random.sample(self.data, k=self.batch_size)        

    @staticmethod
    def _test_batch(mlp: MLP, chunk: list[tuple[arr, arr]]) -> float:
        return sum(sum((y - mlp.forward(x))**2) for x, y in chunk)

    def optimize(self, mlp: MLP) -> MLP:
        chunk = self._random_batch()
        best = (mlp, self._test_batch(mlp, chunk))

        for _ in range(self.epochs):
            candidates = (best[0].mutate(self.mut_scale, self.mut_rate) for _ in range(self.attempts))
            results = ((c, self._test_batch(c, chunk)) for c in candidates)
            best = min(*results, best, key=lambda t: t[1])
            chunk = self._random_batch()
        
        return best[0]
    
    def test(self, mlp: MLP, /, data: list[tuple[arr, arr]] | None = None) -> float:
        return self._test_batch(mlp, data or self.data)