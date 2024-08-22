from __future__ import annotations
import random
from prelude import *
from mlp import MLP

type Data = list[tuple[arr, arr]]

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
    def _test_batch(mlp: MLP, batch: Data) -> float:
        return sum(sum((y - mlp.forward(x))**2) for x, y in batch)

    def optimize(self, mlp: MLP, data: Data) -> MLP:
        batch = self._random_batch(data)
        best = (mlp, self._test_batch(mlp, batch))

        for _ in range(self.epochs):
            candidates = (best[0].mutate(self.mut_scale, self.mut_rate) for _ in range(self.attempts))
            results = ((c, self._test_batch(c, batch)) for c in candidates)
            best = min(*results, best, key=lambda t: t[1])
            batch = self._random_batch(data)
        
        return best[0]
    
    def test(self, mlp: MLP, data: Data) -> float:
        return self._test_batch(mlp, data)