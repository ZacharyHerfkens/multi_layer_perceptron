### A simple implementation of a multi-layer perceptron trained using stocastic gradient decent
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from regex import B

type arr = NDArray[np.float32]

@dataclass(frozen=True)
class Layer:
    w: arr
    b: arr
    s: Callable[[arr], arr] = field(default=np.tanh)

    @staticmethod
    def random(x_dim: int, y_dim: int, /, mu: float = 1.0, s: Callable[[arr], arr] = np.tanh) -> Layer:
        w = np.random.randn(y_dim, x_dim).astype(np.float32) * mu
        b = np.random.randn(y_dim).astype(np.float32) * mu
        return Layer(w, b, s)
    
    def forward(self, x: arr) -> arr:
        return self.s(x @ self.w + self.b)
    

@dataclass(frozen=True)
class MLP:
    layers: list[Layer]

    def forward(self, x: arr) -> arr:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def test(self, data: list[tuple[arr, arr]]) -> float:
        err = 0.0
        for x, y in data:
            y_act = self.forward(x)
            item_err = ((y - y_act) ** 2).sum()
            err += item_err
        
        return err
    
    def train(self, data: list[tuple[arr, arr]], /, epochs: int = 1000, attempts: int = 25, rate: float = 0.01) -> MLP:
        best = (self, self.test(data))

        for _ in range(epochs):
            candidates = (best[0].mutate(rate) for _ in range(attempts))
            tests = ((candidate, candidate.test(data)) for candidate in candidates)
            best_candidate = min(tests, key=lambda t: t[1]) 
            best = min(best_candidate, best, key=lambda t: t[1])
        return best[0]
    
    def mutate(self, rate: float = 0.01) -> MLP:
        mut_layers = []
        for layer in self.layers:
            mut_w = layer.w + np.random.randn(*layer.w.shape) * rate
            mut_b = layer.b + np.random.randn(*layer.b.shape) * rate
            mut_layers.append(Layer(mut_w, mut_b, layer.s))
        return MLP(mut_layers)