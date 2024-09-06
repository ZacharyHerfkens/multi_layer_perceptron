### A simple implementation of a multi-layer perceptron trained using stochastic gradient descent
from __future__ import annotations
from prelude import *
from typing import Callable


class MlpMutator:
    def __init__(self, rate: float = 0.1, scale: float = 0.05) -> None:
        self.rate = rate
        self.scale = scale
    
    def mutate(self, item: MLP) -> MLP:
        return item.mutate(self.scale, self.rate)


class MlpEvaluator:
    def eval(self, candidate: MLP, data: Data) -> float:
        return sum(sum((y - candidate.forward(x))**2) for x, y in data)


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
        return self.s(self.w @ x + self.b)
    

@dataclass(frozen=True)
class MLP:
    layers: list[Layer]

    def forward(self, x: arr) -> arr:
        for layer in self.layers:
            x = layer.forward(x)
        return x
     
    def mutate(self, scale: float, rate: float) -> MLP:
        mut_layers = []
        for layer in self.layers:
            mut_w = layer.w + np.random.randn(*layer.w.shape) * scale * (np.random.rand(*layer.w.shape) < rate)
            mut_b = layer.b + np.random.randn(*layer.b.shape) * scale * (np.random.rand(*layer.b.shape) < rate)
            mut_layers.append(Layer(mut_w, mut_b, layer.s))
        return MLP(mut_layers)