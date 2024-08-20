from __future__ import annotations
from prelude import *
from mlp import MLP, Layer
from optimizer import Optimizer


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def main():
    data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    data = [(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)) for (x, y) in data]
    untrained = MLP([
        Layer.random(2, 3),
        Layer.random(3, 1, s=sigmoid)
    ])

    opt = Optimizer(data, mut_rate=0.5, mut_scale=0.1)

    trained = opt.optimize(untrained)

    for x, y in data:
        y_untrained = untrained.forward(x)
        y_trained = trained.forward(x)
        print(f"{x} -> {y}\n\tuntrained - \t{y_untrained}\n\ttrained - \t{y_trained}")
    
    print(f"cost - untrained: {opt.test(untrained)}, trained: {opt.test(trained)}")


if __name__ == "__main__":
    main()