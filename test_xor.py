from __future__ import annotations
import numpy as np
from mlp import MLP, Layer

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
    net = MLP([
        Layer.random(2, 3),
        Layer.random(3, 1, s=sigmoid)
    ])

    trained = net.train(data)

    for x, y in data:
        y_untrained = net.forward(x)
        y_trained = trained.forward(x)
        print(f"{x} ->\n\tuntrained - \t{y_untrained}\n\ttrained - \t{y_trained}")
    
    print(f"cost - untrained: {net.test(data)}, trained: {trained.test(data)}")


if __name__ == "__main__":
    main()