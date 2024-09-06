from __future__ import annotations
from prelude import *
from mlp import MLP, Layer, MlpEvaluator, MlpMutator
from optimizer import optimize, ListData


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def print_progress(epoch: int, _best: MLP, cost: float) -> None:
    if epoch % 5 != 0:
        return
    print(f"{epoch: >5d}: {cost}")


def main():
    data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    data = [
        (np.array(x, dtype=np.float32), np.array(y, dtype=np.float32))
        for (x, y) in data
    ]
    untrained = MLP([Layer.random(2, 3), Layer.random(3, 1, s=sigmoid)])
    mutator = MlpMutator()
    eval = MlpEvaluator()

    print("begin training...")
    trained = optimize(untrained, ListData(data), mutator, eval, on_epoch_complete=print_progress)

    for x, y in data:
        y_untrained = untrained.forward(x)
        y_trained = trained.forward(x)
        print(f"{x} -> {y}\n\tuntrained - \t{y_untrained}\n\ttrained - \t{y_trained}")

    print(f"cost - untrained: {eval.eval(untrained, data)}, trained: {eval.eval(trained, data)}")


if __name__ == "__main__":
    main()
