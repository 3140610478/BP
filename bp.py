from __future__ import annotations
import typing
from typing import overload
from abc import ABC, abstractmethod
import numpy as np
import ucimlrepo
import pickle


class Dataset:
    Classes = {'Iris-setosa': np.array([[1, 0, 0]]).T,
               'Iris-versicolor': np.array([[0, 1, 0]]).T,
               'Iris-virginica': np.array([[0, 0, 1]]).T}

    @staticmethod
    def fetch_data(Classes: dict[str, list[int]]) -> list:
        # fetch dataset
        iris = ucimlrepo.fetch_ucirepo(id=53)

        # data (as pandas dataframes)
        X = iris.data.features
        y = iris.data.targets

        # # metadata
        # print(iris.metadata)

        # # variable information
        # print(iris.variables)

        X = X.values
        _y = []
        for i in y.values:
            _y.append(Classes[i[0]])
        y = np.array(_y)

        return list(zip(X, y))
    Original_data = fetch_data(Classes)
    Size = len(Original_data)

    def __init__(self, split_point: float = 0.75):
        shuffled = self.Original_data[:]
        np.random.shuffle(shuffled)
        split = int(split_point * self.Size)
        self.train, self.test = shuffled[:split:], shuffled[split::]

    def reshuffle(self):
        np.random.shuffle(self.train)
        np.random.shuffle(self.test)


class Layer(ABC):
    ...


class Layer(ABC):
    def __init__(self, size: int, pre=None):
        self.size = size
        self.output = None
        self.pre = pre

    @abstractmethod
    def __call__(self, x: np.ndarray[np.float64]) -> np.ndarray:
        pass

    @abstractmethod
    def attach(self, pre: Layer) -> None:
        pass

    @abstractmethod
    def backward(self, eta: float, feedback: np.ndarray) -> np.ndarray:
        pass


class Input(Layer):
    def __init__(self, size: int):
        super().__init__(size, None)

    def __call__(self, x) -> np.ndarray:
        self.output = x.reshape([self.size, 1])
        return self.output.copy()

    def attach(self, pre: Layer) -> None:
        self.pre = None

    def backward(self, eta, feedback):
        return np.zeros([1, self.size])


class Linear(Layer):
    def __init__(self, size: int, pre: Layer | None = None):
        super().__init__(size, pre)

    def __call__(self, x):
        self.output = self.W @ x + self.b
        return self.output.copy()

    def attach(self, pre: Layer):
        self.pre = pre
        self.W = np.random.rand(self.size, pre.size)
        self.b = np.random.rand(self.size, 1)

    def backward(self, eta, feedback):
        W = self.W.copy()
        input = self.pre.output
        self.W -= eta * np.transpose(input @ feedback)
        self.b -= eta * np.transpose(feedback)
        return feedback @ W


class Sigmoid(Layer):
    def __init__(self, size: int = 0, pre: Layer | None = None):
        super().__init__(size, pre)

    def __call__(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output.copy()

    def attach(self, pre: Layer):
        self.pre = pre
        self.size = pre.size
        self.output = np.zeros([self.size, 1])

    def backward(self, eta, feedback):
        s = self.output.T
        return feedback * s * (1 - s)


# unmodified yet

class ReLU(Layer):
    def __init__(self, size: int = 0, pre: Layer | None = None):
        super().__init__(size, pre)

    def __call__(self, x) -> np.ndarray:
        self.output = np.max(np.stack([x, np.zeros(x.shape)], axis=1), axis=1)
        self.output.reshape(x.shape)
        return self.output.copy()

    def attach(self, pre: Layer):
        self.pre = pre
        self.size = pre.size
        self.output = np.zeros([self.size, 1])

    def backward(self, eta, feedback):
        r = np.array([0 if i[0] <= 0 else 1 for i in self.pre.output])
        r.resize([1, self.size])
        return r * feedback


class Softmax(Layer):
    def __init__(self, size: int = 0, pre: Layer | None = None):
        super().__init__(size, pre)

    def __call__(self, x):
        x = x - x.min()
        x = np.exp(x)
        self.output = x / float(x.sum())
        return self.output.copy()

    def attach(self, pre: Layer):
        self.pre = pre
        self.size = pre.size
        self.output = np.zeros([self.size, 1])

    def backward(self, eta, feedback):
        g = np.diag(self.output.reshape(self.size)) - \
            self.output @ self.output.T
        return feedback @ g


class LayerNorm(Layer):
    def __init__(self, size: int = 0, pre: Layer | None = None):
        super().__init__(size, pre)
        self.mu, self.sigma = float(0), float(0)
        
    def __call__(self, x) -> np.ndarray:
        self.mu, self.sigma = np.mean(x), np.var(x)
        self.output = (x - self.mu) / self.sigma
        return self.output.copy()
    
    def attach(self, pre: Layer):
        self.pre = pre
        self.size = pre.size
        self.output = np.zeros([self.size, 1])
        
    def backward(self, eta, feedback):
        n, mu, sigma = self.size, self.mu, self.sigma
        jacobian = (- 1 / n + mu / (n * sigma**2) * (self.pre.output - mu) + np.eye(n)).T / sigma
        return feedback @ jacobian
# =================================================================================================


class Network:
    @overload
    def __init__(self, filename: str): ...
    @overload
    def __init__(self, l: list[Layer]): ...

    def __init__(self, arg):
        if type(arg) == list:
            tmp = None
            for i in arg:
                if not isinstance(i, Layer):
                    raise TypeError(
                        "Illegal argument passed to bp.Network.__init__")
                i.attach(tmp)
                tmp = i
            self.layers = arg
            self.size = len(arg)
        elif type(arg) == str:
            tmp = self.load_from(arg)
            self.layers, self.size = tmp.layers, tmp.size
        else:
            raise TypeError("Illegal argument passed to bp.Network.__init__")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        tmp = x
        for layer in self.layers:
            tmp = layer(tmp)
        return tmp

    def backward(self, eta, grad):
        for i in range(self.size):
            grad = self.layers[-1 - i].backward(eta, grad)
        return grad

    @staticmethod
    def CrossEntropy(output: np.ndarray, label: np.ndarray, acc: float = 1e-5) -> tuple[float, np.ndarray]:
        size = max(output.shape)
        output.resize(size)
        label.resize(size)
        loss = (-label * np.log(output + acc)
                - (1 - label) * np.log(1 - output + acc)).sum()
        grad = -label / (output+acc) + (1-label) / (1-output+acc)
        grad.resize([1, size])
        return loss, grad

    def train_batch(self, samples: list[tuple[np.ndarray, np.ndarray]], backward: bool = False, eta: float | typing.Callable[[int], float] = 0.001) -> tuple[float, float]:
        total_loss = 0
        n = len(samples)
        e = eta if isinstance(eta, typing.Callable) else lambda x: eta
        acc = 0
        for i in range(n):
            x, label = samples[i]
            output = self(x)
            if np.argmax(output) == np.argmax(label):
                acc += 1
            loss, grad = self.CrossEntropy(output, label)
            total_loss += loss
            if backward:
                self.backward(e(i), grad)
        return total_loss / n, acc / n

    def save_to(self, filename: str) -> None:
        with open(filename, mode='wb') as file:
            pickle.dump(self, file)

    @classmethod
    def save_to(cls, obj, filename: str) -> None:
        with open(filename, mode='wb') as file:
            pickle.dump(obj, file)

    @classmethod
    def load_from(cls, filename: str) -> Network:
        with open(filename, mode='rb') as file:
            return pickle.load(file)


if __name__ == '__main__':
    dataset = Dataset()
    bpnn = Network([
        Input(4),
        Linear(8),
        Sigmoid(),
        Linear(8),
        Sigmoid(),
        Linear(3),
        Softmax()
    ])
    for i in range(1, 2001):
        rtrain, rtest = bpnn.train_batch(
            dataset.train, True, 0.01), bpnn.train_batch(dataset.test, False)
        dataset.reshuffle()
        if i % 10 == 0:
            print(i, rtrain, rtest)
    bpnn.save_to("test.bpnn")
    bpnn = Network.load_from("test.bpnn")
    pass
