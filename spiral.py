import numpy as np
import matplotlib.pyplot as plt


def spiral_data(samples, classes):
    # Точки нескольких классов, закрученные в спирали.
    # samples — сколько точек в каждом классе, classes — сколько классов.
    # X: (samples * classes, 2) — координаты точек; y: (samples * classes,) — номер класса.
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype=int)
    for c in range(classes):
        start, end = samples * c, samples * (c + 1)
        r = np.linspace(0.0, 1.0, samples)  # расстояние от центра: от 0 до 1
        t = np.linspace(c * 4, (c + 1) * 4, samples) + np.random.randn(samples) * 0.2  # угол + немного шума
        X[start:end, 0] = r * np.sin(t * 2.5)
        X[start:end, 1] = r * np.cos(t * 2.5)
        y[start:end] = c
    return X, y


# Этот блок работает только при запуске самого файла, а при import пропускается
if __name__ == '__main__':
    np.random.seed(0)
    X, y = spiral_data(100, 3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', s=15)
    plt.title('Спираль: 3 класса по 100 точек')
    plt.axis('equal')
    plt.show()
