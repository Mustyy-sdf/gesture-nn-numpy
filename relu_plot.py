import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(x, 0)


# 201 точка от 0 (кулак) до 1 (раскрытая ладонь)
x = np.linspace(0, 1, 201)

# Первый слой: три нейрона, у каждого вес 1 и своё смещение, потом ReLU.
# Второй слой: один нейрон складывает их выходы с весами 1, -2, 1.
# n1, n2, n3 — вклад каждого нейрона в итог (уже умноженный на вес второго слоя).
n1 = relu(x - 0.4)
n2 = -2 * relu(x - 0.5)
n3 = relu(x - 0.6)

with_relu = n1 + n2 + n3
without_relu = (x - 0.4) - 2 * (x - 0.5) + (x - 0.6)

# Окно с двумя графиками рядом: left — слева, right — справа
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

# Левый график: каждый нейрон по отдельности, видно три «шарнира»
left.plot(x, n1, label='нейрон 1: ReLU(x - 0.4)')
left.plot(x, n2, label='нейрон 2: -2 * ReLU(x - 0.5)')
left.plot(x, n3, label='нейрон 3: ReLU(x - 0.6)')
left.set_title('Три нейрона по отдельности')
left.legend()

# Правый график: сумма с ReLU и без неё
right.plot(x, with_relu, linewidth=3, label='с ReLU')
right.plot(x, without_relu, '--', label='без ReLU')
right.set_title('Сумма трёх нейронов')
right.legend()

# Общие настройки для обоих графиков
for ax in (left, right):
    ax.set_xlabel('x: насколько раскрыта рука')
    ax.grid(True)

plt.tight_layout()
plt.show()
