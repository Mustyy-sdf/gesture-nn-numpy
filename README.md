# Hand gesture recognition with a neural network written from scratch in NumPy

> **Status: in development.** The neural network framework is being built step by step; the camera part comes next.
> Progress log: [diary.md](diary.md) (in Russian).

## Idea

Webcam → MediaPipe finds 21 hand landmarks → 42 normalized numbers → **my own neural network in pure NumPy** → gesture name on screen → action (switching presentation slides).

No PyTorch, TensorFlow or Keras: every layer, activation, loss and the backpropagation are written by hand, so I understand how a neural network actually works inside.

## Progress

- [x] Dense layer, ReLU, Softmax — forward pass
- [x] Categorical cross-entropy loss, accuracy
- [x] Backpropagation for Dense and ReLU, checked against numerical gradients
- [ ] Softmax + cross-entropy backward, SGD optimizer, training loop
- [ ] Training on the "spiral" toy dataset
- [ ] Camera + MediaPipe HandLandmarker, own gesture dataset
- [ ] Training on my gestures, comparison with scikit-learn
- [ ] Live demo and slide control

## Project structure

```
nn/            my framework: layers.py, activations.py, losses.py
spiral.py      toy dataset (3 intertwined spirals)
test_*.py      checks of the framework on known numbers
diary.md       development diary
docs/          diagrams
```

## How to run

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python test_forward.py
```

## About AI assistance

I study with an AI tutor (Claude). The core of the network (`nn/`, training code) is written by me; the tutor explains ideas, reviews my code and points out mistakes. Helper code written by the tutor (`spiral.py`, `relu_plot.py`, `test_dense.py`, `docs/backprop.svg`) is marked as such in the diary.

---

## Кратко по-русски

Распознавание жестов руки нейросетью, написанной с нуля на NumPy, без готовых библиотек машинного обучения. Проект в разработке, ход работы — в [дневнике](diary.md).
