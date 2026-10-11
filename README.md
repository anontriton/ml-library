# ml-library

A small neural network library written from scratch in C: matrix math,
reverse-mode automatic differentiation over a computational graph, and
mini-batch SGD. No dependencies beyond libc and libm.

The included example trains a 3-layer network (with a residual connection) on
MNIST and reaches **~93-94% test accuracy in about 15 seconds** on a laptop CPU.

## Build and run

Requires a C11 compiler on macOS or Linux, and Python to fetch the dataset.

```sh
# 1. download MNIST and convert it to .mat files (one-time)
python3 -m venv .venv && source .venv/bin/activate
pip install numpy tensorflow-datasets
python3 mnist.py

# 2. build and train
make run
```

Other targets: `make debug` builds with AddressSanitizer and UBSan,
`make clean` removes build outputs.

## How it works

- **Memory**: all allocations come from arenas that reserve a large range of
  virtual memory up front and commit pages as needed. Nothing is freed
  individually.
- **Math layer**: row-major `f32` matrices with add, matmul (all four
  transpose variants, no copies), ReLU, softmax and cross-entropy, plus the
  backward pass for each.
- **Autograd**: the model is built as a graph of `model_var` nodes. A
  DFS-based topological sort compiles it into a flat program once; the
  forward pass walks it front to back and backprop walks it back to front,
  accumulating gradients with the chain rule.
- **Training**: mini-batch SGD with per-epoch shuffling and Xavier
  initialization.

## What I added

This started by following
[Magicalbat's machine learning video](https://github.com/Magicalbat/videos).
On top of that I added:

- a versioned binary file format (`MAT1` header) with shape validation
- input validation for labels and dataset shapes
- numerically stable softmax (max subtraction) and clamped cross-entropy
- clock-based RNG seeding
- redesign the core so that the graph only builds nodes, the model is given
  its input/output/cost nodes explicitly and training is its own layer

## Roadmap

- Split the single `main.c` into library modules (`src/`) and an MNIST
   example (`examples/`).
- Hardening: check every return value and allocation, an unbiased
   (Fisher-Yates) shuffle, an O(n) softmax backward pass, and a seed that
   can be passed on the command line for reproducible runs.
- Gradient-check tests for every op.
- A character-level tokenizer.
- Single-head causal self-attention, then a small transformer trained to
   generate text.

My study notes are in [NOTES.md](NOTES.md).
