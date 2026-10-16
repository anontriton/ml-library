# ml-library

A small neural network library written from scratch in C: matrix math,
reverse-mode automatic differentiation over a computational graph, and
mini-batch SGD. No dependencies beyond libc and libm.

The included example trains a 3-layer network (with a residual connection) on
MNIST and reaches **~93-94% test accuracy in about 15 seconds** on a laptop CPU.

## Build and run

Requires a C11 compiler on macOS or Linux, and Python to fetch the dataset.

```sh
# 1. download MNIST and convert it to .mat files in data/ (one-time)
python3 -m venv .venv && source .venv/bin/activate
pip install numpy tensorflow-datasets
python3 tools/mnist.py

# 2. build and train
make run
```

Run everything from the repo root, the example loads `data/*.mat` relative to
it. Each run prints its seed, and `make run SEED=42` (or
`./build/release/mnist 42`) reproduces a run exactly. Other targets:
`make debug` / `make run-debug` build and run with AddressSanitizer and UBSan,
`make clean` removes `build/`.

## Layout

```
src/              the library, built into build/<release|debug>/libml.a
  arena           memory arenas
  prng            PCG random number generator
  matrix          matrix ops and their backward passes
  graph           autograd: nodes, topological sort, forward and backprop
  model           wraps a graph's input/output/cost nodes and programs
  train           mini-batch SGD with a test pass per epoch
examples/mnist/   MNIST digit classifier, links against the library
tools/mnist.py    downloads MNIST and writes data/*.mat
```

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
- RNG seeded from the clock or from the command line, so any run can be
  reproduced from its printed seed
- redesign the core so that the graph only builds nodes, the model is given
  its input/output/cost nodes explicitly and training is its own layer
- split the single `main.c` into separately compiled library modules
  (`src/`, built as a static library) and an MNIST example (`examples/`)
- hardening: every allocation and op result is checked (graph building
  returns NULL up to a single error, a failing op while training aborts with
  the node), overflow-safe arena pushes and `.mat` shapes, an unbiased
  Fisher-Yates shuffle, and an O(n) softmax backward pass

## Roadmap

- Gradient-check tests for every op.
- A character-level tokenizer.
- Single-head causal self-attention, then a small transformer trained to
   generate text.

My study notes are in [NOTES.md](NOTES.md).
