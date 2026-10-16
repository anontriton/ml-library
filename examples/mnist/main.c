#include "arena.h"
#include "base.h"
#include "graph.h"
#include "matrix.h"
#include "model.h"
#include "prng.h"
#include "train.h"

#include <errno.h>
#include <stdlib.h>
#include <time.h>

#define MNIST_IMG_SIDE 28 /* each digit is a 28 x 28 pixel grayscale image */
#define MNIST_IMG_SIZE (MNIST_IMG_SIDE * MNIST_IMG_SIDE) /* 784 */
/* digits 0-9, the width of a one-hot label row */
#define MNIST_NUM_CLASSES 10
/* width of each hidden layer, a model choice, not a fact about mnist */
#define MNIST_HIDDEN_SIZE 16

static b32 parse_seed(const char *str, u64 *out);
static void draw_mnist_digit(f32 *data);
static model_context *create_mnist_model(mem_arena *arena, model_graph *graph);

int main(int argc, char **argv) {
  /* usage: mnist [seed]
   * with no seed the rng is seeded from the clock so each run starts from
   * different weights, the seed is printed either way, so passing it back in
   * reproduces a run exactly
   */
  if (argc > 2) {
    fprintf(stderr, "usage: %s [seed]\n", argv[0]);
    return 1;
  }

  u64 seed = (u64)time(NULL);
  if (argc == 2 && !parse_seed(argv[1], &seed)) {
    fprintf(stderr, "seed must be a whole number from 0 to %llu, got \"%s\"\n",
            (unsigned long long)UINT64_MAX, argv[1]);
    return 1;
  }
  prng_seed(seed, 1);
  printf("seed: %llu\n", (unsigned long long)seed);

  /* 1 GiB is only reserved, pages get committed as the arena grows */
  mem_arena *perm_arena = arena_create(GiB(1), MiB(1));
  if (perm_arena == NULL) {
    fprintf(stderr, "failed to create arena\n");
    return 1;
  }

  /* mat_load returns NULL if a file is missing or has a bad header
   * early returns skip arena_destroy, the OS frees it when the process exits
   */
  matrix *train_images = mat_load(perm_arena, "data/train_images.mat");
  if (train_images == NULL) {
    fprintf(stderr, "failed to load train_images.mat (run tools/mnist.py?)\n");
    return 1;
  }
  matrix *test_images = mat_load(perm_arena, "data/test_images.mat");
  if (test_images == NULL) {
    fprintf(stderr, "failed to load test_images.mat (run tools/mnist.py?)\n");
    return 1;
  }

  matrix *train_labels_file = mat_load(perm_arena, "data/train_labels.mat");
  if (train_labels_file == NULL) {
    fprintf(stderr, "failed to load train_labels.mat (run tools/mnist.py?)\n");
    return 1;
  }
  matrix *test_labels_file = mat_load(perm_arena, "data/test_labels.mat");
  if (test_labels_file == NULL) {
    fprintf(stderr, "failed to load test_labels.mat (run tools/mnist.py?)\n");
    return 1;
  }

  /* shape checks, everything below indexes by these shapes so a mismatched or
   * truncated file would read/write out of bounds instead of failing here
   * images: one 784-pixel row per sample, labels: one digit per sample
   */
  if (train_images->cols != MNIST_IMG_SIZE ||
      test_images->cols != MNIST_IMG_SIZE) {
    fprintf(stderr, "image files must have %u cols\n", MNIST_IMG_SIZE);
    return 1;
  }
  if (train_labels_file->cols != 1 || test_labels_file->cols != 1) {
    fprintf(stderr, "label files must have 1 col\n");
    return 1;
  }
  if (train_images->rows != train_labels_file->rows) {
    fprintf(stderr, "train set has %u images but %u labels\n",
            train_images->rows, train_labels_file->rows);
    return 1;
  }
  if (test_images->rows != test_labels_file->rows) {
    fprintf(stderr, "test set has %u images but %u labels\n", test_images->rows,
            test_labels_file->rows);
    return 1;
  }

  matrix *train_labels =
    mat_create(perm_arena, train_labels_file->rows, MNIST_NUM_CLASSES);
  matrix *test_labels =
    mat_create(perm_arena, test_labels_file->rows, MNIST_NUM_CLASSES);

  if (train_labels == NULL || test_labels == NULL) {
    fprintf(stderr, "failed to allocate the one-hot label matrices\n");
    return 1;
  }

  /* the label is used as an index into the one-hot row, so a bad value would
   * write into the next sample's row or past the end of the array
   * checked as a float first since converting a negative float to u32 is
   * undefined behavior in C
   */
  for (u32 i = 0; i < train_labels_file->rows; i++) {
    f32 val = train_labels_file->data[i];
    if (val < 0.0f || val >= MNIST_NUM_CLASSES) {
      fprintf(stderr, "train label %u is %f, expected 0-9\n", i, val);
      return 1;
    }
    u32 num = val;
    train_labels->data[i * MNIST_NUM_CLASSES + num] = 1.0f;
  }

  for (u32 i = 0; i < test_labels_file->rows; i++) {
    f32 val = test_labels_file->data[i];
    if (val < 0.0f || val >= MNIST_NUM_CLASSES) {
      fprintf(stderr, "test label %u is %f, expected 0-9\n", i, val);
      return 1;
    }
    u32 num = val;
    test_labels->data[i * MNIST_NUM_CLASSES + num] = 1.0f;
  }

  draw_mnist_digit(test_images->data);
  for (u32 i = 0; i < MNIST_NUM_CLASSES; i++) {
    printf("%.0f ", test_labels->data[i]);
  }
  printf("\n\n");

  /* on the stack, it only has to outlive model, and main outlives both */
  model_graph graph = {0};
  model_context *model = create_mnist_model(perm_arena, &graph);
  if (model == NULL) {
    return 1;
  }

  /* run test image 0 through the untrained model, with random weights the
   * guesses should come out roughly even (~0.1 each) */
  memcpy(model->input->val->data, test_images->data,
         sizeof(f32) * MNIST_IMG_SIZE);
  model_feedforward(model);
  printf("pre-training output:  ");
  for (u32 i = 0; i < MNIST_NUM_CLASSES; i++) {
    printf("%.2f ", model->output->val->data[i]);
  }
  printf("\n\n");

  model_training_desc training_desc = {
      .train_images = train_images,
      .train_labels = train_labels,
      .test_images = test_images,
      .test_labels = test_labels,

      .epochs = 10,
      .batch_size = 50,
      .learning_rate = 0.01f,
  };
  if (!model_train(model, &training_desc)) {
    return 1;
  }

  /* same image after training, the probability should pile up on its label */
  memcpy(model->input->val->data, test_images->data,
         sizeof(f32) * MNIST_IMG_SIZE);
  model_feedforward(model);
  printf("post-training output: ");
  for (u32 i = 0; i < MNIST_NUM_CLASSES; i++) {
    printf("%.2f ", model->output->val->data[i]);
  }
  printf("\n\n");

  arena_destroy(perm_arena);

  return 0;
}

/* strtoull alone is too forgiving: it returns 0 for "abc", stops quietly at
 * "42abc", clamps "99999999999999999999" to the max, and turns "-1" into
 * 2^64 - 1, so each of those gets checked here
 */
static b32 parse_seed(const char *str, u64 *out) {
  if (str[0] < '0' || str[0] > '9') {
    return false;
  }

  char *end;
  errno = 0;
  unsigned long long val = strtoull(str, &end, 10);
  if (errno == ERANGE || *end != '\0') {
    return false;
  }

  *out = val;
  return true;
}

/* prints one 28 x 28 image to the terminal, 2 spaces per pixel so it comes
 * out roughly square, with the pixel's brightness as the background color
 * data points at the image's first pixel, so pass images->data + n * 784 to
 * draw sample n
 */
static void draw_mnist_digit(f32 *data) {
  for (u32 y = 0; y < MNIST_IMG_SIDE; y++) {
    for (u32 x = 0; x < MNIST_IMG_SIDE; x++) {
      f32 num = data[x + y * MNIST_IMG_SIDE];
      /* 256-color grayscale ramp is codes 232-255 (24 shades), so scale the
       * 0-1 pixel by 23 to land on 232-255, 24 would make white = 256 */
      u32 col = 232 + (u32)(num * 23);
      printf("\x1b[48;5;%um  ", col);
    }
    printf("\n");
  }
  printf("\x1b[0m");
}

/* builds the network into graph and wraps it in a model, NULL on failure
 *   input (784) -> layer 0 (16) -> layer 1 (16) -> layer 2 (10) -> softmax
 * each layer is W * x + b, the hidden ones followed by a relu
 * shapes: W is (outputs x inputs) so W * x turns an (inputs x 1) column into
 * an (outputs x 1) column, b is (outputs x 1) to match
 * if a shape is wrong, mv_matmul/mv_add return NULL, the NULL carries through
 * every op after it, and model_create reports it, only the weights need a
 * check here since mat_fill_rand uses them directly
 */
static model_context *create_mnist_model(mem_arena *arena, model_graph *graph) {
  model_var *input = mv_create(arena, graph, MNIST_IMG_SIZE, 1, 0);

  model_var *W0 = mv_create(arena, graph, MNIST_HIDDEN_SIZE, MNIST_IMG_SIZE,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *W1 = mv_create(arena, graph, MNIST_HIDDEN_SIZE, MNIST_HIDDEN_SIZE,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *W2 = mv_create(arena, graph, MNIST_NUM_CLASSES, MNIST_HIDDEN_SIZE,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  if (W0 == NULL || W1 == NULL || W2 == NULL) {
    fprintf(stderr, "failed to create the weight matrices\n");
    return NULL;
  }

  /* xavier/glorot init: uniform in +-sqrt(6 / (fan_in + fan_out))
   * keeps each layer's outputs about the same size as its inputs, so values
   * don't blow up or shrink to nothing as they pass through the layers */
  f32 bound0 = sqrtf(6.0f / (MNIST_IMG_SIZE + MNIST_HIDDEN_SIZE));
  f32 bound1 = sqrtf(6.0f / (MNIST_HIDDEN_SIZE + MNIST_HIDDEN_SIZE));
  f32 bound2 = sqrtf(6.0f / (MNIST_HIDDEN_SIZE + MNIST_NUM_CLASSES));
  mat_fill_rand(W0->val, -bound0, bound0);
  mat_fill_rand(W1->val, -bound1, bound1);
  mat_fill_rand(W2->val, -bound2, bound2);

  model_var *b0 = mv_create(arena, graph, MNIST_HIDDEN_SIZE, 1,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *b1 = mv_create(arena, graph, MNIST_HIDDEN_SIZE, 1,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *b2 = mv_create(arena, graph, MNIST_NUM_CLASSES, 1,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

  /* biases start at 0 (PUSH_ARRAY zeroes), only the weights need to be
   * random to break the symmetry */

  /* layer 0: a0 = relu(W0 * input + b0) */
  model_var *z0_a = mv_matmul(arena, graph, W0, input, 0);
  model_var *z0_b = mv_add(arena, graph, z0_a, b0, 0);
  model_var *a0 = mv_relu(arena, graph, z0_b, 0);

  /* layer 1: a1 = a0 + relu(W1 * a0 + b1)
   * adding a0 back in is a residual (skip) connection, the layer only has to
   * learn a change to a0, and grads get a shortcut straight back through
   * the add, needs W1 square so both sides of the add have the same shape */
  model_var *z1_a = mv_matmul(arena, graph, W1, a0, 0);
  model_var *z1_b = mv_add(arena, graph, z1_a, b1, 0);
  model_var *z1_c = mv_relu(arena, graph, z1_b, 0);
  model_var *a1 = mv_add(arena, graph, a0, z1_c, 0);

  /* layer 2: output = softmax(W2 * a1 + b2), 10 probabilities, no relu
   * since softmax already turns the scores into the final answer */
  model_var *z2_a = mv_matmul(arena, graph, W2, a1, 0);
  model_var *z2_b = mv_add(arena, graph, z2_a, b2, 0);
  model_var *output = mv_softmax(arena, graph, z2_b, 0);

  /* training only: y holds the one-hot label, cost compares it to output
   * neither is in forward_prog since output doesn't depend on them */
  model_var *y = mv_create(arena, graph, MNIST_NUM_CLASSES, 1, 0);
  model_var *cost = mv_cross_entropy(arena, graph, y, output, 0);

  model_desc desc = {
      .graph = graph,
      .input = input,
      .output = output,
      .desired_output = y,
      .cost = cost,
  };

  return model_create(arena, &desc);
}
