#include "arena.h"
#include "prng.h"
#include "matrix.h"
#include "graph.h"

#include <time.h>

#define MNIST_IMG_SIDE 28 /* each digit is a 28 x 28 pixel grayscale image */
#define MNIST_IMG_SIZE (MNIST_IMG_SIDE * MNIST_IMG_SIDE) /* 784 */
/* digits 0-9, the width of a one-hot label row */
#define MNIST_NUM_CLASSES 10
/* width of each hidden layer, a model choice, not a fact about mnist */
#define MNIST_HIDDEN_SIZE 16

/* model layer
 * a graph's special nodes plus the two programs compiled from them
 * forward_prog: input -> output, used for inference
 * cost_prog: input + desired_output -> cost, used for training, it contains
 * the forward pass too since cost depends on output
 * only made by model_create, so every field is always set
 */
typedef struct {
  model_graph *graph;

  model_var *input;
  model_var *output;
  model_var *desired_output;
  model_var *cost;

  model_program forward_prog;
  model_program cost_prog;
} model_context;

/* everything model_create needs, all fields required */
typedef struct {
  model_graph *graph;

  model_var *input;
  model_var *output;
  model_var *desired_output;
  model_var *cost;
} model_desc;

typedef struct {
  matrix *train_images;
  matrix *train_labels;
  matrix *test_images;
  matrix *test_labels;

  u32 epochs;
  u32 batch_size;
  f32 learning_rate;
} model_training_desc;

/* model and training */
model_context *model_create(mem_arena *arena, const model_desc *desc);
void model_feedforward(model_context *model);
void model_train(model_context *model,
                 const model_training_desc *training_desc);

void draw_mnist_digit(f32 *data);
model_context *create_mnist_model(mem_arena *arena, model_graph *graph);

int main(void) {
  /* seed the rng from the clock so each run starts from different weights
   * the seed is printed for reference, but there's no way to pass it back
   * in yet, so a run can't be reproduced
   */
  u64 seed = 42; // (u64)time(NULL);
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
  matrix *train_images = mat_load(perm_arena, "train_images.mat");
  if (train_images == NULL) {
    fprintf(stderr, "failed to load train_images.mat (run mnist.py?)\n");
    return 1;
  }
  matrix *test_images = mat_load(perm_arena, "test_images.mat");
  if (test_images == NULL) {
    fprintf(stderr, "failed to load test_images.mat (run mnist.py?)\n");
    return 1;
  }

  matrix *train_labels_file = mat_load(perm_arena, "train_labels.mat");
  if (train_labels_file == NULL) {
    fprintf(stderr, "failed to load train_labels.mat (run mnist.py?)\n");
    return 1;
  }
  matrix *test_labels_file = mat_load(perm_arena, "test_labels.mat");
  if (test_labels_file == NULL) {
    fprintf(stderr, "failed to load test_labels.mat (run mnist.py?)\n");
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
  model_train(model, &training_desc);

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

/* prints one 28 x 28 image to the terminal, 2 spaces per pixel so it comes
 * out roughly square, with the pixel's brightness as the background color
 * data points at the image's first pixel, so pass images->data + n * 784 to
 * draw sample n
 */
void draw_mnist_digit(f32 *data) {
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
 * if a shape is wrong, mv_matmul/mv_add return NULL and nothing here checks,
 * so a crash right after changing these sizes is most likely a mismatch
 */
model_context *create_mnist_model(mem_arena *arena, model_graph *graph) {
  model_var *input = mv_create(arena, graph, MNIST_IMG_SIZE, 1, 0);

  model_var *W0 = mv_create(arena, graph, MNIST_HIDDEN_SIZE, MNIST_IMG_SIZE,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *W1 = mv_create(arena, graph, MNIST_HIDDEN_SIZE, MNIST_HIDDEN_SIZE,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *W2 = mv_create(arena, graph, MNIST_NUM_CLASSES, MNIST_HIDDEN_SIZE,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

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

/* checks desc, then builds both programs
 * returns NULL (with the reason on stderr) if desc is invalid, so a non-NULL
 * model always has its nodes and programs, nothing after this re-checks
 */
model_context *model_create(mem_arena *arena, const model_desc *desc) {
  if (desc->graph == NULL || desc->input == NULL || desc->output == NULL ||
      desc->desired_output == NULL || desc->cost == NULL) {
    fprintf(stderr, "model_create: graph and all four nodes must be set\n");
    return NULL;
  }

  /* cross entropy compares element by element */
  matrix *out = desc->output->val;
  matrix *want = desc->desired_output->val;
  if (out->rows != want->rows || out->cols != want->cols) {
    fprintf(stderr,
            "model_create: output is %u x %u but desired output is %u x %u\n",
            out->rows, out->cols, want->rows, want->cols);
    return NULL;
  }

  /* backprop seeds cost->grad with 1s, and grad only exists with this flag */
  if ((desc->cost->flags & MV_FLAG_REQUIRES_GRAD) == 0) {
    fprintf(stderr, "model_create: cost doesn't depend on any parameter\n");
    return NULL;
  }

  model_context *model = PUSH_STRUCT(arena, model_context);

  model->graph = desc->graph;
  model->input = desc->input;
  model->output = desc->output;
  model->desired_output = desc->desired_output;
  model->cost = desc->cost;

  /* sorted once here so training doesn't redo the sort on every sample */
  model->forward_prog = model_prog_create(arena, desc->graph, desc->output);
  model->cost_prog = model_prog_create(arena, desc->graph, desc->cost);

  return model;
}

/* runs input -> output, copy an image into model->input->val first */
void model_feedforward(model_context *model) {
  model_prog_compute(&model->forward_prog);
}

/* trains with mini-batch stochastic gradient descent
 * each epoch: shuffle the training order, then for each batch run every
 * sample forward + backward so the parameter grads add up, then step every
 * parameter against its averaged grad
 * after each epoch, measures cost and accuracy on the test set
 */
void model_train(model_context *model,
                 const model_training_desc *training_desc) {
  matrix *train_images = training_desc->train_images;
  matrix *train_labels = training_desc->train_labels;
  matrix *test_images = training_desc->test_images;
  matrix *test_labels = training_desc->test_labels;

  u32 num_examples = train_images->rows;
  u32 input_size = train_images->cols;
  u32 output_size = train_labels->cols;
  u32 num_tests = test_images->rows;

  /* integer division, leftover samples that don't fill a batch are
   * skipped that epoch */
  u32 num_batches = num_examples / training_desc->batch_size;

  mem_arena_temp scratch = arena_scratch_get(NULL, 0);

  /* shuffle a list of sample indices instead of the data itself, moving
   * one u32 is cheaper than moving a 784-float row */
  u32 *training_order = PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);
  for (u32 i = 0; i < num_examples; i++) {
    training_order[i] = i;
  }

  for (u32 epoch = 0; epoch < training_desc->epochs; epoch++) {
    /* num_examples random swaps, so each epoch sees a new order */
    for (u32 i = 0; i < num_examples; i++) {
      u32 a = prng_rand() % num_examples;
      u32 b = prng_rand() % num_examples;

      u32 tmp = training_order[b];
      training_order[b] = training_order[a];
      training_order[a] = tmp;
    }

    for (u32 batch = 0; batch < num_batches; batch++) {
      /* parameter grads add up over the batch, so start them at 0 */
      for (u32 i = 0; i < model->cost_prog.size; i++) {
        model_var *cur = model->cost_prog.vars[i];

        if (cur->flags & MV_FLAG_PARAMETER) {
          mat_clear(cur->grad);
        }
      }

      f32 avg_cost = 0.0f;
      for (u32 i = 0; i < training_desc->batch_size; i++) {
        u32 order_index = batch * training_desc->batch_size + i;
        u32 index = training_order[order_index];

        memcpy(model->input->val->data, train_images->data + index * input_size,
               sizeof(f32) * input_size);

        memcpy(model->desired_output->val->data,
               train_labels->data + index * output_size,
               sizeof(f32) * output_size);

        /* forward, then backward, which adds onto parameter grads */
        model_prog_compute(&model->cost_prog);
        model_prog_compute_grads(&model->cost_prog);

        avg_cost += mat_sum(model->cost->val);
      }
      avg_cost /= (f32)training_desc->batch_size;

      /* gradient descent step: w -= (lr / batch_size) * grad
       * grad is a sum over the batch, dividing by batch_size averages
       * it, and subtracting moves w downhill on the cost */
      for (u32 i = 0; i < model->cost_prog.size; i++) {
        model_var *cur = model->cost_prog.vars[i];

        if ((cur->flags & MV_FLAG_PARAMETER) != MV_FLAG_PARAMETER) {
          continue;
        }

        mat_scale(cur->grad,
                  training_desc->learning_rate / training_desc->batch_size);
        mat_sub(cur->val, cur->val, cur->grad);
      }

      printf("Epoch %2d / %2d, Batch %4d / %4d, Avg Cost: %.4f\r", epoch + 1,
             training_desc->epochs, batch + 1, num_batches, avg_cost);
      fflush(stdout);
    }
    printf("\n");

    /* test pass: forward only, no grads or updates, on samples the
     * model never trained on */
    u32 num_correct = 0;
    f32 avg_cost = 0;
    for (u32 i = 0; i < num_tests; i++) {
      memcpy(model->input->val->data, test_images->data + i * input_size,
             sizeof(f32) * input_size);

      memcpy(model->desired_output->val->data,
             test_labels->data + i * output_size, sizeof(f32) * output_size);

      model_prog_compute(&model->cost_prog);

      avg_cost += mat_sum(model->cost->val);
      num_correct += mat_argmax(model->output->val) ==
                     mat_argmax(model->desired_output->val);
    }

    avg_cost /= (f32)num_tests;
    printf("Test Completed. Accuracy: %5d / %5d (%.1f%%), Avg Cost: %.4f\n",
           num_correct, num_tests, (f32)num_correct / num_tests * 100.0f,
           avg_cost);
  }

  arena_scratch_release(scratch);
}
