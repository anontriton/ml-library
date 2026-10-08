#include "arena.h"
#include "prng.h"

#include "arena.c"
#include "prng.c"

#include <time.h>

#define MNIST_IMG_SIDE 28 /* each digit is a 28 x 28 pixel grayscale image */
#define MNIST_IMG_SIZE (MNIST_IMG_SIDE * MNIST_IMG_SIDE) /* 784 */
#define MNIST_NUM_CLASSES 10 /* digits 0-9, the width of a one-hot label row */
/* width of each hidden layer, a model choice, not a fact about mnist */
#define MNIST_HIDDEN_SIZE 16

/* math layer */

/* smallest q cross entropy will use, so log(q) and p / q stay finite when a
 * softmax output underflows to 0, -log(1e-7) is only ~16
 */
#define CROSS_ENTROPY_EPS 1e-7f

/* a 2D grid of floats
 * element (r, c) lives at data[c + r * cols]
 * the struct only holds the shape and a ptr, the floats live in an arena
 */
typedef struct {
  u32 rows, cols;
  /* row-major */
  f32 *data;
} matrix;

/* the 16-byte header at the start of every .mat file (written by save_mat in
 * mnist.py), followed by rows * cols little-endian f32s
 * fields line up byte for byte with struct.pack("<4sIII"), so one fread
 * fills the whole struct
 */
typedef struct {
  char magic[4];
  u32 version;
  u32 rows;
  u32 cols;
} mat_file_header;

/* matrix functions */
matrix *mat_load(mem_arena *arena, const char *path);
matrix *mat_create(mem_arena *arena, u32 rows, u32 cols);
b32 mat_copy(matrix *dst, matrix *src);
void mat_clear(matrix *mat);
void mat_fill(matrix *mat, f32 x);
void mat_fill_rand(matrix* mat, f32 lower, f32 upper);
void mat_scale(matrix *mat, f32 scale);
f32 mat_sum(matrix *mat);
u64 mat_argmax(matrix* mat);
b32 mat_add(matrix *out, const matrix *a, const matrix *b);
b32 mat_sub(matrix *out, const matrix *a, const matrix *b);
b32 mat_mul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
            b8 transpose_a, b8 transpose_b);
b32 mat_relu(matrix *out, const matrix *in);
b32 mat_softmax(matrix *out, const matrix *in);
b32 mat_cross_entropy(matrix *out, const matrix *p, const matrix *q);
b32 mat_relu_add_grad(matrix *out, const matrix *in, const matrix* grad);
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out, 
                         const matrix* grad);
b32 mat_cross_entropy_add_grad(matrix* p_grad, matrix* q_grad, 
                               const matrix* p, const matrix* q,
                               const matrix* grad) ;

/* model layer
 * the model is a computational graph: every matrix in the network (input,
 * weights, intermediate results, cost) is a node (model_var), and each node
 * remembers which op made it and from which input nodes
 */

/* bit flags describing a model_var, combined with | and tested with &
 * e.g. a weight is MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER
 * each flag is its own bit so any combination fits in one u32
 * REQUIRES_GRAD:  gets a grad matrix, backprop computes d(cost)/d(this)
 * PARAMETER:      a weight or bias, what training actually updates
 * INPUT:          where an image gets copied in
 * OUTPUT:         the network's prediction
 * DESIRED_OUTPUT: where the one-hot label gets copied in
 * COST:           the loss, the end of the graph that backprop starts from
 */
typedef enum {
  MV_FLAG_NONE = 0,

  MV_FLAG_REQUIRES_GRAD  = (1 << 0),
  MV_FLAG_PARAMETER      = (1 << 1),
  MV_FLAG_INPUT          = (1 << 2),
  MV_FLAG_OUTPUT         = (1 << 3),
  MV_FLAG_DESIRED_OUTPUT = (1 << 4),
  MV_FLAG_COST           = (1 << 5),
} model_var_flags;

/* which operation produced a model_var
 * CREATE: made directly by mv_create (input, weights), no inputs
 * ops are grouped by how many inputs they take, and the _START markers split
 * the groups so MV_NUM_INPUTS gets the count with two comparisons
 * the markers aren't real ops, the leading _ signals that
 */
typedef enum {
  MV_OP_NULL = 0,
  MV_OP_CREATE,

  _MV_OP_UNARY_START,

  MV_OP_RELU,
  MV_OP_SOFTMAX,

  _MV_OP_BINARY_START,

  MV_OP_ADD,
  MV_OP_SUB,
  MV_OP_MATMUL,
  MV_OP_CROSS_ENTROPY,
} model_var_op;

/* binary ops take the most inputs, so that's the size of inputs[] */
#define MODEL_VAR_MAX_INPUTS 2
/* given an operation will return the num of inputs */
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))

/* one node in the graph */
typedef struct model_var {
  u32 index; /* unique id handed out by mv_create, used to index visited[] */
  u32 flags; /* MV_FLAG_* bits */

  matrix* val; /* the value, filled in by model_prog_compute */
  /* d(cost)/d(val), same shape as val, only allocated if REQUIRES_GRAD */
  matrix* grad;

  model_var_op op; /* how val gets computed from inputs */
  /* the nodes this one was computed from, unused slots stay NULL */
  struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

/* a flat list of nodes where every node comes after its inputs (a
 * topological order), built once by model_prog_create
 * walking it front to back = forward pass, back to front = backprop
 */
typedef struct {
  model_var** vars;
  u32 size;
} model_program;

/* the whole model: shortcuts to the special nodes (set by mv_create from
 * their flags) plus the two compiled programs
 * forward_prog: input -> output, used for inference
 * cost_prog: input + desired_output -> cost, used for training, it contains
 * the forward pass too since cost depends on output
 */
typedef struct {
  u32 num_vars; /* how many nodes exist, also the next node's index */

  model_var* input;
  model_var* output;
  model_var* desired_output;
  model_var* cost;

  model_program forward_prog;
  model_program cost_prog;
} model_context;

/* everything model_train needs, bundled so the call isn't 7 args long */
typedef struct {
  matrix* train_images;
  matrix* train_labels;
  matrix* test_images;
  matrix* test_labels;

  u32 epochs;
  u32 batch_size; /* samples averaged together per weight update */
  f32 learning_rate; /* how big a step each update takes */
} model_training_desc;

/* graph building, these only add nodes, no math happens yet */
model_var* mv_create(mem_arena* arena, model_context* model,
                     u32 rows, u32 cols, u32 flags);

model_var* mv_relu(mem_arena* arena, model_context* model, 
                   model_var* input, u32 flags);
model_var* mv_softmax(mem_arena* arena, model_context* model, 
                      model_var* input, u32 flags);

model_var* mv_add(mem_arena* arena, model_context* model, 
                  model_var* a, model_var* b, u32 flags);
model_var* mv_sub(mem_arena* arena, model_context* model, 
                  model_var* a, model_var* b, u32 flags);
model_var* mv_matmul(mem_arena* arena, model_context* model,
                     model_var* a, model_var* b, u32 flags);
model_var* mv_cross_entropy(mem_arena* arena, model_context* model, 
                            model_var* p, model_var* q, u32 flags);

/* programs, the sorted node lists that actually run the math */
model_program model_prog_create(mem_arena* arena, model_context* model,
                                model_var* out_var);
void model_prog_compute(model_program* prog);
void model_prog_compute_grads(model_program* prog);

/* model, the high-level api main uses */
model_context* model_create(mem_arena* arena);
void model_compile(mem_arena* arena, model_context* model);
void model_feedforward(model_context* model); /* run function */
void model_train(model_context* model,
                 const model_training_desc* training_desc);

void draw_mnist_digit(f32* data);
void create_mnist_model(mem_arena* arena, model_context* model);

int main(void) {
  /* seed the rng from the clock so each run starts from different weights
   * printed so a weird run can be reproduced by passing the same seed back
   * in, cast since u64's printf specifier differs between platforms
   */
  u64 seed = (u64)time(NULL);
  prng_seed(seed, 1);
  printf("seed: %llu\n", (unsigned long long)seed);

  /* create big perm arena so we don't have to worry about mem space */
  mem_arena *perm_arena = arena_create(GiB(1), MiB(1));
  if (perm_arena == NULL) {
    fprintf(stderr, "failed to create arena\n");
    return 1;
  }

  /* mat_load returns NULL if a file is missing or has a bad header
   * early returns skip arena_destroy, the OS frees it when the process exits
   */
  matrix* train_images = mat_load(perm_arena, "train_images.mat");
  if (train_images == NULL) {
    fprintf(stderr, "failed to load train_images.mat (run mnist.py?)\n");
    return 1;
  }
  matrix* test_images = mat_load(perm_arena, "test_images.mat");
  if (test_images == NULL) {
    fprintf(stderr, "failed to load test_images.mat (run mnist.py?)\n");
    return 1;
  }

  matrix* train_labels_file = mat_load(perm_arena, "train_labels.mat");
  if (train_labels_file == NULL) {
    fprintf(stderr, "failed to load train_labels.mat (run mnist.py?)\n");
    return 1;
  }
  matrix* test_labels_file = mat_load(perm_arena, "test_labels.mat");
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
    fprintf(stderr, "test set has %u images but %u labels\n",
            test_images->rows, test_labels_file->rows);
    return 1;
  }

  matrix* train_labels = mat_create(perm_arena, train_labels_file->rows,
                                    MNIST_NUM_CLASSES);
  matrix* test_labels = mat_create(perm_arena, test_labels_file->rows, 
                                   MNIST_NUM_CLASSES);

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

  /* build the graph, then sort it into programs, compile has to come after
   * create_mnist_model since the sort needs the finished graph */
  model_context* model = model_create(perm_arena);
  create_mnist_model(perm_arena, model);
  model_compile(perm_arena, model);

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

  /* designated initializer (c99), sets fields by name so the order doesn't
   * matter, any field left out is zeroed */
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
void draw_mnist_digit(f32* data) {
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

/* builds the network as a graph:
 *   input (784) -> layer 0 (16) -> layer 1 (16) -> layer 2 (10) -> softmax
 * each layer is W * x + b, the hidden ones followed by a relu
 * shapes: W is (outputs x inputs) so W * x turns an (inputs x 1) column into
 * an (outputs x 1) column, b is (outputs x 1) to match
 * if a shape is wrong, mv_matmul/mv_add return NULL and nothing here checks,
 * so a crash right after changing these sizes is most likely a mismatch
 */
void create_mnist_model(mem_arena* arena, model_context* model) {
  model_var* input = mv_create(arena, model, MNIST_IMG_SIZE, 1, MV_FLAG_INPUT);

  model_var* W0 = mv_create(arena, model, MNIST_HIDDEN_SIZE, MNIST_IMG_SIZE,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var* W1 = mv_create(arena, model, MNIST_HIDDEN_SIZE, MNIST_HIDDEN_SIZE,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var* W2 = mv_create(arena, model, MNIST_NUM_CLASSES, MNIST_HIDDEN_SIZE,
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

  model_var* b0 = mv_create(arena, model, MNIST_HIDDEN_SIZE, 1,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var* b1 = mv_create(arena, model, MNIST_HIDDEN_SIZE, 1,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var* b2 = mv_create(arena, model, MNIST_NUM_CLASSES, 1,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

  /* biases start at 0 (PUSH_ARRAY zeroes), only the weights need to be
   * random to break the symmetry */

  /* layer 0: a0 = relu(W0 * input + b0) */
  model_var* z0_a = mv_matmul(arena, model, W0, input, 0);
  model_var* z0_b = mv_add(arena, model, z0_a, b0, 0);
  model_var* a0 = mv_relu(arena, model, z0_b, 0);

  /* layer 1: a1 = a0 + relu(W1 * a0 + b1)
   * adding a0 back in is a residual (skip) connection, the layer only has to
   * learn a change to a0, and grads get a shortcut straight back through
   * the add, needs W1 square so both sides of the add have the same shape */
  model_var* z1_a = mv_matmul(arena, model, W1, a0, 0);
  model_var* z1_b = mv_add(arena, model, z1_a, b1, 0);
  model_var* z1_c = mv_relu(arena, model, z1_b, 0);
  model_var* a1 = mv_add(arena, model, a0, z1_c, 0);

  /* layer 2: output = softmax(W2 * a1 + b2), 10 probabilities, no relu
   * since softmax already turns the scores into the final answer */
  model_var* z2_a = mv_matmul(arena, model, W2, a1, 0);
  model_var* z2_b = mv_add(arena, model, z2_a, b2, 0);
  model_var* output = mv_softmax(arena, model, z2_b, MV_FLAG_OUTPUT);

  /* training only: y holds the one-hot label, cost compares it to output
   * neither is in forward_prog since output doesn't depend on them */
  model_var* y = mv_create(arena, model, MNIST_NUM_CLASSES, 1,
                           MV_FLAG_DESIRED_OUTPUT);

  model_var* cost = mv_cross_entropy(arena, model, y, output, MV_FLAG_COST);
  (void)cost; /* stored as model->cost by its flag, named for readability */
}

/* makes a new matrix and returns a ptr to it
 * matrix lives in arena the caller passes in
 * no free, matrix goes away when its arena is cleared
 * nums are stored as one flat array
 */
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
  matrix* mat = PUSH_STRUCT(arena, matrix);

  mat->rows = rows;
  mat->cols = cols;
  mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);

  return mat;
}

/* reads a .mat file (header + floats) into a new matrix in arena
 * the shape comes from the header, so the caller doesn't pass one
 * returns NULL if the file is missing, the header is wrong, or the file has
 * fewer floats than the header says
 */
matrix *mat_load(mem_arena *arena, const char *path) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) { return NULL; }

  /* read the 16-byte header into the struct */
  mat_file_header header;
  if (fread(&header, sizeof(header), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  /* verify */
  if (memcmp(header.magic, "MAT1", 4) != 0 || header.version != 1) {
    fclose(f);
    return NULL;
  }

  /* allocate using the shape from the file, then read the data into it */
  matrix *mat = mat_create(arena, header.rows, header.cols);
  u64 count = (u64)header.rows * header.cols;
  if (fread(mat->data, sizeof(f32), count, f) != count) {
    fclose(f);
    return NULL;
  }

  fclose(f);
  return mat;
}

/* copies all the nums from src into dst
 * only copies the nums, not the struct, dst must already exist
 * memcpy works on bytes, which is why it multiplies by sizeof(f32)
 * returns false if the shapes don't match
 */
b32 mat_copy(matrix *dst, matrix *src) {
  if (dst->rows != src->rows || dst->cols != src->cols) {
    return false;
  }

  memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);

  return true;
}

/* sets every element to zero, a single memset that writes zero bytes over
 * the whole data array
 * useful for resetting gradients before each training step
 */
void mat_clear(matrix *mat) {
  memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

/* sets every element to x
 * it computes the total element count (rows * cols as u64),
 * then loops over the flat array and assigns x to each slot
 * because the storage is flat, one loop covers the whole matrix
 * and shape doesn't matter
 */
void mat_fill(matrix *mat, f32 x) {
  u64 size = (u64)mat->rows * mat->cols;

  for (u64 i = 0; i < size; i++) {
    mat->data[i] = x;
  }
}

/* sets every element to a random value in [lower, upper]
 * prng_randf gives 0-1, * (upper - lower) stretches it to the range's width,
 * + lower slides it into place
 * ML use: initializing weights, if they all started equal every neuron in a
 * layer would get the same grad and learn the same thing
 */
void mat_fill_rand(matrix* mat, f32 lower, f32 upper) {
  u64 size = (u64)mat->rows * mat->cols;

  for (u64 i = 0; i < size; i++) {
    mat->data[i] = prng_randf() * (upper - lower) + lower;
  }

}

/* multiplies every element by scale, in place. original values get overwritten
 * same flat loop as mat_fill, with data[i] *= scale
 * ML use: the learning-rate step (gradient * learning_rate),
 * or averaging (multiplying by 1/N)
 */
void mat_scale(matrix *mat, f32 scale) {
  u64 size = (u64)mat->rows * mat->cols;

  for (u64 i = 0; i < size; i++) {
    mat->data[i] *= scale;
  }
}

/* adds up every element and returns the total as one f32
 * starts sum at 0.0f, loops over the flat array adding each value
 * ML use: turning a matrix of per-sample losses into one total loss number
 */
f32 mat_sum(matrix *mat) {
  u64 size = (u64)mat->rows * mat->cols;

  f32 sum = 0.0f;
  for (u64 i = 0; i < size; i++) {
    sum += mat->data[i];
  }

  return sum;
}

/* single pass over that flat array, similar to mat_sum
 * keep track of best index seen so far
 * start at zero, update when pass over larger value
 * ML use: the model's guess is the argmax of its output probabilities
 */
u64 mat_argmax(matrix* mat) {
  u64 size = (u64)mat->rows * mat->cols;

  u64 max_i = 0;
  for (u64 i = 0; i < size; i++) {
    if (mat->data[i] > mat->data[max_i]) {
      max_i = i;
    }
  }

  return max_i;
}

/* add element by element, out = a + b
 * each slot of out get sthe sum of the matching slots in a and b
 * the caller provies out (already created) and the function only fills it in
 * because storage is flat and checks for matching shapes, idx i points to the
 * same row, col in all three matrices, so one loop is enough
 * ML use: adding a bias, and the update step weights = weights - lr * grad
 */
b32 mat_add(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    return false;
  }
  if (out->rows != a->rows || out->cols != a->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = a->data[i] + b->data[i];
  }

  return true;
}

/* same as mat_add but loop does a - b instead */
b32 mat_sub(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    return false;
  }
  if (out->rows != a->rows || out->cols != a->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = a->data[i] - b->data[i];
  }

  return true;
}

/* out += op(a) * op(b), the 4 variants of the inner loops
 * suffix letters are for a then b: n = normal, t = transposed
 * a transposed copy is never built, the loop just reads element (r, c) from
 * the stored (c, r) slot, and the row stride is always the stored cols
 * k is the shared dimension that gets summed over
 * these += instead of =, mat_mul clears out first when zero_out is set
 * loop order is chosen so the innermost loop walks memory in order where it
 * can (nn is i, k, j), which keeps the cpu cache happy
 */
void _mat_mul_nn(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->cols; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
          a->data[k + i * a->cols] *
          b->data[j + k * b->cols];
      }
    }
  }
}

void _mat_mul_nt(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i =0; i < out->rows; i++) {
    for (u64 j = 0; j < out->cols; j++) {
      for (u64 k=0; k < a->cols; k++) {
        out->data[j + i * out->cols] +=
          a->data[k + i * a->cols] *
          b->data[k + j * b->cols];
      }
    }
  }
}

void _mat_mul_tn(matrix *out, const matrix *a, const matrix *b) {
  for (u64 k =0; k < a->rows; k++) {
    for (u64 i = 0; i < out->rows; i++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
          a->data[i + k * a->cols] *
          b->data[j + k * b->cols];
      }
    }
  }
}

void _mat_mul_tt(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 j = 0; j < out->cols; j++) {
      for (u64 k = 0; k < a->rows; k++) {
        out->data[j + i * out->cols] +=
          a->data[i + k * a->cols] *
          b->data[k + j * b->cols];
      }
    }
  }
}

/* out = op(a) * op(b), op being an optional transpose of each input
 * a_rows, a_cols, etc. are the shapes after transposing, for the checks
 * zero_out = 0 adds into out instead of overwriting it, backprop uses this
 * to accumulate gradients
 * the two flags get packed into a 2-bit number to pick the variant
 * ML use: every layer is W * x, backprop needs grad * x^T and W^T * grad
 */
b32 mat_mul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
            b8 transpose_a, b8 transpose_b) {
  u32 a_rows = transpose_a ? a->cols : a->rows;
  u32 a_cols = transpose_a ? a->rows : a->cols;
  u32 b_rows = transpose_b ? b->cols : b->rows;
  u32 b_cols = transpose_b ? b->rows : b->cols;

  if (a_cols != b_rows) { return false; }
  if (out->rows != a_rows || out->cols != b_cols) { return false; }

  if (zero_out) {
    mat_clear(out);
  }

  u32 transpose = (transpose_a << 1) | transpose_b;
  switch (transpose) {
    case 0b00: { _mat_mul_nn(out, a, b); } break;
    case 0b01: { _mat_mul_nt(out, a, b); } break;
    case 0b10: { _mat_mul_tn(out, a, b); } break;
    case 0b11: { _mat_mul_tt(out, a, b); } break;
  }

  return true;
}

/* relu(x) = max(0, x), element by element
 * ML use: the activation between layers, without a nonlinearity like this
 * stacked layers would collapse into one big linear function
 */
b32 mat_relu(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = MAX(0, in->data[i]);
  }

  return true;
}

/* turns raw scores into probabilities: all positive, summing to 1
 * treats the whole matrix as one distribution, so pass a single vector
 * subtract the max before expf to avoid overflow
 * ML use: the last layer, so output[i] reads as "chance the digit is i"
 */
b32 mat_softmax(matrix *out, const matrix *in) {
  /* o_i = e^a_i / sum(e^a_i) */
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;

  f32 max = in->data[0];
  for (u64 i = 1; i < size; i++) {
    max = MAX(max, in->data[i]);
  }

  f32 sum = 0.0f;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = expf(in->data[i] - max);
    sum += out->data[i];
  }

  mat_scale(out, 1.0f / sum);

  return true;
}

/* loss function
 * p is expected probability distribution
 * q is actual output
 */
b32 mat_cross_entropy(matrix *out, const matrix *p, const matrix *q) {
  if (p->rows != q->rows || p->cols != q->cols) { return false; }
  if (out->rows != p->rows || out->cols != p->cols) { return false; }

  /* p * -log(q)
   * q is clamped to CROSS_ENTROPY_EPS, -logf(0) is inf, so a wrong prediction
   * could make the whole avg cost inf
   */
  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    f32 q_safe = MAX(q->data[i], CROSS_ENTROPY_EPS);
    out->data[i] = p->data[i] == 0.0f ?
      0.0f : p->data[i] * -logf(q_safe);
  }

  return true;
}

/* backprop through relu: out += grad where the input was > 0, else nothing
 * relu's slope is 1 where it let the value through and 0 where it clamped
 * it, so by the chain rule the incoming grad either passes or gets blocked
 * in is the relu's input (not its output), out is that input's grad
 */
b32 mat_relu_add_grad(matrix *out, const matrix *in, const matrix* grad) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }
  if (out->rows != grad->rows || out->cols != grad->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
  }

  return true;
}

/* backprop through softmax: out += J * grad
 * every output depends on every input (they share the sum), so the slope is
 * a full size x size matrix, the jacobian J[i][j] = s_i * (delta_ij - s_j)
 * (i == j) is the delta, 1 on the diagonal and 0 elsewhere
 * J is symmetric, so J^T * grad (what the chain rule wants) is just J * grad
 * only works on vectors, hence the rows/cols == 1 check
 * J is temporary, so it goes in a scratch arena and is freed on release
 */
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out,
                         const matrix* grad) {
  if (softmax_out->rows != 1 && softmax_out->cols != 1) {
    return false;
  }

  mem_arena_temp scratch = arena_scratch_get(NULL, 0);

  u32 size = MAX(softmax_out->rows, softmax_out->cols);
  matrix* jacobian = mat_create(scratch.arena, size, size);

  for (u32 i = 0; i < size; i++) {
    for (u32 j = 0; j < size; j++) {
      jacobian->data[j + i * size] =
       softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
    }
  }

  mat_mul(out, jacobian, grad, 0, 0, 0);

  arena_scratch_release(scratch);

  return true;
}

/* backprop through cross entropy L = p * -log(q), element by element
 * dL/dp = -log(q) and dL/dq = -p / q, each times the incoming grad
 * either grad can be NULL to skip it, in training p is the label, which
 * never needs a grad
 */
b32 mat_cross_entropy_add_grad(matrix* p_grad, matrix* q_grad,
                               const matrix* p, const matrix* q,
                               const matrix* grad
) {
  if (p->rows != q->rows || p->cols != q->cols) { return false; }

  u64 size = (u64)p->rows * p->cols;

  if (p_grad != NULL) {
    if (p_grad->rows != p->rows || p_grad->cols != p->cols) {
      return false;
    }

    for (u64 i = 0; i < size; i++) {
      f32 q_safe = MAX(q->data[i], CROSS_ENTROPY_EPS);
      p_grad->data[i] += -logf(q_safe) * grad->data[i];
    }
  }

  if (q_grad != NULL) {
    if (q_grad->rows != q->rows || q_grad->cols != q->cols) {
      return false;
    }

    /* same clamp as the forward pass, -p / 0 would be -inf, which turns
     * into NaN in the weights on the next update
     * when clamped, the grad is still big and points the right way (raise
     * q), it's just finite */
    for (u64 i = 0; i < size; i++) {
      f32 q_safe = MAX(q->data[i], CROSS_ENTROPY_EPS);
      q_grad->data[i] += -p->data[i] / q_safe * grad->data[i];
    }
  }

  return true;
}

/* adds a new node to the graph and returns it
 * index = num_vars++ hands out ids 0, 1, 2, ... in creation order
 * grad is only allocated when something needs it (saves memory on inputs)
 * the special flags also register the node on the model, so training knows
 * where to copy images/labels in and where to read output/cost from
 */
model_var* mv_create(mem_arena* arena, model_context* model,
                     u32 rows, u32 cols, u32 flags) {
  model_var* out = PUSH_STRUCT(arena, model_var);

  out->index = model->num_vars++;
  out->flags = flags;
  out->op = MV_OP_CREATE;
  out->val = mat_create(arena, rows, cols);

  if (flags & MV_FLAG_REQUIRES_GRAD) {
    out->grad = mat_create(arena, rows, cols);
  }

  if (flags & MV_FLAG_INPUT) { model->input = out; }
  if (flags & MV_FLAG_OUTPUT) { model->output = out; }
  if (flags & MV_FLAG_DESIRED_OUTPUT) { model->desired_output = out; }
  if (flags & MV_FLAG_COST) { model->cost = out; }

  return out;
}

/* shared body for one-input ops: make the node, record its op and input
 * if the input needs a grad, so does the output, otherwise backprop would
 * have no path from the cost back to the input
 */
model_var* _mv_unary_impl(mem_arena* arena, model_context* model,
                          model_var* input, u32 rows, u32 cols, 
                          u32 flags, model_var_op op) {
  if (input->flags & MV_FLAG_REQUIRES_GRAD) {
    flags |= MV_FLAG_REQUIRES_GRAD;
  }

  model_var* out = mv_create(arena, model, rows, cols, flags);

  out->op = op;
  out->inputs[0] = input;

  return out;
}

/* same as _mv_unary_impl for two-input ops, the output needs a grad if
 * either input does
 */
model_var* _mv_binary_impl(mem_arena* arena, model_context* model,
                           model_var* a, model_var* b, u32 rows, u32 cols, 
                           u32 flags, model_var_op op) {
  if (
    (a->flags & MV_FLAG_REQUIRES_GRAD) ||
    (b->flags & MV_FLAG_REQUIRES_GRAD)
  ) {
    flags |= MV_FLAG_REQUIRES_GRAD;
  }

  model_var* out = mv_create(arena, model, rows, cols, flags);

  out->op = op;
  out->inputs[0] = a;
  out->inputs[1] = b;

  return out;
}

/* the mv_* op functions build the graph, they don't compute anything
 * each one checks shapes, works out the output shape, and adds a node
 * (NULL on a shape mismatch), the math runs later in model_prog_compute
 */
model_var* mv_relu(mem_arena* arena, model_context* model,
                   model_var* input, u32 flags) {
  return _mv_unary_impl(
    arena, model, input,
    input->val->rows, input->val->cols,
    flags, MV_OP_RELU
  );
}

model_var* mv_softmax(mem_arena* arena, model_context* model, 
                      model_var* input, u32 flags) {
  return _mv_unary_impl(
    arena, model, input,
    input->val->rows, input->val->cols,
    flags, MV_OP_SOFTMAX
  );
}

model_var* mv_add(mem_arena* arena, model_context* model, 
                  model_var* a, model_var* b, u32 flags) {
  if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
    return NULL;
  }

  return _mv_binary_impl(
    arena, model, a, b,
    a->val->rows, a->val->cols,
    flags, MV_OP_ADD
  );
}

model_var* mv_sub(mem_arena* arena, model_context* model, 
                  model_var* a, model_var* b, u32 flags) {
  if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
    return NULL;
  }

  return _mv_binary_impl(
    arena, model, a, b,
    a->val->rows, a->val->cols,
    flags, MV_OP_SUB
  );
}

model_var* mv_matmul(mem_arena* arena, model_context* model, 
                     model_var* a, model_var* b, u32 flags) {
  /* (m x n) * (n x p) -> (m x p), inner dims must match */
  if (a->val->cols != b->val->rows) {
    return NULL;
  }

  return _mv_binary_impl(
    arena, model, a, b,
    a->val->rows, b->val->cols,
    flags, MV_OP_MATMUL
  );
}

/* p is the desired output (label), q is the prediction, same as in
 * mat_cross_entropy
 */
model_var* mv_cross_entropy(mem_arena* arena, model_context* model,
                            model_var* p, model_var* q, u32 flags) {
  if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
    return NULL;
  }

  return _mv_binary_impl(
    arena, model, p, q,
    p->val->rows, p->val->cols,
    flags, MV_OP_CROSS_ENTROPY
  );
}

/* topological sort with DFS
 * turns the graph into a model_program: every node out_var depends on,
 * sorted so each node comes after its inputs (a topological sort)
 * it's a depth-first search using a manual stack instead of recursion:
 * - 1st time a node is popped: mark it visited, push it back, then push its
 *   inputs on top of it, so they get handled first
 * - 2nd time it's popped: its inputs are all done, so append it to out
 * nodes out_var doesn't depend on are never reached, which is why the
 * forward program leaves out the label and cost nodes
 */
model_program model_prog_create(mem_arena* arena, model_context* model,
                                model_var* out_var) {
  /* &arena as a conflict guarantees scratch isn't the arena prog.vars gets
   * pushed to, otherwise releasing scratch would pop prog.vars too */
  mem_arena_temp scratch = arena_scratch_get(&arena, 1);

  /* indexed by model_var.index, PUSH_ARRAY zeroes it so all start false */
  b8* visited = PUSH_ARRAY(scratch.arena, b8, model->num_vars);

  /* no list can hold more than every node, so num_vars is a safe size */
  u32 stack_size = 0;
  u32 out_size = 0;
  model_var** stack = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);
  model_var** out = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);

  stack[stack_size++] = out_var;

  while (stack_size > 0) {
    model_var* cur = stack[--stack_size];

    /* guard, a bad index would read past the end of visited[] */
    if (cur->index >= model->num_vars) { continue; }

    /* 2nd pop, inputs are done */
    if (visited[cur->index]) {
      if (out_size < model->num_vars) {
        out[out_size++] = cur;
      }
      continue;
    }

    /* 1st pop, push cur back under its inputs */
    visited[cur->index] = true;

    if (stack_size < model->num_vars) {
      stack[stack_size++] = cur;
    }

    u32 num_inputs = MV_NUM_INPUTS(cur->op);
    for (u32 i = 0; i < num_inputs; i++) {
      model_var* input = cur->inputs[i];

      if (input->index >= model->num_vars || visited[input->index]) {
        continue;
      }

      /* if input is already deeper in the stack (another node pushed
       * it), pull it out and re-push it on top, otherwise cur would
       * pop for the 2nd time and be emitted before its own input */
      for (u32 j = 0; j < stack_size; j++) {
        if (stack[j] == input) {
          for (u32 k = j; k < stack_size-1; k++) {
            stack[k] = stack[k+1];
          }
          stack_size--;
        }
      }

      if (stack_size < model->num_vars) {
        stack[stack_size++] = input;
      }
    }
  }

  /* out is in scratch, copy it to the caller's arena so it outlives this
   * function, NZ since memcpy overwrites it right away */
  model_program prog = {
    .size = out_size,
    .vars = PUSH_ARRAY_NZ(arena, model_var*, out_size)
  };

  memcpy(prog.vars, out, sizeof(model_var*) * out_size);

  arena_scratch_release(scratch);

  return prog;
}

/* forward pass: run each node's op in order, writing into its val
 * the order from model_prog_create guarantees a and b are already computed
 * CREATE nodes have no op, their val was set from outside (an image copied
 * in, or the weights)
 * the _START markers are only listed so the switch covers every enum value
 * (clang's -Wswitch warns otherwise)
 */
void model_prog_compute(model_program* prog) {
  for (u32 i = 0; i < prog->size; i++) {
    model_var* cur = prog->vars[i];

    model_var* a = cur->inputs[0];
    model_var* b = cur->inputs[1];

    switch (cur->op) {
      case MV_OP_NULL:
      case MV_OP_CREATE: break;

      case _MV_OP_UNARY_START: break;

      case MV_OP_RELU: { mat_relu(cur->val, a->val); } break;
      case MV_OP_SOFTMAX: { mat_softmax(cur->val, a->val); } break;

      case _MV_OP_BINARY_START: break;

      case MV_OP_ADD: { mat_add(cur->val, a->val, b->val); } break;
      case MV_OP_SUB: { mat_sub(cur->val, a->val, b->val); } break;
      case MV_OP_MATMUL: {
        /* zero_out = 1, the forward pass overwrites val */
        mat_mul(cur->val, a->val, b->val, 1, 0, 0);
      } break;
      case MV_OP_CROSS_ENTROPY: {
        mat_cross_entropy(cur->val, a->val, b->val);
      } break;
    }
  }
}

/* backprop: fill in grad for every node that needs one
 * walks the program backwards (cost first), and each node pushes its grad
 * to its inputs with the chain rule:
 *   input.grad += (d cur / d input) * cur.grad
 * += because one node can feed several others, and its grad is the sum of
 * what flows back from all of them
 */
void model_prog_compute_grads(model_program* prog) {
  /* zero the grads left over from the last run, except parameters, their
   * grads add up across a whole batch and model_train clears them */
  for (u32 i = 0; i < prog->size; i++) {
    model_var* cur = prog->vars[i];

    if ((cur->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
      continue;
    }

    if (cur->flags & MV_FLAG_PARAMETER) {
      continue;
    }

    mat_clear(cur->grad);
  }

  /* seed: d(cost)/d(cost) = 1, the last node in the program is the cost
   * cost is a vector that gets summed (mat_sum), so a grad of all 1s is
   * the gradient of that sum */
  mat_fill(prog->vars[prog->size-1]->grad, 1.0f);

  for (i64 i = (i64)prog->size - 1; i >= 0; i--) {
    model_var* cur = prog->vars[i];

    if ((cur->flags & MV_FLAG_REQUIRES_GRAD) == 0) {
      continue;
    }

    model_var* a = cur->inputs[0];
    model_var* b = cur->inputs[1];

    u32 num_inputs = MV_NUM_INPUTS(cur->op);

    /* skip if no input needs a grad, nowhere to send it */
    if (
      num_inputs == 1 &&
      (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
    ) {
      continue;
    }

    if (
      num_inputs == 2 &&
      (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD && 
      (b->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
    ) {
      continue;
    }

    switch (cur->op) {
      case MV_OP_NULL:
      case MV_OP_CREATE: break;

      case _MV_OP_UNARY_START: break;

      case MV_OP_RELU: {
        mat_relu_add_grad(a->grad, a->val, cur->grad);
      } break;
      case MV_OP_SOFTMAX: {
        mat_softmax_add_grad(a->grad, cur->val, cur->grad);
      } break;

      case _MV_OP_BINARY_START: break;

      /* d(a + b)/da = d(a + b)/db = 1, grad passes through as is */
      case MV_OP_ADD: {
        if (a->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_add(a->grad, a->grad, cur->grad);
        }

        if (b->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_add(b->grad, b->grad, cur->grad);
        }
      } break;

      /* d(a - b)/da = 1, d(a - b)/db = -1, so b's grad is subtracted */
      case MV_OP_SUB: {
        if (a->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_add(a->grad, a->grad, cur->grad);
        }

        if (b->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_sub(b->grad, b->grad, cur->grad);
        }
      } break;

      /* C = A * B: dA += dC * B^T, dB += A^T * dC
       * the transpose flags do this without building copies, and
       * zero_out = 0 accumulates */
      case MV_OP_MATMUL: {
        if (a->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
        }

        if (b->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
        }
      } break;

      /* p = label (no REQUIRES_GRAD, so p->grad is NULL and skipped),
       * q = prediction */
      case MV_OP_CROSS_ENTROPY: {
        model_var* p = a;
        model_var* q = b;

        mat_cross_entropy_add_grad(
          p->grad, q->grad, p->val, q->val, cur->grad
        );
      } break;
    }
  }
}

/* an empty model, PUSH_STRUCT zeroes it, so num_vars = 0 and every ptr is
 * NULL until mv_create fills them in
 */
model_context* model_create(mem_arena* arena) {
  model_context* model = PUSH_STRUCT(arena, model_context);

  return model;
}

/* sorts the finished graph into programs once, so training doesn't redo
 * the sort on every sample
 */
void model_compile(mem_arena* arena, model_context* model) {
  if (model->output != NULL) {
    model->forward_prog = model_prog_create(arena, model, model->output);
  }

  if (model->cost != NULL) {
    model->cost_prog = model_prog_create(arena, model, model->cost);
  }
}

/* runs input -> output, copy an image into model->input->val first */
void model_feedforward(model_context* model) {
  model_prog_compute(&model->forward_prog);
}

/* trains with mini-batch stochastic gradient descent
 * each epoch: shuffle the training order, then for each batch run every
 * sample forward + backward so the parameter grads add up, then step every
 * parameter against its averaged grad
 * after each epoch, measures cost and accuracy on the test set
 */
void model_train(model_context* model,
                 const model_training_desc* training_desc) {
  matrix* train_images = training_desc->train_images;
  matrix* train_labels = training_desc->train_labels;
  matrix* test_images = training_desc->test_images;
  matrix* test_labels = training_desc->test_labels;

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
  u32* training_order = PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);
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
        model_var* cur = model->cost_prog.vars[i];

        if (cur->flags & MV_FLAG_PARAMETER) {
          mat_clear(cur->grad);
        }
      }

      f32 avg_cost = 0.0f;
      for (u32 i = 0; i < training_desc->batch_size; i++) {
        u32 order_index = batch * training_desc->batch_size + i;
        u32 index = training_order[order_index];

        /* load sample index: its image row into the input node,
         * its one-hot row into the desired_output node */
        memcpy(
          model->input->val->data,
          train_images->data + index * input_size,
          sizeof(f32) * input_size
        );

        memcpy(
          model->desired_output->val->data,
          train_labels->data + index * output_size,
          sizeof(f32) * output_size
        );

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
        model_var* cur = model->cost_prog.vars[i];

        if ((cur->flags & MV_FLAG_PARAMETER) != MV_FLAG_PARAMETER) {
          continue;
        }

        mat_scale(
          cur->grad,
          training_desc->learning_rate /
          training_desc->batch_size
        );
        mat_sub(cur->val, cur->val, cur->grad);
      }

      /* \r goes back to the start of the line without a newline, so
       * the progress line overwrites itself, fflush shows it now */
      printf(
        "Epoch %2d / %2d, Batch %4d / %4d, Avg Cost: %.4f\r",
        epoch + 1, training_desc->epochs,
        batch + 1, num_batches, avg_cost
      );
      fflush(stdout);
    }
    printf("\n");

    /* test pass: forward only, no grads or updates, on samples the
     * model never trained on */
    u32 num_correct = 0;
    f32 avg_cost = 0;
    for (u32 i = 0; i < num_tests; i++) {
      memcpy(
        model->input->val->data,
        test_images->data + i * input_size,
        sizeof(f32) * input_size
      );

      memcpy(
        model->desired_output->val->data,
        test_labels->data + i * output_size,
        sizeof(f32) * output_size
      );

      model_prog_compute(&model->cost_prog);

      avg_cost += mat_sum(model->cost->val);
      num_correct +=
        mat_argmax(model->output->val) == 
        mat_argmax(model->desired_output->val);
    }

    avg_cost /= (f32)num_tests;
    printf(
      "Test Completed. Accuracy: %5d / %5d (%.1f%%), Avg Cost: %.4f\n",
      num_correct, num_tests, (f32)num_correct / num_tests * 100.0f,
      avg_cost
    );
  }

  arena_scratch_release(scratch);
}
