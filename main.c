#include "arena.h"
#include "prng.h"

#include "arena.c"
#include "prng.c"

#define MNIST_IMG_SIDE 28 /* each digit is a 28 x 28 pixel grayscale image */
#define MNIST_IMG_SIZE (MNIST_IMG_SIDE * MNIST_IMG_SIDE) /* 784, one row per image */
#define MNIST_NUM_CLASSES 10 /* digits 0-9, the width of a one-hot label row */

/* math layer */
typedef struct {
  u32 rows, cols;
  /* row-major */
  f32 *data;
} matrix;

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
void mat_scale(matrix *mat, f32 scale);
f32 mat_sum(matrix *mat);
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

void draw_mnist_digit(f32* data);

int main(void) {
  /* create permanent arena */
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

  arena_destroy(perm_arena);

  return 0;
}

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

/* makes a new matrix and returns a ptr to it
 * matrix lives in arena the caller passes in
 * no free, matrix goes away when its arena is cleared
 * nums are stored as one flat array
 */
matrix *mat_create(mem_arena *arena, u32 rows, u32 cols) {
  matrix *mat = PUSH_STRUCT(arena, matrix);

  mat->rows = rows;
  mat->cols = cols;
  mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);

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

b32 mat_softmax(matrix *out, const matrix *in) {
  /* o_i = e^a_i / sum(e^a_i) */
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;
  
  f32 sum = 0.0f;
  for (u64 i = 0; i <size; i++) {
    out->data[i] = expf(in->data[i]);
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

  /* p * -log(q) */
  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = p->data[i] == 0.0f ? 
      0.0f : p->data[i] * -logf(q->data[i]);
  }

  return true;
}

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
      p_grad->data[i] += -logf(q->data[i]) * grad->data[i];
    }
  }

  if (q_grad != NULL) {
    if (q_grad->rows != q->rows || q_grad->cols != q->cols) {
      return false;
    }

    for (u64 i = 0; i < size; i++) {
      q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
    }
  }

  return true;
}
