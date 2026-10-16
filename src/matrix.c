/* math layer: the matrix ops everything else is built from
 * conventions:
 * - outputs come first in the params, and the caller allocates them
 * - b32 functions return false on a shape mismatch
 * - *_add_grad functions are the backprop versions, they += into out so
 *   grads from several uses of the same node add up
 */

#include "matrix.h"

#include "arena.h"
#include "base.h"
#include "prng.h"

/* smallest q cross entropy will use, so log(q) and p / q stay finite when a
 * softmax output underflows to 0, -log(1e-7) is only ~16
 */
#define CROSS_ENTROPY_EPS 1e-7f

/* the 16-byte header of a .mat file (written by save_mat in mnist.py),
 * followed by rows * cols little-endian f32s
 * lines up byte for byte with struct.pack("<4sIII"), so one fread fills it
 */
typedef struct {
  char magic[4];
  u32 version;
  u32 rows;
  u32 cols;
} mat_file_header;

matrix *mat_create(mem_arena *arena, u32 rows, u32 cols) {
  /* rows * cols fits in a u64, but 4 bytes per f32 can wrap it, and a
   * wrapped size would push a tiny buffer for a huge matrix
   */
  u64 count = (u64)rows * cols;
  if (count > UINT64_MAX / sizeof(f32)) {
    return NULL;
  }

  /* if data doesn't fit, pop the struct too so a failed create leaves the
   * arena how it was
   */
  mem_arena_temp undo = arena_temp_begin(arena);

  matrix *mat = PUSH_STRUCT(arena, matrix);
  if (mat == NULL) {
    return NULL;
  }

  mat->rows = rows;
  mat->cols = cols;
  mat->data = PUSH_ARRAY(arena, f32, count);
  if (mat->data == NULL) {
    arena_temp_end(undo);
    return NULL;
  }

  return mat;
}

/* reads a .mat file (header + floats) into a new matrix in arena
 * the shape comes from the header, so the caller doesn't pass one
 * returns NULL if the file is missing, the header is wrong, or the file has
 * fewer floats than the header says
 */
matrix *mat_load(mem_arena *arena, const char *path) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    return NULL;
  }

  mat_file_header header;
  if (fread(&header, sizeof(header), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  if (memcmp(header.magic, "MAT1", 4) != 0 || header.version != 1) {
    fclose(f);
    return NULL;
  }

  matrix *mat = mat_create(arena, header.rows, header.cols);
  if (mat == NULL) {
    fclose(f);
    return NULL;
  }
  u64 count = (u64)header.rows * header.cols;
  if (fread(mat->data, sizeof(f32), count, f) != count) {
    fclose(f);
    return NULL;
  }

  fclose(f);
  return mat;
}

b32 mat_copy(matrix *dst, matrix *src) {
  if (dst->rows != src->rows || dst->cols != src->cols) {
    return false;
  }

  memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);

  return true;
}

void mat_clear(matrix *mat) {
  memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

void mat_fill(matrix *mat, f32 x) {
  u64 size = (u64)mat->rows * mat->cols;

  for (u64 i = 0; i < size; i++) {
    mat->data[i] = x;
  }
}

/* sets every element to a uniform random value in [lower, upper]
 * ML use: initializing weights, if they all started equal every neuron in a
 * layer would get the same grad and learn the same thing
 */
void mat_fill_rand(matrix *mat, f32 lower, f32 upper) {
  u64 size = (u64)mat->rows * mat->cols;

  for (u64 i = 0; i < size; i++) {
    mat->data[i] = prng_randf() * (upper - lower) + lower;
  }
}

void mat_scale(matrix *mat, f32 scale) {
  u64 size = (u64)mat->rows * mat->cols;

  for (u64 i = 0; i < size; i++) {
    mat->data[i] *= scale;
  }
}

f32 mat_sum(matrix *mat) {
  u64 size = (u64)mat->rows * mat->cols;

  f32 sum = 0.0f;
  for (u64 i = 0; i < size; i++) {
    sum += mat->data[i];
  }

  return sum;
}

u64 mat_argmax(matrix *mat) {
  u64 size = (u64)mat->rows * mat->cols;

  u64 max_i = 0;
  for (u64 i = 0; i < size; i++) {
    if (mat->data[i] > mat->data[max_i]) {
      max_i = i;
    }
  }

  return max_i;
}

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

/* out += op(a) * op(b), the 4 variants of the inner loops, static because
 * only mat_mul calls them (it checks the shapes first)
 * suffix letters are for a then b: n = normal, t = transposed
 * a transposed copy is never built, the loop just reads element (r, c) from
 * the stored (c, r) slot, and the row stride is always the stored cols
 * k is the shared dimension that gets summed over
 * these += instead of =, mat_mul clears out first when zero_out is set
 * loop order is chosen so the innermost loop walks memory in order where it
 * can (nn is i, k, j), which keeps the cpu cache happy
 */
static void mat_mul_nn(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->cols; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[k + i * a->cols] * b->data[j + k * b->cols];
      }
    }
  }
}

static void mat_mul_nt(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 j = 0; j < out->cols; j++) {
      for (u64 k = 0; k < a->cols; k++) {
        out->data[j + i * out->cols] +=
            a->data[k + i * a->cols] * b->data[k + j * b->cols];
      }
    }
  }
}

static void mat_mul_tn(matrix *out, const matrix *a, const matrix *b) {
  for (u64 k = 0; k < a->rows; k++) {
    for (u64 i = 0; i < out->rows; i++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[i + k * a->cols] * b->data[j + k * b->cols];
      }
    }
  }
}

static void mat_mul_tt(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 j = 0; j < out->cols; j++) {
      for (u64 k = 0; k < a->rows; k++) {
        out->data[j + i * out->cols] +=
            a->data[i + k * a->cols] * b->data[k + j * b->cols];
      }
    }
  }
}

/* out = op(a) * op(b), op being an optional transpose of each input
 * zero_out = 0 adds into out instead of overwriting it, backprop uses this
 * to accumulate gradients
 * ML use: every layer is W * x, backprop needs grad * x^T and W^T * grad
 */
b32 mat_mul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
            b8 transpose_a, b8 transpose_b) {
  u32 a_rows = transpose_a ? a->cols : a->rows;
  u32 a_cols = transpose_a ? a->rows : a->cols;
  u32 b_rows = transpose_b ? b->cols : b->rows;
  u32 b_cols = transpose_b ? b->rows : b->cols;

  if (a_cols != b_rows) {
    return false;
  }
  if (out->rows != a_rows || out->cols != b_cols) {
    return false;
  }

  if (zero_out) {
    mat_clear(out);
  }

  u32 transpose = (transpose_a << 1) | transpose_b;
  switch (transpose) {
  case 0b00: {
    mat_mul_nn(out, a, b);
  } break;
  case 0b01: {
    mat_mul_nt(out, a, b);
  } break;
  case 0b10: {
    mat_mul_tn(out, a, b);
  } break;
  case 0b11: {
    mat_mul_tt(out, a, b);
  } break;
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

/* o_i = e^a_i / sum(e^a_j), raw scores -> probabilities that sum to 1
 * treats the whole matrix as one distribution, so pass a single vector
 * subtract the max before expf to avoid overflow
 */
b32 mat_softmax(matrix *out, const matrix *in) {
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

/* loss function, p is the expected distribution, q is the prediction */
b32 mat_cross_entropy(matrix *out, const matrix *p, const matrix *q) {
  if (p->rows != q->rows || p->cols != q->cols) {
    return false;
  }
  if (out->rows != p->rows || out->cols != p->cols) {
    return false;
  }

  /* p * -log(q)
   * q is clamped to CROSS_ENTROPY_EPS, -logf(0) is inf, so a wrong prediction
   * could make the whole avg cost inf
   */
  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    f32 q_safe = MAX(q->data[i], CROSS_ENTROPY_EPS);
    out->data[i] = p->data[i] == 0.0f ? 0.0f : p->data[i] * -logf(q_safe);
  }

  return true;
}

/* backprop through relu: out += grad where the input was > 0, else nothing
 * relu's slope is 1 where it let the value through and 0 where it clamped
 * it, so by the chain rule the incoming grad either passes or gets blocked
 * in is the relu's input (not its output), out is that input's grad
 */
b32 mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad) {
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
 * J never has to be built, row i of J * grad works out to
 *   sum_j s_i * (delta_ij - s_j) * g_j = s_i * g_i - s_i * sum_j s_j * g_j
 *                                      = s_i * (g_i - dot(s, g))
 * so it's one pass for the dot product and one for the update, O(n) instead
 * of an n x n matrix and an O(n^2) matmul
 * only works on column vectors, hence the cols == 1 check (see mv_softmax)
 */
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out,
                         const matrix *grad) {
  if (softmax_out->cols != 1) {
    return false;
  }
  if (out->rows != softmax_out->rows || out->cols != softmax_out->cols) {
    return false;
  }
  if (out->rows != grad->rows || out->cols != grad->cols) {
    return false;
  }

  u32 size = softmax_out->rows;

  f32 dot = 0.0f;
  for (u32 i = 0; i < size; i++) {
    dot += softmax_out->data[i] * grad->data[i];
  }

  for (u32 i = 0; i < size; i++) {
    out->data[i] += softmax_out->data[i] * (grad->data[i] - dot);
  }

  return true;
}

/* backprop through cross entropy L = p * -log(q), element by element
 * dL/dp = -log(q) and dL/dq = -p / q, each times the incoming grad
 * either grad can be NULL to skip it, in training p is the label, which
 * never needs a grad
 */
b32 mat_cross_entropy_add_grad(matrix *p_grad, matrix *q_grad, const matrix *p,
                               const matrix *q, const matrix *grad) {
  if (p->rows != q->rows || p->cols != q->cols) {
    return false;
  }
  if (grad->rows != p->rows || grad->cols != p->cols) {
    return false;
  }
  if (p_grad != NULL && (p_grad->rows != p->rows || p_grad->cols != p->cols)) {
    return false;
  }
  if (q_grad != NULL && (q_grad->rows != q->rows || q_grad->cols != q->cols)) {
    return false;
  }

  u64 size = (u64)p->rows * p->cols;

  if (p_grad != NULL) {
    for (u64 i = 0; i < size; i++) {
      f32 q_safe = MAX(q->data[i], CROSS_ENTROPY_EPS);
      p_grad->data[i] += -logf(q_safe) * grad->data[i];
    }
  }

  if (q_grad != NULL) {
    /* same clamp as the forward pass, -p / 0 would be -inf, which turns
     * into NaN in the weights on the next update
     * when clamped, the grad is still big and points the right way (raise
     * q), it's just finite
     */
    for (u64 i = 0; i < size; i++) {
      f32 q_safe = MAX(q->data[i], CROSS_ENTROPY_EPS);
      q_grad->data[i] += -p->data[i] / q_safe * grad->data[i];
    }
  }

  return true;
}
