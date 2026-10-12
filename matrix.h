#ifndef ARENA_H
#define ARENA_H

#include "arena.h"
#include "base.h"

/* row-major: element (r, c) lives at data[c + r * cols], in an arena */
typedef struct {
  u32 rows, cols;
  f32 *data;
} matrix;

matrix *mat_load(mem_arena *arena, const char *path);
matrix *mat_create(mem_arena *arena, u32 rows, u32 cols);
b32 mat_copy(matrix *dst, matrix *src);
void mat_clear(matrix *mat);
void mat_fill(matrix *mat, f32 x);
void mat_fill_rand(matrix *mat, f32 lower, f32 upper);
void mat_scale(matrix *mat, f32 scale);
f32 mat_sum(matrix *mat);
u64 mat_argmax(matrix *mat);
b32 mat_add(matrix *out, const matrix *a, const matrix *b);
b32 mat_sub(matrix *out, const matrix *a, const matrix *b);
b32 mat_mul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
            b8 transpose_a, b8 transpose_b);
b32 mat_relu(matrix *out, const matrix *in);
b32 mat_softmax(matrix *out, const matrix *in);
b32 mat_cross_entropy(matrix *out, const matrix *p, const matrix *q);
b32 mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad);
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out,
                         const matrix *grad);
b32 mat_cross_entropy_add_grad(matrix *p_grad, matrix *q_grad, const matrix *p,
                               const matrix *q, const matrix *grad);

#endif /* ARENA_H */
