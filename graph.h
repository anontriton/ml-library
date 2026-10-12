#ifndef GRAPH_H
#define GRAPH_H

#include "base.h"

#include "arena.h"
#include "matrix.h"

/* graph layer
 * every matrix in the network (input, weights, intermediate results, cost) is
 * a node (model_var) that remembers which op made it from which input nodes
 */

/* CREATE: made directly by mv_create (input, weights), no inputs
 * ops are grouped by input count, the _START markers (not real ops) split
 * the groups so MV_NUM_INPUTS is just two comparisons
 */
typedef enum {
  MV_OP_NULL = 0,
  MV_OP_CREATE,

  MV_OP_UNARY_START,

  MV_OP_RELU,
  MV_OP_SOFTMAX,

  MV_OP_BINARY_START,

  MV_OP_ADD,
  MV_OP_SUB,
  MV_OP_MATMUL,
  MV_OP_CROSS_ENTROPY,
} model_var_op;

/* binary ops take the most inputs, so that's the size of inputs[] */
#define MODEL_VAR_MAX_INPUTS 2

/* REQUIRES_GRAD: gets a grad matrix, backprop computes d(cost)/d(this)
 * PARAMETER:     a weight or bias, what training updates
 */
typedef enum {
  MV_FLAG_NONE = 0,
  MV_FLAG_REQUIRES_GRAD = (1 << 0),
  MV_FLAG_PARAMETER = (1 << 1),
} model_var_flags;

/* hands out node ids, the nodes live in the arena and are linked by their
 * inputs[] ptrs, so the graph doesn't store them
 */
typedef struct {
  u32 num_vars; /* number of nodes that exist, also next node's index */
} model_graph;

/* one node in the graph */
typedef struct model_var {
  u32 index; /* unique id from mv_create, used to index visited[] */
  u32 flags;

  matrix *val;
  /* d(cost)/d(val), same shape as val, only allocated if REQUIRES_GRAD */
  matrix *grad;

  model_var_op op;
  /* unused slots stay NULL */
  struct model_var *inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

/* a flat list of nodes where every node comes after its inputs (a
 * topological order), built once by model_prog_create
 * walking it front to back = forward pass, back to front = backprop
 */
typedef struct {
  model_var **vars;
  u32 size;
} model_program;

model_var *mv_create(mem_arena *arena, model_graph *graph, u32 rows, u32 cols,
                     u32 flags);

model_var *mv_relu(mem_arena *arena, model_graph *graph, model_var *input,
                   u32 flags);
model_var *mv_softmax(mem_arena *arena, model_graph *graph, model_var *input,
                      u32 flags);

model_var *mv_add(mem_arena *arena, model_graph *graph, model_var *a,
                  model_var *b, u32 flags);
model_var *mv_sub(mem_arena *arena, model_graph *graph, model_var *a,
                  model_var *b, u32 flags);
model_var *mv_matmul(mem_arena *arena, model_graph *graph, model_var *a,
                     model_var *b, u32 flags);
model_var *mv_cross_entropy(mem_arena *arena, model_graph *graph, model_var *p,
                            model_var *q, u32 flags);

/* programs, the sorted node lists that actually run the math */
model_program model_prog_create(mem_arena *arena, model_graph *graph,
                                model_var *out_var);
void model_prog_compute(model_program *prog);
void model_prog_compute_grads(model_program *prog);

#endif /* GRAPH_H */
