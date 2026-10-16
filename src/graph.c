/* graph layer: builds the network as nodes, sorts them into programs, and
 * runs the forward pass and backprop over a program
 * the math itself is in matrix.c, every op here calls a mat_* function
 */

#include "graph.h"

#include "arena.h"
#include "base.h"
#include "matrix.h"

#include <stdlib.h>

/* how many of inputs[] an op uses, relies on the op enum's grouping */
#define MV_NUM_INPUTS(op)                                                      \
  ((op) < MV_OP_UNARY_START ? 0 : ((op) < MV_OP_BINARY_START ? 1 : 2))

/* ids are handed out 0, 1, 2, ... in creation order
 * grad is only allocated when something needs it (saves memory on inputs)
 * NULL if the arena is full, the id is only taken once everything fits
 */
model_var *mv_create(mem_arena *arena, model_graph *graph, u32 rows, u32 cols,
                     u32 flags) {
  model_var *out = PUSH_STRUCT(arena, model_var);
  if (out == NULL) {
    return NULL;
  }

  out->flags = flags;
  out->op = MV_OP_CREATE;

  out->val = mat_create(arena, rows, cols);
  if (out->val == NULL) {
    return NULL;
  }

  if (flags & MV_FLAG_REQUIRES_GRAD) {
    out->grad = mat_create(arena, rows, cols);
    if (out->grad == NULL) {
      return NULL;
    }
  }

  out->index = graph->num_vars++;

  return out;
}

/* shared body for one-input ops: make the node, record its op and input
 * if the input needs a grad, so does the output, otherwise backprop would
 * have no path from the cost back to the input
 */
static model_var *mv_unary_impl(mem_arena *arena, model_graph *graph,
                                model_var *input, u32 rows, u32 cols, u32 flags,
                                model_var_op op) {
  if (input->flags & MV_FLAG_REQUIRES_GRAD) {
    flags |= MV_FLAG_REQUIRES_GRAD;
  }

  model_var *out = mv_create(arena, graph, rows, cols, flags);
  if (out == NULL) {
    return NULL;
  }

  out->op = op;
  out->inputs[0] = input;

  return out;
}

/* same as mv_unary_impl for two-input ops, the output needs a grad if
 * either input does
 */
static model_var *mv_binary_impl(mem_arena *arena, model_graph *graph,
                                 model_var *a, model_var *b,
                                 u32 rows, u32 cols, u32 flags,
                                 model_var_op op) {
  if ((a->flags & MV_FLAG_REQUIRES_GRAD) ||
      (b->flags & MV_FLAG_REQUIRES_GRAD)) {
    flags |= MV_FLAG_REQUIRES_GRAD;
  }

  model_var *out = mv_create(arena, graph, rows, cols, flags);
  if (out == NULL) {
    return NULL;
  }

  out->op = op;
  out->inputs[0] = a;
  out->inputs[1] = b;

  return out;
}

/* the mv_* op functions build the graph, they don't compute anything
 * each one checks shapes, works out the output shape, and adds a node
 * (NULL on a shape mismatch), the math runs later in model_prog_compute
 * a NULL input gives a NULL output, so one failure anywhere ends up as a NULL
 * output or cost, which model_create reports, the caller doesn't have to
 * check every node
 */
model_var *mv_relu(mem_arena *arena, model_graph *graph, model_var *input,
                   u32 flags) {
  if (input == NULL) {
    return NULL;
  }

  return mv_unary_impl(arena, graph, input, input->val->rows, input->val->cols,
                       flags, MV_OP_RELU);
}

/* column vectors only, mat_softmax treats the whole matrix as one
 * distribution, which is never what a matrix input wants (attention needs a
 * softmax per row), so reject it here instead of quietly doing the wrong math
 */
model_var *mv_softmax(mem_arena *arena, model_graph *graph, model_var *input,
                      u32 flags) {
  if (input == NULL || input->val->cols != 1) {
    return NULL;
  }

  return mv_unary_impl(arena, graph, input, input->val->rows, input->val->cols,
                       flags, MV_OP_SOFTMAX);
}

model_var *mv_add(mem_arena *arena, model_graph *graph, model_var *a,
                  model_var *b, u32 flags) {
  if (a == NULL || b == NULL) {
    return NULL;
  }
  if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
    return NULL;
  }

  return mv_binary_impl(arena, graph, a, b, a->val->rows, a->val->cols, flags,
                        MV_OP_ADD);
}

model_var *mv_sub(mem_arena *arena, model_graph *graph, model_var *a,
                  model_var *b, u32 flags) {
  if (a == NULL || b == NULL) {
    return NULL;
  }
  if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
    return NULL;
  }

  return mv_binary_impl(arena, graph, a, b, a->val->rows, a->val->cols, flags,
                        MV_OP_SUB);
}

model_var *mv_matmul(mem_arena *arena, model_graph *graph, model_var *a,
                     model_var *b, u32 flags) {
  if (a == NULL || b == NULL) {
    return NULL;
  }
  /* (m x n) * (n x p) -> (m x p), inner dims must match */
  if (a->val->cols != b->val->rows) {
    return NULL;
  }

  return mv_binary_impl(arena, graph, a, b, a->val->rows, b->val->cols, flags,
                        MV_OP_MATMUL);
}

/* p is the desired output (label), q is the prediction, same as in
 * mat_cross_entropy
 */
model_var *mv_cross_entropy(mem_arena *arena, model_graph *graph, model_var *p,
                            model_var *q, u32 flags) {
  if (p == NULL || q == NULL) {
    return NULL;
  }
  if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
    return NULL;
  }

  return mv_binary_impl(arena, graph, p, q, p->val->rows, p->val->cols, flags,
                        MV_OP_CROSS_ENTROPY);
}

/* topological sort: every node out_var depends on, each after its inputs
 * a depth-first search using a manual stack instead of recursion:
 * - 1st time a node is popped: mark it visited, push it back, then push its
 *   inputs on top of it, so they get handled first
 * - 2nd time it's popped: its inputs are all done, so append it to out
 * nodes out_var doesn't depend on are never reached, which is why the
 * forward program leaves out the label and cost nodes
 * out of memory gives a program with vars == NULL
 */
model_program model_prog_create(mem_arena *arena, model_graph *graph,
                                model_var *out_var) {
  /* &arena as a conflict guarantees scratch isn't the arena prog.vars gets
   * pushed to, otherwise releasing scratch would pop prog.vars too
   */
  mem_arena_temp scratch = arena_scratch_get(&arena, 1);
  if (scratch.arena == NULL) {
    return (model_program){0};
  }

  /* indexed by model_var.index, PUSH_ARRAY zeroes it so all start false */
  b8 *visited = PUSH_ARRAY(scratch.arena, b8, graph->num_vars);

  /* no list can hold more than every node, so num_vars is a safe size */
  u32 stack_size = 0;
  u32 out_size = 0;
  model_var **stack = PUSH_ARRAY(scratch.arena, model_var *, graph->num_vars);
  model_var **out = PUSH_ARRAY(scratch.arena, model_var *, graph->num_vars);

  if (visited == NULL || stack == NULL || out == NULL) {
    arena_scratch_release(scratch);
    return (model_program){0};
  }

  stack[stack_size++] = out_var;

  while (stack_size > 0) {
    model_var *cur = stack[--stack_size];

    /* guard, a bad index would read past the end of visited[] */
    if (cur->index >= graph->num_vars) {
      continue;
    }

    /* 2nd pop, inputs are done */
    if (visited[cur->index]) {
      if (out_size < graph->num_vars) {
        out[out_size++] = cur;
      }
      continue;
    }

    /* 1st pop, push cur back under its inputs */
    visited[cur->index] = true;

    if (stack_size < graph->num_vars) {
      stack[stack_size++] = cur;
    }

    u32 num_inputs = MV_NUM_INPUTS(cur->op);
    for (u32 i = 0; i < num_inputs; i++) {
      model_var *input = cur->inputs[i];

      if (input->index >= graph->num_vars || visited[input->index]) {
        continue;
      }

      /* if input is already deeper in the stack (another node pushed
       * it), pull it out and re-push it on top, otherwise cur would
       * pop for the 2nd time and be emitted before its own input
       */
      for (u32 j = 0; j < stack_size; j++) {
        if (stack[j] == input) {
          for (u32 k = j; k < stack_size - 1; k++) {
            stack[k] = stack[k + 1];
          }
          stack_size--;
        }
      }

      if (stack_size < graph->num_vars) {
        stack[stack_size++] = input;
      }
    }
  }

  /* out is in scratch, copy it to the caller's arena so it outlives this
   * function, NZ since memcpy overwrites it right away
   */
  model_program prog = {.size = out_size,
                        .vars = PUSH_ARRAY_NZ(arena, model_var *, out_size)};

  if (prog.vars == NULL) {
    arena_scratch_release(scratch);
    return (model_program){0};
  }

  memcpy(prog.vars, out, sizeof(model_var *) * out_size);

  arena_scratch_release(scratch);

  return prog;
}

/* every shape in a program was checked when its nodes were built, so a mat_*
 * op failing while it runs is a bug in this library, not something the
 * caller could fix, stop right there and say which node instead of training
 * on garbage
 */
static void check_op(b32 ok, const char *pass, const model_var *cur) {
  if (!ok) {
    fprintf(stderr, "%s: op %d on node %u failed\n", pass, cur->op, cur->index);
    abort();
  }
}

/* forward pass: run each node's op in order, writing into its val
 * the order from model_prog_create guarantees a and b are already computed
 * CREATE nodes have no op, their val was set from outside (an image copied
 * in, or the weights)
 * the _START markers are only listed so the switch covers every enum value
 * (clang's -Wswitch warns otherwise)
 */
void model_prog_compute(model_program *prog) {
  for (u32 i = 0; i < prog->size; i++) {
    model_var *cur = prog->vars[i];

    model_var *a = cur->inputs[0];
    model_var *b = cur->inputs[1];

    switch (cur->op) {
    case MV_OP_NULL:
    case MV_OP_CREATE:
      break;

    case MV_OP_UNARY_START:
      break;

    case MV_OP_RELU: {
      check_op(mat_relu(cur->val, a->val), "forward", cur);
    } break;
    case MV_OP_SOFTMAX: {
      check_op(mat_softmax(cur->val, a->val), "forward", cur);
    } break;

    case MV_OP_BINARY_START:
      break;

    case MV_OP_ADD: {
      check_op(mat_add(cur->val, a->val, b->val), "forward", cur);
    } break;
    case MV_OP_SUB: {
      check_op(mat_sub(cur->val, a->val, b->val), "forward", cur);
    } break;
    case MV_OP_MATMUL: {
      /* zero_out = 1, the forward pass overwrites val */
      check_op(mat_mul(cur->val, a->val, b->val, 1, 0, 0), "forward", cur);
    } break;
    case MV_OP_CROSS_ENTROPY: {
      check_op(mat_cross_entropy(cur->val, a->val, b->val), "forward", cur);
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
void model_prog_compute_grads(model_program *prog) {
  /* zero the grads left over from the last run, except parameters, their
   * grads add up across a whole batch and model_train clears them
   */
  for (u32 i = 0; i < prog->size; i++) {
    model_var *cur = prog->vars[i];

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
   * the gradient of that sum
   */
  mat_fill(prog->vars[prog->size - 1]->grad, 1.0f);

  for (i64 i = (i64)prog->size - 1; i >= 0; i--) {
    model_var *cur = prog->vars[i];

    if ((cur->flags & MV_FLAG_REQUIRES_GRAD) == 0) {
      continue;
    }

    model_var *a = cur->inputs[0];
    model_var *b = cur->inputs[1];

    u32 num_inputs = MV_NUM_INPUTS(cur->op);

    /* skip if no input needs a grad, nowhere to send it */
    if (num_inputs == 1 &&
        (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
      continue;
    }

    if (num_inputs == 2 &&
        (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD &&
        (b->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
      continue;
    }

    switch (cur->op) {
    case MV_OP_NULL:
    case MV_OP_CREATE:
      break;

    case MV_OP_UNARY_START:
      break;

    case MV_OP_RELU: {
      check_op(mat_relu_add_grad(a->grad, a->val, cur->grad), "backward", cur);
    } break;
    case MV_OP_SOFTMAX: {
      check_op(mat_softmax_add_grad(a->grad, cur->val, cur->grad), "backward",
               cur);
    } break;

    case MV_OP_BINARY_START:
      break;

    /* d(a + b)/da = d(a + b)/db = 1, grad passes through as is */
    case MV_OP_ADD: {
      if (a->flags & MV_FLAG_REQUIRES_GRAD) {
        check_op(mat_add(a->grad, a->grad, cur->grad), "backward", cur);
      }

      if (b->flags & MV_FLAG_REQUIRES_GRAD) {
        check_op(mat_add(b->grad, b->grad, cur->grad), "backward", cur);
      }
    } break;

    /* d(a - b)/da = 1, d(a - b)/db = -1, so b's grad is subtracted */
    case MV_OP_SUB: {
      if (a->flags & MV_FLAG_REQUIRES_GRAD) {
        check_op(mat_add(a->grad, a->grad, cur->grad), "backward", cur);
      }

      if (b->flags & MV_FLAG_REQUIRES_GRAD) {
        check_op(mat_sub(b->grad, b->grad, cur->grad), "backward", cur);
      }
    } break;

    /* C = A * B: dA += dC * B^T, dB += A^T * dC
     * the transpose flags do this without building copies, and
     * zero_out = 0 accumulates
     */
    case MV_OP_MATMUL: {
      if (a->flags & MV_FLAG_REQUIRES_GRAD) {
        check_op(mat_mul(a->grad, cur->grad, b->val, 0, 0, 1), "backward", cur);
      }

      if (b->flags & MV_FLAG_REQUIRES_GRAD) {
        check_op(mat_mul(b->grad, a->val, cur->grad, 0, 1, 0), "backward", cur);
      }
    } break;

    /* p = label (no REQUIRES_GRAD, so p->grad is NULL and skipped),
     * q = prediction
     */
    case MV_OP_CROSS_ENTROPY: {
      model_var *p = a;
      model_var *q = b;

      check_op(mat_cross_entropy_add_grad(p->grad, q->grad, p->val, q->val,
                                          cur->grad),
               "backward", cur);
    } break;
    }
  }
}
