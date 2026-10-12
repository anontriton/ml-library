/* model layer, see model.h */

#include "model.h"

#include "arena.h"
#include "base.h"
#include "graph.h"
#include "matrix.h"

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

/* runs input -> output, copy a sample into model->input->val first, the
 * result ends up in model->output->val
 */
void model_feedforward(model_context *model) {
  model_prog_compute(&model->forward_prog);
}
