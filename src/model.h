#ifndef MODEL_H
#define MODEL_H

#include "arena.h"
#include "graph.h"

/* model layer: wraps a finished graph so training and inference only deal
 * with its 4 special nodes, not the whole graph
 */

/* a graph's special nodes plus the two programs compiled from them
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

model_context *model_create(mem_arena *arena, const model_desc *desc);
void model_feedforward(model_context *model);

#endif /* MODEL_H */
