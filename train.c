/* training layer, see train.h */

#include "train.h"

#include "arena.h"
#include "base.h"
#include "graph.h"
#include "matrix.h"
#include "model.h"
#include "prng.h"

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
   * skipped that epoch
   */
  u32 num_batches = num_examples / training_desc->batch_size;

  mem_arena_temp scratch = arena_scratch_get(NULL, 0);

  /* shuffle a list of sample indices instead of the data itself, moving
   * one u32 is cheaper than moving a whole input row
   */
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
       * it, and subtracting moves w downhill on the cost
       */
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
     * model never trained on
     */
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
