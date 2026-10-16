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
b32 model_train(model_context *model,
                const model_training_desc *training_desc) {
  matrix *train_images = training_desc->train_images;
  matrix *train_labels = training_desc->train_labels;
  matrix *test_images = training_desc->test_images;
  matrix *test_labels = training_desc->test_labels;

  if (train_images == NULL || train_labels == NULL || test_images == NULL ||
      test_labels == NULL) {
    fprintf(stderr, "model_train: all four data matrices must be set\n");
    return false;
  }

  u32 num_examples = train_images->rows;
  u32 input_size = train_images->cols;
  u32 output_size = train_labels->cols;
  u32 num_tests = test_images->rows;

  /* every sample gets memcpy'd straight into the input and label nodes, so
   * a row has to be exactly as long as the node or the copy runs past one of
   * them
   */
  u64 model_input_size = (u64)model->input->val->rows * model->input->val->cols;
  u64 model_output_size =
      (u64)model->desired_output->val->rows * model->desired_output->val->cols;
  if (input_size != model_input_size || test_images->cols != model_input_size) {
    fprintf(stderr,
            "model_train: model input has %llu values but train/test rows "
            "have %u / %u\n",
            (unsigned long long)model_input_size, input_size,
            test_images->cols);
    return false;
  }
  if (output_size != model_output_size ||
      test_labels->cols != model_output_size) {
    fprintf(stderr,
            "model_train: model output has %llu values but train/test label "
            "rows have %u / %u\n",
            (unsigned long long)model_output_size, output_size,
            test_labels->cols);
    return false;
  }

  /* row i of labels goes with row i of the inputs */
  if (train_labels->rows != num_examples || test_labels->rows != num_tests) {
    fprintf(stderr, "model_train: inputs and labels have different counts\n");
    return false;
  }

  /* 0 would divide by zero below, more than num_examples would mean 0
   * batches, so nothing would ever train
   */
  if (training_desc->batch_size == 0 ||
      training_desc->batch_size > num_examples) {
    fprintf(stderr, "model_train: batch_size must be 1 to %u, got %u\n",
            num_examples, training_desc->batch_size);
    return false;
  }

  /* integer division, leftover samples that don't fill a batch are
   * skipped that epoch
   */
  u32 num_batches = num_examples / training_desc->batch_size;

  mem_arena_temp scratch = arena_scratch_get(NULL, 0);
  if (scratch.arena == NULL) {
    fprintf(stderr, "model_train: out of memory\n");
    return false;
  }

  /* shuffle a list of sample indices instead of the data itself, moving
   * one u32 is cheaper than moving a whole input row
   */
  u32 *training_order = PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);
  if (training_order == NULL) {
    arena_scratch_release(scratch);
    fprintf(stderr, "model_train: out of memory\n");
    return false;
  }
  for (u32 i = 0; i < num_examples; i++) {
    training_order[i] = i;
  }

  for (u32 epoch = 0; epoch < training_desc->epochs; epoch++) {
    /* fisher-yates: walk from the back, swap each slot with a random slot
     * at or before it, so every order is equally likely
     * random swaps between any two slots don't do that, some orders come up
     * more than others and ~13.5% of samples never move
     * num_examples >= batch_size >= 1 (checked above), so - 1 can't wrap
     * % (i + 1) is a tiny bit biased since 2^32 isn't a multiple of i + 1,
     * at most ~i / 2^32, about 0.001% for mnist, not worth fixing
     */
    for (u32 i = num_examples - 1; i > 0; i--) {
      u32 j = prng_rand() % (i + 1);

      u32 tmp = training_order[i];
      training_order[i] = training_order[j];
      training_order[j] = tmp;
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

        memcpy(model->input->val->data,
               train_images->data + (u64)index * input_size,
               sizeof(f32) * input_size);

        memcpy(model->desired_output->val->data,
               train_labels->data + (u64)index * output_size,
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
      memcpy(model->input->val->data, test_images->data + (u64)i * input_size,
             sizeof(f32) * input_size);

      memcpy(model->desired_output->val->data,
             test_labels->data + (u64)i * output_size,
             sizeof(f32) * output_size);

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

  return true;
}
