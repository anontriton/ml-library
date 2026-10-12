#ifndef TRAIN_H
#define TRAIN_H

#include "base.h"
#include "matrix.h"
#include "model.h"

/* training layer: mini-batch gradient descent on a model, plus a test pass
 * after every epoch
 */

/* the data sets have one sample per row, the labels one one-hot row per
 * sample (as wide as the model's output), and row i of labels goes with row
 * i of images
 * all fields required
 */
typedef struct {
  matrix *train_images;
  matrix *train_labels;
  matrix *test_images;
  matrix *test_labels;

  u32 epochs;
  u32 batch_size;
  f32 learning_rate;
} model_training_desc;

void model_train(model_context *model,
                 const model_training_desc *training_desc);

#endif /* TRAIN_H */
