import struct
import tensorflow_datasets as tfds
import numpy as np

def save_mat(path, arr):
    arr = np.asarray(arr, dtype="<f4") # 4-byte float = 32-bit floats
    rows, cols = arr.shape

    with open(path, "wb") as f:
        # < little-endian
        # 4s a 4-byte string, b"MAT1"
        # I an unsigned 320but int, three times for version, rows, cols
        f.write(struct.pack("<4sIII", b"MAT1", 1, rows, cols))
        f.write(arr.tobytes())

def ds_to_numpy(ds):
    images = []
    labels = []
    for image, label in ds:
        images.append(image.numpy())
        labels.append(label.numpy())

    return np.array(images), np.array(labels)

train_ds, test_ds = tfds.load("mnist", split=["train", "test"], as_supervised=True)

train_images, train_labels = ds_to_numpy(train_ds)
test_images, test_labels = ds_to_numpy(test_ds)

# normalize pixels, better for training
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

# save_mat needs 2D arrays: images (N, 28, 28, 1) -> (N, 784), labels (N,) -> (N, 1)
train_images = train_images.reshape(len(train_images), -1)
test_images = test_images.reshape(len(test_images), -1)
train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)

save_mat("train_images.mat", train_images)
save_mat("train_labels.mat", train_labels)
save_mat("test_images.mat", test_images)
save_mat("test_labels.mat", test_labels)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
