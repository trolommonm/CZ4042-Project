import os
import tensorflow as tf


def adience_data_gen(val_fold, train=True):
    data_path = "data"
    txt_file = "train.txt" if train else "val.txt"
    txt_file_path = os.path.join(data_path, f"Adience/fold_{val_fold}", txt_file)

    with open(txt_file_path) as f:
        for line in f:
            image_file = line.split(" ")[0]
            label = int(line.split(" ")[1])
            img = tf.io.read_file(os.path.join(data_path, "Adience/aligned", image_file))
            img_tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)

            yield img_tensor, tf.expand_dims(label, axis=0)


def load_adience_dataset(val_fold):
    train_ds = tf.data.Dataset.from_generator(adience_data_gen,
                                              output_signature=(
                                                  tf.TensorSpec(shape=(816, 816, 3), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(None,), dtype=tf.float32)
                                              ),
                                              args=(val_fold, True))
    val_ds = tf.data.Dataset.from_generator(adience_data_gen,
                                            output_signature=(
                                                tf.TensorSpec(shape=(816, 816, 3), dtype=tf.float32),
                                                tf.TensorSpec(shape=(None,), dtype=tf.float32)
                                            ),
                                            args=(val_fold, False))

    return train_ds, val_ds


def celeba_data_gen(train=True):
    data_path = "data"
    txt_file = "train.txt" if train else "test.txt"
    txt_file_path = os.path.join(data_path, "CelebA", txt_file)

    with open(txt_file_path) as f:
        for line in f:
            image_file = line.split(" ")[0]
            label = int(line.split(" ")[1])
            img = tf.io.read_file(os.path.join(data_path, "CelebA/img_align_celeba", image_file))
            img_tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)

            yield img_tensor, tf.expand_dims(label, axis=0)


def load_celeba_dataset():
    train_ds = tf.data.Dataset.from_generator(celeba_data_gen,
                                              output_signature=(
                                                  tf.TensorSpec(shape=(218, 178, 3), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(None,), dtype=tf.float32)
                                              ),
                                              args=(True,))
    val_ds = tf.data.Dataset.from_generator(celeba_data_gen,
                                            output_signature=(
                                                tf.TensorSpec(shape=(218, 178, 3), dtype=tf.float32),
                                                tf.TensorSpec(shape=(None,), dtype=tf.float32)
                                            ),
                                            args=(False,))

    return train_ds, val_ds
