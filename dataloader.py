import os
import tensorflow as tf
from PIL import Image


def adience_data_gen(val_fold, train=True):
    data_path = "data"
    txt_file = "train.txt" if train else "val.txt"
    txt_file_path = os.path.join(data_path, f"Adience/fold_{val_fold}", txt_file)

    with open(txt_file_path) as f:
        for line in f:
            image_file = line.split(" ")[0]
            label = int(line.split(" ")[1])
            pil_img = Image.open(os.path.join(data_path, "Adience/aligned", image_file))
            img_array = tf.keras.utils.img_to_array(pil_img)

            yield img_array, label


def load_adience_dataset(val_fold):
    train_ds = tf.data.Dataset.from_generator(adience_data_gen,
                                              output_types=(tf.float32, tf.int8),
                                              args=(val_fold, True))
    val_ds = tf.data.Dataset.from_generator(adience_data_gen,
                                            output_types=(tf.float32, tf.int8),
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
            pil_img = Image.open(os.path.join(data_path, "CelebA/img_align_celeba", image_file))
            img_array = tf.keras.utils.img_to_array(pil_img)

            yield img_array, label


def load_celeba_dataset():
    train_ds = tf.data.Dataset.from_generator(celeba_data_gen,
                                              output_types=(tf.float32, tf.int8),
                                              args=(True,))
    val_ds = tf.data.Dataset.from_generator(celeba_data_gen(),
                                            output_types=(tf.float32, tf.int8),
                                            args=(False,))

    return train_ds, val_ds
