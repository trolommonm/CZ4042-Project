# Levi Hassner Model ===========================================================================
# G. Levi and T. Hassner, “Age and gender classification using convolutional neural networks.” 
# in IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) workshops, 2015
# https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from dataloader import load_adience_dataset
from datapreparation import get_adience_num_images
from utils import CustomTensorBoard
from datetime import datetime
import json
import argparse
import os

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256


# bs: batch_size
def load_data(fold, bs):
    def img_preprocessing(x):
        # resize the image 
        x = tf.image.resize(x, size=(RESIZE_HEIGHT, RESIZE_WIDTH))
        return x

    train_ds, val_ds = load_adience_dataset(fold)

    # Scale the images tp [-1, 1]
    # train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    # val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.map(lambda x, y: (img_preprocessing(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (img_preprocessing(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.cache() \
                       .shuffle(100) \
                       .batch(bs, drop_remainder=True) \
                       .repeat() \
                       .prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache() \
                   .batch(bs) \
                   .prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_model(augment=False):
    inputs = keras.Input(shape=(RESIZE_HEIGHT, RESIZE_WIDTH, 3))
    x = inputs
    if augment:
        x = keras.layers.RandomFlip('horizontal')(x)
        x = keras.layers.RandomRotation(0.2)(x)
        x = keras.layers.RandomZoom(0.2, 0.2)(x)
        x = keras.layers.RandomTranslation(0.2, 0.2)(x)
    x = keras.layers.Conv2D(96, (7, 7), strides=(4, 4), padding="valid", activation="relu")(x)
    x = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="valid")(x)
    x = tf.nn.local_response_normalization(x, alpha=0.0001, beta=0.75)
    x = keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="valid")(x)
    x = tf.nn.local_response_normalization(x, alpha=0.0001, beta=0.75)
    x = keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="valid")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


def train(fold, args, output_dir):
    train_ds, val_ds = load_data(fold, args.bs)
    num_train, num_val = get_adience_num_images(fold)

    model = build_model(args.augment)

    model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(output_dir,
                                                                      "best_model"),
                                                save_weights_only=False,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True)
    tensorboard_callback = CustomTensorBoard(log_dir=os.path.join(output_dir, "logs"))
    csv_callback = CSVLogger(os.path.join(output_dir, "training.log"))

    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.SGD(lr=args.lr)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    model.fit(train_ds,
              epochs=args.num_epochs,
              steps_per_epoch=num_train // args.bs,
              validation_data=val_ds,
              callbacks=[model_checkpoint_callback, tensorboard_callback, csv_callback])


def main(args):
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    for i in range(5):
        output_dir = os.path.join("./output/levihassner/", datetime_now, f"fold_{i}")
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        train(i, args, output_dir)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        f.write(json.dumps(args.__dict__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline Model Training using Adience dataset')
    parser.add_argument("-lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("-bs", type=int, default=64, help="batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--augment", type=bool, default=False, help="apply image augmentation")
    parser.add_argument("-mp", "--mixed-precision", type=bool, default=True, help="mixed precision training")

    main(parser.parse_args())
