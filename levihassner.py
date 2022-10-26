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
    # 
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


def build_model():
    input_shape = (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.RandomFlip('horizontal')(inputs)
    x = keras.layers.RandomRotation(0.2)(x)
    x = keras.layers.RandomZoom(0.2, 0.2)(x)
    x = keras.layers.RandomTranslation(0.2, 0.2)(x)
    x = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)(x)
    conv1 = keras.layers.Conv2D(96, [7, 7], [4, 4], activation='relu', padding='VALID')(x)
    pool1 = keras.layers.MaxPooling2D(3, 2, padding='VALID')(conv1)
    norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, beta=0.75)
    conv2 = keras.layers.Conv2D(256, [5, 5], [1, 1], activation='relu', padding='SAME')(norm1)
    pool2 = keras.layers.MaxPooling2D(3, 2, padding='VALID')(conv2)
    norm2 = tf.nn.local_response_normalization(pool2, 5, alpha=0.0001, beta=0.75)
    conv3 = keras.layers.Conv2D(384, [3, 3], [1, 1], activation='relu', padding='SAME')(norm2)
    pool3 = keras.layers.MaxPooling2D(3, 2, padding='VALID')(conv3)
    flat = keras.layers.Flatten()(pool3)
    full1 = keras.layers.Dense(512)(flat)
    drop1 = keras.layers.Dropout(0.5)(full1)
    full2 = keras.layers.Dense(512)(drop1)
    drop2 = keras.layers.Dropout(0.5)(full2)
    outputs = keras.layers.Dense(1, activation="softmax")(drop2)
    model = keras.Model(inputs, outputs)

    return model


def train(fold, args, output_dir):
    train_ds, val_ds = load_data(fold, args.bs)
    num_train, num_val = get_adience_num_images(fold)

    model = build_model()

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
    parser.add_argument("-lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("-bs", type=int, default=128, help="batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs")

    main(parser.parse_args())
