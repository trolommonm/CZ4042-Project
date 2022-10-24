import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from utils import resnet18
from dataloader import load_celeba_dataset
from datapreparation import get_celeba_num_images
from utils import CustomTensorBoard
import argparse
import datetime
import os


def load_data(bs):
    def img_preprocessing(x):
        x = tf.image.resize(x, size=(224, 224))
        return x

    train_ds, val_ds = load_celeba_dataset()
    train_ds = train_ds.map(lambda x, y: (img_preprocessing(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (img_preprocessing(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.cache() \
                       .shuffle(100) \
                       .batch(bs, drop_remainder=True) \
                       .repeat() \
                       .prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(bs)

    return train_ds, val_ds


def build_model():
    input_shape = (224, 224, 3)
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.RandomFlip('horizontal')(inputs)
    x = keras.layers.RandomRotation(0.2)(x)
    x = keras.layers.RandomZoom(0.2, 0.2)(x)
    x = keras.layers.RandomTranslation(0.2, 0.2)(x)
    x = resnet18(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    return model


def main(args):
    if args.mp:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    bs = args.bs
    train_ds, val_ds = load_data(bs)
    num_train, num_val = get_celeba_num_images()

    model = build_model()
    model.summary()
    print()

    output_dir = os.path.join("./output/pretraining/", datetime.now().strftime("%Y%m%d-%H%M%S"))
    logs_dir = os.path.join(output_dir, "logs")

    model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(output_dir,
                                                                      "best_model_{epoch:02d}-{val_loss:.2f}"),
                                                save_weights_only=False,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True)
    tensorboard_callback = CustomTensorBoard(log_dir=logs_dir)
    csv_callback = CSVLogger(os.path.join(output_dir, "training.log"))

    loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=args.lr)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    model.fit(train_ds,
              epochs=args.num_epochs,
              steps_per_epoch=num_train // bs,
              validation_data=val_ds,
              callbacks=[model_checkpoint_callback, tensorboard_callback, csv_callback])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretraining using CelebA dataset')
    parser.add_argument("-lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("-bs", type=int, default=128, help="batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("-mp", "--mixed-precision", type=bool, default=True, help="mixed precision training")

    main(parser.parse_args())
