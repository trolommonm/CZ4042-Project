import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from dataloader import load_adience_dataset
from datapreparation import get_adience_num_images
from utils import CustomTensorBoard
import argparse
from datetime import datetime
import os

tf.keras.mixed_precision.set_global_policy("mixed_float16")


def load_data(fold, bs):
    def img_preprocessing(x):
        x = tf.image.resize(x, size=(224, 224))

        return x

    ad_train_ds, ad_val_ds = load_adience_dataset(fold)
    ad_train_ds = ad_train_ds.map(lambda x, y: (img_preprocessing(x), y))
    ad_val_ds = ad_val_ds.map(lambda x, y: (img_preprocessing(x), y))

    ad_train_ds = ad_train_ds.cache() \
        .shuffle(100) \
        .batch(bs, drop_remainder=True) \
        .repeat()
    ad_train_ds = ad_train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    ad_val_ds = ad_val_ds.cache() \
                         .batch(bs) \
                         .prefetch(buffer_size=tf.data.AUTOTUNE)

    return ad_train_ds, ad_val_ds


def train(fold, args, output_dir):
    ad_train_ds, ad_val_ds = load_data(fold, args.bs)
    ad_num_train, _ = get_adience_num_images(fold)

    base_model = keras.models.load_model(args.model_path)
    x = keras.layers.Dense(512, activation="relu")(base_model.layers[-4].output)
    # x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    # x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(base_model.layers[0].input, outputs)

    # first warm up the model by freezing the base layers and
    # only training the fc layers
    for l in model.layers[:-3]:
        l.trainable = False
    assert len([l for l in model.layers if l.trainable == True]) == 3

    optimizer = keras.optimizers.Adam(lr=0.01)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    print("Warm up for 10 epochs...")
    model.fit(ad_train_ds, epochs=10, steps_per_epoch=ad_num_train // args.bs, validation_data=ad_val_ds)
    print()

    # fine tuning
    for l in model.layers[:-26]:
        l.trainable = False
    for l in model.layers[-26:]:
        l.trainable = True

    model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(output_dir,
                                                                      "best_model_{epoch:02d}-{val_accuracy:.2f}"),
                                                save_weights_only=False,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True)
    tensorboard_callback = CustomTensorBoard(log_dir=os.path.join(output_dir, "logs"))
    csv_callback = CSVLogger(os.path.join(output_dir, "training.log"))

    optimizer = keras.optimizers.Adam(lr=0.001)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    print("Finetuning...")
    model.fit(ad_train_ds,
              epochs=100,
              steps_per_epoch=ad_num_train // args.bs,
              validation_data=ad_val_ds,
              callbacks=[model_checkpoint_callback, tensorboard_callback, csv_callback])


def main(args):
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    for i in range(5):
        output_dir = os.path.join("./output/finetuning/", datetime_now, f"fold_{i}")
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        train(i, args, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetuning pretrained (with CelebA) model on Adience dataset')
    parser.add_argument("--model-path", help="path to the pretrained model (SavedModel format)")
    parser.add_argument("-bs", type=int, default=128, help="batch size")

    main(parser.parse_args())
