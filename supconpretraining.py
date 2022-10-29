import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from utils import resnet18, SupervisedContrastiveLoss, CustomTensorBoard
from dataloader import load_celeba_dataset
from datapreparation import get_celeba_num_images
import argparse
import os
import json
from datetime import datetime


def create_augmentation():
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.02),
            layers.RandomWidth(0.2),
            layers.RandomHeight(0.2),
        ]
    )

    return data_augmentation


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


def create_encoder(augmentation):
    # resnet = keras.applications.ResNet50V2(
    #     include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    # )

    inputs = keras.Input(shape=(224, 224, 3))
    x = augmentation(inputs)
    x = resnet18(x)
    outputs = layers.GlobalAveragePooling2D()(x)
    # outputs = resnet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def add_projection_head(encoder):
    inputs = keras.Input(shape=(224, 224, 3))
    features = encoder(inputs)
    outputs = layers.Dense(128, activation="relu")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def main(args):
    if args.mp:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_ds, val_ds = load_data(args.bs)
    num_train, num_val = get_celeba_num_images()

    t, _ = load_celeba_dataset()
    t = t.map(lambda x, y: x)
    augmentation = create_augmentation()
    augmentation.layers[0].adapt(t, steps=num_train // args.bs)

    encoder = create_encoder(augmentation)
    encoder_proj_head = add_projection_head(encoder)
    encoder_proj_head.summary()
    print()

    output_dir = os.path.join("./output/supcon_pretraining/", datetime.now().strftime("%Y%m%d-%H%M%S"))
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        f.write(json.dumps(args.__dict__))

    model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(output_dir,
                                                                      "best_model_{epoch:02d}-{val_accuracy:.2f}"),
                                                save_weights_only=False,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True)
    tensorboard_callback = CustomTensorBoard(log_dir=logs_dir)
    csv_callback = CSVLogger(os.path.join(output_dir, "training.log"))

    encoder_proj_head.compile(
        optimizer=keras.optimizers.SGD(args.lr, momentum=0.9),
        loss=SupervisedContrastiveLoss(args.temperature),
    )
    encoder_proj_head.fit(train_ds,
                          epochs=100,
                          steps_per_epoch=num_train // args.bs,
                          callbacks=[model_checkpoint_callback, tensorboard_callback, csv_callback])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretraining on CelebA dataset using Supervised Contrastive Learning')
    parser.add_argument("-lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("-bs", type=int, default=1024, help="batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--temperature", type=float, default=0.05, help="temperature for the SupCon loss")
    parser.add_argument("-mp", "--mixed-precision", type=bool, default=True, help="mixed precision training")

    main(parser.parse_args())
