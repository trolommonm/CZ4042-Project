import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from dataloader import load_adience_dataset
from datapreparation import get_adience_num_images
from utils import CustomTensorBoard, CosineAnnealWithWamrup
import argparse
from datetime import datetime
import os
import json

IMG_SIZE = (150, 150)


def load_data(fold, bs):
    def img_preprocessing(x):
        x = tf.image.resize(x, size=IMG_SIZE)

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


def create_classifier(encoder, trainable=True):
    hidden_units = 512
    dropout_rate = 0.5

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    features = encoder(inputs)
    # features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    # features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(1, activation="sigmoid")(features)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def train(fold, args, output_dir):
    ad_train_ds, ad_val_ds = load_data(fold, args.bs)
    ad_num_train, ad_num_val = get_adience_num_images(fold)

    # remove the projection head from the pre-trained model and only keep the encoder
    pretrained_model = keras.models.load_model(args.model_path, compile=False)
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    outputs = pretrained_model.layers[1](inputs)
    encoder = keras.Model(inputs=inputs, outputs=outputs)

    # add a fully connected layer on top of the encoder and freeze the weights in the encoder
    classifier = create_classifier(encoder, trainable=False)
    classifier.compile(
        optimizer=keras.optimizers.SGD(args.lr, momentum=0.9),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    classifier.summary()

    model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(output_dir,
                                                                      "best_model"),
                                                save_weights_only=False,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True)
    tensorboard_callback = CustomTensorBoard(log_dir=os.path.join(output_dir, "logs"))
    csv_callback = CSVLogger(os.path.join(output_dir, "training.log"))
    lr_scheduler = CosineAnnealWithWamrup(learning_rate_base=args.lr,
                                          total_steps=args.num_epochs * ad_num_train // args.bs,
                                          warmup_learning_rate=0.0,
                                          warmup_steps=args.num_warmup_epochs * ad_num_train // args.bs,
                                          hold_base_rate_steps=0)

    classifier.fit(ad_train_ds,
                   epochs=args.num_epochs,
                   steps_per_epoch=ad_num_train // args.bs,
                   validation_data=ad_val_ds,
                   callbacks=[model_checkpoint_callback, tensorboard_callback, csv_callback, lr_scheduler])


def main(args):
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

    for i in range(5):
        output_dir = os.path.join("./output/supcon_finetuning/", datetime_now, f"fold_{i}")
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        train(i, args, output_dir)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        f.write(json.dumps(args.__dict__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetuning on Adience dataset with pretrained encoder on CelebA')
    parser.add_argument("--model-path", help="path to the pretrained encoder (SavedModel format)")
    parser.add_argument("-bs", type=int, default=128, help="batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--num-warmup-epochs", type=int, default=100, help="number of warmup epochs")
    parser.add_argument("-lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("-mp", "--mixed-precision", type=bool, default=True, help="mixed precision training")

    main(parser.parse_args())