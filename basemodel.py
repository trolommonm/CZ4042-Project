# Base Model =================================================================================
# G. Levi and T. Hassner, “Age and gender classification using convolutional neural networks.” 
# in IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) workshops, 2015
# https://paperswithcode.com/paper/age-and-gender-classification-using

from json import load
from pyexpat import model
import tensorflow as tf 
from tensorflow import keras
from dataloader import load_adience_dataset
from datapreparation import get_adience_num_images
import argparse

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256 
VAL_FOLD = 1    # specify the fold number for loading the adience dataset

# bs: batch_size
def load_data(bs):
    def img_preprocessing(x): 
        # resize the image 
        x = tf.image.resize(x, size=(RESIZE_HEIGHT, RESIZE_WIDTH))
        return x
    
    train_ds, val_ds = load_adience_dataset(VAL_FOLD)

    # Normalizing the images
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

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
    input_shape = (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    inputs = keras.Input(shape=input_shape)
    conv1 = keras.layers.Conv2D(96, [7,7], [4,4], activation='relu', padding='VALID')(inputs)
    pool1 = keras.layers.MaxPooling2D(3, 2, padding='VALID')(conv1)
    norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, beta=0.75)
    conv2 = keras.layers.Conv2D(256, [5,5], [1,1], activation='relu', padding='SAME')(norm1)
    pool2 = keras.layers.MaxPooling2D(3, 2, padding='VALID')(conv2)
    norm2 = tf.nn.local_response_normalization(pool2, 5, alpha=0.0001, beta=0.75)
    conv3 = keras.layers.Conv2D(384, [3,3], [1,1], activation='relu', padding='SAME')(norm2)
    pool3 = keras.layers.MaxPooling2D(3, 2, padding='VALID')(conv3)
    flat = keras.layers.Flatten()(pool3)
    full1 = keras.layers.Dense(512)(flat)
    drop1 = keras.layers.Dropout(0.5)(full1)
    full2 = keras.layers.Dense(512)(drop1)
    drop2 = keras.layers.Dropout(0.5)(full2)
    outputs = keras.layers.Dense(1, activation="softmax")(drop2)
    model = keras.Model(inputs, outputs)

    return model

def main(args):
    bs = args.bs
    train_ds, val_ds = load_data(bs)
    num_train, num_val = get_adience_num_images(VAL_FOLD)

    model = build_model()
    model.summary()
    print()

    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(lr=args.lr)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    model.fit(train_ds, epochs=args.num_epochs, steps_per_epoch=num_train // bs, validation_data=val_ds)
    model.save('model/basemodel_test_2.h5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base Model Training using Adience dataset')
    parser.add_argument("-lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("-bs", type=int, default=50, help="batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="number of epochs")

    main(parser.parse_args())