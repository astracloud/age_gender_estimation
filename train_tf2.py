import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


TRAIN_IMAGE_PATH = 'faces_hawk/train_set/*.jpg'
VALID_IMAGE_PATH = 'faces_hawk/val_set/*.jpg'
AUTOTUNE = tf.data.experimental.AUTOTUNE

GENDER_CATEGORY = np.array(['f', 'm'])
IMG_SIZE = 128
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
USE_BIAS = False
WEIGHT_INIT = "he_normal"
MAX_AGE = 70

EMBEDDING_SIZE = 128
WEIGHT_DECAY = 0.01
EPOCHS = 2000
BATCH_SIZE = 128


def main():
    model = get_model(INPUT_SHAPE)

    train_data_dir = glob.glob(TRAIN_IMAGE_PATH)
    train_image_count = len(train_data_dir)
    train_steps_per_epoch = train_image_count // BATCH_SIZE
    print(f'{train_image_count} of images have to train.')

    valid_data_dir = glob.glob(VALID_IMAGE_PATH)
    valid_image_count = len(valid_data_dir)
    valid_steps_per_epoch = valid_image_count // BATCH_SIZE
    print(f'{valid_image_count} of images have to valid.')

    train_list_ds = tf.data.Dataset.list_files(TRAIN_IMAGE_PATH)
    train_main_ds = train_list_ds.map(process_path_with_aug, num_parallel_calls=AUTOTUNE)
    train_main_ds = add_batch(train_main_ds, BATCH_SIZE, training=True)

    valid_list_ds = tf.data.Dataset.list_files(VALID_IMAGE_PATH)
    valid_main_ds = valid_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    valid_main_ds = add_batch(valid_main_ds, BATCH_SIZE)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=1,
            mode='min',
            baseline=None,
            restore_best_weights=False),
        ModelCheckpoint(
            'checkpoints/{epoch:02d}-{val_loss:.2f}.h5',
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min")
    ]

    model.fit(
        train_main_ds,
        steps_per_epoch=train_steps_per_epoch,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_main_ds,
        validation_steps=valid_steps_per_epoch)


def process_path_with_aug(file_path):
    age, gender = get_label(file_path)
    img = decode_img(file_path, aug=True)
    return img, (age, gender)


def process_path(file_path):
    age, gender = get_label(file_path)
    img = decode_img(file_path)
    return img, (age, gender)


def get_label(file_path):
    file_name = tf.strings.split(file_path, '/')[-1]
    parts = tf.strings.split(file_name, '_')
    one_hot = tf.cast(parts[1] == GENDER_CATEGORY, tf.float32)
    return tf.strings.to_number(parts[0]) / MAX_AGE, one_hot


def decode_img(path, aug=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    if aug:
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_saturation(img, 0.6, 1.6)
        img = tf.image.random_contrast(img, 0.6, 1.4)
        img = tf.image.random_flip_left_right(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2.0)
    return img


def add_batch(ds, size, training=False, cache=False):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if training:
        ds = ds.shuffle(2000)
    ds = ds.batch(size)
    if training:
        ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def get_model(input_shape):

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                  include_top=False,
                                                  weights='imagenet')

    model = GlobalAveragePooling2D()(base_model.output)
    model = Dense(EMBEDDING_SIZE,
                  use_bias=USE_BIAS)(model)
    model = BatchNormalization()(model)
    model = Activation("relu")(model)

    predictions_a = Dense(
        units=1,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="sigmoid",
        name='age')(model)
    predictions_g = Dense(
        units=2,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="softmax",
        name='gender')(model)

    model = Model(inputs=base_model.input, outputs=[predictions_a, predictions_g])
    model.summary()

    opt = Adam(lr=0.001)
    model.compile(
        optimizer=opt,
        loss={
            "age": "mse",
            "gender": "categorical_crossentropy",
        },
        loss_weights={
            "age": 1,
            "gender": 1,
        },
        metrics={
            'age': 'mse',
            'gender': 'acc'
        })

    return model


if __name__ == '__main__':
    main()
