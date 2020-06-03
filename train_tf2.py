import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from generator.data_gen import DataGenerator

TRAIN_IMAGE_PATH = 'images/train_set/*.jpg'
VALID_IMAGE_PATH = 'images/afad_align_set/*.jpg'
TRAIN_DETAIL_DF = 'train_set_label.csv'
VALID_DETAIL_DF = 'afad_label.csv'
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_SIZE = 128
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
USE_BIAS = False
WEIGHT_INIT = "he_normal"

EMBEDDING_SIZE = 128
WEIGHT_DECAY = 0.01
EPOCHS = 2000
BATCH_SIZE = 64


def main():
    model = get_model(INPUT_SHAPE)

    train_gen = DataGenerator()
    train_gen.parse(TRAIN_IMAGE_PATH, TRAIN_DETAIL_DF)
    print(f'{train_gen.sample_count()} of images have to train.')

    train_main_ds = tf.data.Dataset.from_generator(
        lambda: train_gen.generate(aug=True),
        (tf.float32,  # input
         (tf.float32, tf.float32,  # age, gender
          tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,  # 6 status
          tf.float32)),  # 8 emo
        (tf.TensorShape([IMG_SIZE, IMG_SIZE, 3]),
         (tf.TensorShape([]), tf.TensorShape([2]),
          tf.TensorShape([2]), tf.TensorShape([2]), tf.TensorShape([2]),
          tf.TensorShape([2]), tf.TensorShape([2]), tf.TensorShape([2]),
          tf.TensorShape([8]))))
    train_main_ds = add_batch(train_main_ds, BATCH_SIZE, training=True)

    # iter_d = iter(train_main_ds)
    # print(next(iter_d))

    valid_gen = DataGenerator()
    valid_gen.parse(VALID_IMAGE_PATH, VALID_DETAIL_DF)
    print(f'{valid_gen.sample_count()} of images have to valid.')

    valid_main_ds = tf.data.Dataset.from_generator(
        lambda: valid_gen.generate(aug=False),
        (tf.float32,  # input
         (tf.float32, tf.float32,  # age, gender
          tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,  # 6 status
          tf.float32)),  # 8 emo
        (tf.TensorShape([IMG_SIZE, IMG_SIZE, 3]),
         (tf.TensorShape([]), tf.TensorShape([2]),
          tf.TensorShape([2]), tf.TensorShape([2]), tf.TensorShape([2]),
          tf.TensorShape([2]), tf.TensorShape([2]), tf.TensorShape([2]),
          tf.TensorShape([8]))))
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
        steps_per_epoch=train_gen.sample_count() // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_main_ds,
        validation_steps=valid_gen.sample_count() // BATCH_SIZE)


def add_batch(ds, size, training=False, cache=False):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
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
                  kernel_initializer=WEIGHT_INIT,
                  use_bias=USE_BIAS)(model)
    model = BatchNormalization()(model)
    model = Activation("relu")(model)

    p_age = Dense(
        units=1,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="tanh",
        name='age')(model)
    p_gender = Dense(
        units=2,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="sigmoid",
        name='gender')(model)
    p_beard = Dense(
        units=2,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="sigmoid",
        name='beard')(model)
    p_eyes_glasses = Dense(
        units=2,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="sigmoid",
        name='eyes_glasses')(model)
    p_eyes_open = Dense(
        units=2,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="sigmoid",
        name='eyes_open')(model)
    p_mouth_open = Dense(
        units=2,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="sigmoid",
        name='mouth_open')(model)
    p_mustache = Dense(
        units=2,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="sigmoid",
        name='mustache')(model)
    p_sunglasses = Dense(
        units=2,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="sigmoid",
        name='sunglasses')(model)
    p_expression = Dense(
        units=8,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        activation="softmax",
        name='expression')(model)

    model = Model(inputs=base_model.input, outputs=[p_age, p_gender, p_beard, p_eyes_glasses, p_eyes_open,
                                                    p_mouth_open, p_mustache, p_sunglasses, p_expression])
    model.summary()

    opt = Adam(lr=0.001)
    model.compile(
        optimizer=opt,
        loss={
            "age": "mae",
            "gender": "binary_crossentropy",
            "beard": "binary_crossentropy",
            "eyes_glasses": "binary_crossentropy",
            "eyes_open": "binary_crossentropy",
            "mouth_open": "binary_crossentropy",
            "mustache": "binary_crossentropy",
            "sunglasses": "binary_crossentropy",
            "expression": "categorical_crossentropy",
        },
        loss_weights={
            "age": 1,
            "gender": 1,
            "beard": 0.5,
            "eyes_glasses": 0.5,
            "eyes_open": 0.5,
            "mouth_open": 0.5,
            "mustache": 0.5,
            "sunglasses": 0.5,
            "expression": 0.5,
        },
        metrics={
            'age': 'mae',
            'gender': 'acc',
            "beard": 'acc',
            "eyes_glasses": 'acc',
            "eyes_open": 'acc',
            "mouth_open": 'acc',
            "mustache": 'acc',
            "sunglasses": 'acc',
            "expression": 'acc'
        })

    return model


if __name__ == '__main__':
    main()
