from generator.data_gen import DataGenerator
from train_tf2 import get_model, add_batch
import tensorflow as tf

VALID_IMAGE_PATH = 'images/afad_align_set/*.jpg'
VALID_DETAIL_DF = 'train_set_label.csv'
AUTOTUNE = tf.data.experimental.AUTOTUNE

H5_PATH = 'checkpoints/124-1.40.h5'
IMG_SIZE = 128
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

BATCH_SIZE = 64


def main():
    model = get_model(INPUT_SHAPE)
    model.load_weights(H5_PATH, by_name=True)

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

    model.evaluate(
        valid_main_ds,
        steps=valid_gen.sample_count() // BATCH_SIZE, )


if __name__ == '__main__':
    main()
