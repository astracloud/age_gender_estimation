from generator.data_gen import DataGenerator
from train_tf2 import get_model, add_batch
import tensorflow as tf
import cv2
import numpy as np

from web_utils import age_gender_preprocessing

VALID_IMAGE_PATH = 'images/afad_align_set/*.jpg'
VALID_DETAIL_DF = 'train_set_label.csv'
AUTOTUNE = tf.data.experimental.AUTOTUNE

H5_PATH = 'checkpoints/203-1.32.h5'
IMG_SIZE = 128
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

BATCH_SIZE = 64


def single():
    model = get_model(INPUT_SHAPE)
    model.load_weights(H5_PATH, by_name=True)

    img = cv2.imread('images/afad_align_set/2_f_0s3fb-0-1c182fb5-05cb-43bc-847c-1ebc72688996.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = age_gender_preprocessing(img)

    result = model.predict(np.expand_dims(img, axis=0))

    print(f'age:{result[0][0] * 35 + 35}')
    print(f'female:{result[1][0]}')
    print(f'Beard:{result[2][0]}')
    print(f'Eyeglasses:{result[3][0]}')
    print(f'EyesOpen:{result[4][0]}')
    print(f'MouthOpen:{result[5][0]}')
    print(f'Mustache:{result[6][0]}')
    print(f'Sunglasses:{result[7][0]}')
    print(f'expression:{np.argmax(result[8][0])}')


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
    # main()
    single()
