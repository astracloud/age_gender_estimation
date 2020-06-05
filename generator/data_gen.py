import glob
import pandas as pd
import sklearn
import tensorflow as tf
import numpy as np

GENDER_CATEGORY = np.array(['f', 'm'])
MAX_AGE = 70


class DataGenerator:
    def __init__(self):
        self.img_file_path = []
        self.face_detail_info = None

    def parse(self, img_folder, csv_path):
        self.img_file_path = glob.glob(img_folder)
        df = pd.read_csv(csv_path)
        self.face_detail_info = {}
        for index, row in df.iterrows():
            self.face_detail_info[row['Filename']] = [row['Beard'],
                                                      row['Eyeglasses'],
                                                      row['EyesOpen'],
                                                      row['MouthOpen'],
                                                      row['Mustache'],
                                                      row['Sunglasses'],
                                                      row['FEAR'],
                                                      row['DISGUSTED'],
                                                      row['CONFUSED'],
                                                      row['SAD'],
                                                      row['CALM'],
                                                      row['ANGRY'],
                                                      row['HAPPY'],
                                                      row['SURPRISED'],
                                                      ]

    def sample_count(self):
        return len(self.img_file_path)

    def generate(self, aug=True):

        while True:
            if aug:
                self.img_file_path = sklearn.utils.shuffle(self.img_file_path)

            for path in self.img_file_path:
                img = _decode_img(path, aug=aug)
                age, gender = _get_label(path)

                info = self._df_info(path)

                yield img, (age, gender, info[0], info[1], info[2], info[3], info[4], info[5], info[6])

    def _df_info(self, path):
        fn = path.split('/')[-1]
        val = self.face_detail_info[fn][0]
        beard = tf.cast(np.array([val, 1 - val]), tf.float32)
        val = self.face_detail_info[fn][1]
        eyeglasses = tf.cast(np.array([val, 1 - val]), tf.float32)
        val = self.face_detail_info[fn][2]
        eyes_open = tf.cast(np.array([val, 1 - val]), tf.float32)
        val = self.face_detail_info[fn][3]
        mouth_open = tf.cast(np.array([val, 1 - val]), tf.float32)
        val = self.face_detail_info[fn][4]
        mustache = tf.cast(np.array([val, 1 - val]), tf.float32)
        val = self.face_detail_info[fn][5]
        sunglasses = tf.cast(np.array([val, 1 - val]), tf.float32)
        emo = tf.cast(np.array([self.face_detail_info[fn][6], self.face_detail_info[fn][7],
                                self.face_detail_info[fn][8], self.face_detail_info[fn][9],
                                self.face_detail_info[fn][10], self.face_detail_info[fn][11],
                                self.face_detail_info[fn][12], self.face_detail_info[fn][13]]), tf.float32)
        return beard, eyeglasses, eyes_open, mouth_open, mustache, sunglasses, emo


def _get_label(file_path):
    file_name = tf.strings.split(file_path, '/')[-1]
    parts = tf.strings.split(file_name, '_')
    one_hot = tf.cast(parts[1] == GENDER_CATEGORY, tf.float32)
    return (tf.strings.to_number(parts[0]) - (MAX_AGE / 2)) / (MAX_AGE / 2), one_hot


def _decode_img(path, aug=False):
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
