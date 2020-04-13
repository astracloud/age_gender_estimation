import glob
import timeit

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from cv_core.detector.age_gender_classifier import KerasAgeGenderClassifier
from cv_core.detector.face_inference import FaceVectorEncoder, FaceLandmarksDetector
from tensorflow.keras.models import load_model
import numpy as np

from align_dataset import align

model_name = 'checkpoints/296-0.19.h5'
img_path = 'test_images/AFAD-Lite/35/111/*.jpg'
model = load_model(model_name)
# model = tf.keras.models.load_model('saved_model/age_gender')

IMG_SIZE = 128
max_age = 70
real_age = 35
TO_ALIGN = False


def main():
    ages_jason = []
    ages_paul = []
    gender_jason = [0, 0]
    gender_paul = [0, 0]
    keras_classifier = KerasAgeGenderClassifier()
    face_encoder = FaceVectorEncoder()
    landmarks_detector = FaceLandmarksDetector()

    idx=0
    for image_path in glob.glob(img_path):
        img = cv2.imread(image_path)
        h, w, _ = img.shape

        if w < 100 or h < 100:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start = timeit.default_timer()
        vector = face_encoder.predict(img)
        _, paul_age_pred, paul_gender_pred = keras_classifier.predict(vector)
        if idx == 30:
            print(f'Paul Time cost {(timeit.default_timer() - start):.2f}s')

        if paul_gender_pred == 'f':
            gender_paul[0] += 1
        else:
            gender_paul[1] += 1
        ages_paul.append(paul_age_pred)

        start = timeit.default_timer()
        if TO_ALIGN:
            landmarks = landmarks_detector.predict(img)
            left_eye_ball = ((landmarks['left_eye'][0][0] + landmarks['left_eye'][3][0]) / 2,
                             (landmarks['left_eye'][0][1] + landmarks['left_eye'][3][1]) / 2,)
            right_eye_ball = ((landmarks['right_eye'][0][0] + landmarks['right_eye'][3][0]) / 2,
                              (landmarks['right_eye'][0][1] + landmarks['right_eye'][3][1]) / 2)
            landmark = np.asarray([left_eye_ball, right_eye_ball,
                                   landmarks['nose_bridge'][3], landmarks['top_lip'][0], landmarks['top_lip'][6]])

            img, _ = align(img, landmark, IMG_SIZE)

        img = img - 127.5
        img = img * 0.0078125

        jason_age_pred, jason_gender_pred = model.predict(np.expand_dims(img, axis=0))
        if idx == 30:
            print(f'Jason Time cost {(timeit.default_timer() - start):.2}s')

        ages_jason.append((jason_age_pred[0] * max_age/2 + max_age/2)[0])
        if np.argmax(jason_gender_pred[0]) == 0:
            gender_jason[0] += 1
        else:
            gender_jason[1] += 1

        idx+=1

    print(f'paul: {gender_paul}')
    print(f'jason: {gender_jason}')
    sns.kdeplot(ages_jason, color='g', shade=True, label='New model')
    sns.kdeplot(ages_paul, color='b', shade=True, label='Old model')
    plt.axvline(real_age, color='r')
    plt.xlabel('age')
    plt.ylabel('propability')
    plt.show()


if __name__ == "__main__":
    main()
