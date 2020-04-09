import numpy as np

MAX_AGE = 70


def age_gender_preprocessing(img):
    img = img - 127.5
    img = img * 0.0078125
    img = img.astype(np.float32)

    return img


def age_gender_postprocessing(result):
    age, gender = result

    age = age[0] * (MAX_AGE / 2) + (MAX_AGE / 2)
    gender = np.argmax(gender)
    gender = 'f' if gender == 0 else 'm'

    return age, gender
