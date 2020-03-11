import glob
import os

import cv2

SOURCE_FOLDER = 'faces_hawk/origin_train_set'
TARGET_FOLDER = 'faces_hawk/train_set'
TARGET_SIZE = (128, 128)


def main():
    if not os.path.isdir(TARGET_FOLDER):
        os.mkdir(TARGET_FOLDER)

    for path in glob.glob(os.path.join(SOURCE_FOLDER, '*.jpg')):
        name = path.split('/')[-1]
        img = cv2.imread(path)
        img = cv2.resize(img, TARGET_SIZE)
        cv2.imwrite(os.path.join(TARGET_FOLDER, name), img)


if __name__ == '__main__':
    main()
