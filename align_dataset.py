import glob
import os

import cv2
import numpy as np
from cv_core.detector.face_inference import FaceLandmarksDetector
from skimage import transform as trans

SOURCE_FOLDER_NAME = 'origin_train_set'
SIZE = 128

# 極左臉，只有一個眼睛
left_profile_landmarks = np.array([
    [44, 32],
    [86, 32],
    [41, 64],
    [38, 86],
    [64, 86]], dtype=np.float32)

# 左臉，兩個眼睛，但左邊的鼻子半邊被遮住
left_landmarks = np.array([
    [40, 32],
    [86, 32],
    [56, 72],
    [46, 92],
    [76, 92]], dtype=np.float32)

front_landmarks = np.array([
    [39, 45],
    [89, 45],
    [64, 74],
    [41, 94],
    [87, 94]], dtype=np.float32)

# 右臉，兩個眼睛，但右邊的鼻子半邊被遮住
right_landmarks = np.array([
    [42, 32],
    [88, 32],
    [76, 72],
    [52, 92],
    [80, 92]], dtype=np.float32)

# 極右臉，只有一個眼睛
right_profile_landmarks = np.array([
    [42, 32],
    [84, 32],
    [87, 64],
    [64, 86],
    [90, 86]], dtype=np.float32)

landmark_src = np.array([left_profile_landmarks, left_landmarks,
                         front_landmarks, right_landmarks, right_profile_landmarks])


def main():
    target_folder_path = os.path.join('faces_hawk', SOURCE_FOLDER_NAME)
    save_folder_path = os.path.join('faces_hawk', 'align_set')
    landmarks_detector = FaceLandmarksDetector()

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    for idx, image_path in enumerate(glob.glob(os.path.join(target_folder_path, f'*.jpg'))):
        file_name = image_path.split('/')[-1]

        img = cv2.imread(image_path)
        h, w, _ = img.shape
        if w < 100 or h < 100:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))

        landmarks = landmarks_detector.predict(img)
        left_eye_ball = ((landmarks['left_eye'][0][0] + landmarks['left_eye'][3][0]) / 2,
                         (landmarks['left_eye'][0][1] + landmarks['left_eye'][3][1]) / 2,)
        right_eye_ball = ((landmarks['right_eye'][0][0] + landmarks['right_eye'][3][0]) / 2,
                          (landmarks['right_eye'][0][1] + landmarks['right_eye'][3][1]) / 2)
        landmark = np.asarray([left_eye_ball, right_eye_ball,
                               landmarks['nose_bridge'][3], landmarks['top_lip'][0], landmarks['top_lip'][6]])

        img, _ = align(img, landmark, SIZE)

        # landmarks = landmarks_detector.predict(img)
        # left_eye_ball = ((landmarks['left_eye'][0][0] + landmarks['left_eye'][3][0]) / 2,
        #                  (landmarks['left_eye'][0][1] + landmarks['left_eye'][3][1]) / 2,)
        # right_eye_ball = ((landmarks['right_eye'][0][0] + landmarks['right_eye'][3][0]) / 2,
        #                   (landmarks['right_eye'][0][1] + landmarks['right_eye'][3][1]) / 2)
        # landmark = np.asarray([left_eye_ball, right_eye_ball,
        #                        landmarks['nose_bridge'][3], landmarks['top_lip'][0], landmarks['top_lip'][6]])
        # print(landmark)
        # cv2.circle(img, (int(left_eye_ball[0]), int(left_eye_ball[1])), 3, (255, 0, 0), -1)
        # cv2.circle(img, (int(right_eye_ball[0]), int(right_eye_ball[1])), 3, (255, 0, 0), -1)
        # cv2.circle(img, (int(landmarks['nose_bridge'][3][0]), int(landmarks['nose_bridge'][3][1])), 3, (255, 0, 0), -1)
        # cv2.circle(img, (int(landmarks['top_lip'][0][0]), int(landmarks['top_lip'][0][1])), 3, (255, 0, 0), -1)
        # cv2.circle(img, (int(landmarks['top_lip'][6][0]), int(landmarks['top_lip'][6][1])), 3, (255, 0, 0), -1)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_folder_path, f'{file_name}'), img)

        if idx % 100 == 0:
            print(f'finish {idx}.')


def estimate_norm(lmk):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    for i in range(len(landmark_src)):
        tform.estimate(lmk, landmark_src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - landmark_src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

    return min_M, min_index


def align(img, landmark, image_size):
    M, pose_index = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, pose_index


if __name__ == '__main__':
    main()
