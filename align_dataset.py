import cv2
import numpy as np
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


def find_biggest_loc(face_locs):
    areas = [(end_x - start_x) * (end_y - start_y) for start_x, start_y, end_x, end_y in face_locs]

    return face_locs[np.argmax(areas)]


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

