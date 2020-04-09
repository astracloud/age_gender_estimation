from typing import Optional, Awaitable

import cv2
import face_recognition as fr
import numpy as np
import tornado.ioloop
import tornado.web

from align_dataset import align
from inference_grpc import request_server
from web_utils import age_gender_preprocessing, age_gender_postprocessing

OFFSET = 20
IMAGE_SIZE = 128
HOST_PORT = 8888
SERVING_URL = 'serving:8500'


class MainHandler(tornado.web.RequestHandler):
    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def get(self):
        self.write("Hello, world")


class AgeGenderHandler(tornado.web.RequestHandler):
    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def post(self):
        file_body = self.request.files['media'][0]['body']
        img_encode = np.fromstring(file_body, np.uint8)
        img = cv2.imdecode(img_encode, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locs = fr.face_locations(img)
        face_locs = [(loc[3], loc[0], loc[1], loc[2]) for loc in face_locs]

        ages = []
        genders = []
        face_crops = []

        for face_loc in face_locs:
            start_x, start_y, end_x, end_y = face_loc

            face_img = img[start_y:end_y, start_x:end_x, :]

            h, w, _ = face_img.shape
            landmarks = fr.api.face_landmarks(face_img, [(0, w, h, 0)])[0]
            left_eye_ball = (
                (landmarks['left_eye'][0][0] + landmarks['left_eye'][3][0]) / 2,
                (landmarks['left_eye'][0][1] + landmarks['left_eye'][3][1]) / 2,
            )
            right_eye_ball = (
                (landmarks['right_eye'][0][0] + landmarks['right_eye'][3][0]) / 2,
                (landmarks['right_eye'][0][1] + landmarks['right_eye'][3][1]) / 2)
            landmarks = np.asarray([
                left_eye_ball, right_eye_ball, landmarks['nose_bridge'][3],
                landmarks['top_lip'][0], landmarks['top_lip'][6]
            ])
            landmarks = np.asarray([(landmark[0] + start_x,
                                     landmark[1] + start_y)
                                    for landmark in landmarks])

            align_face_img, _ = align(img, landmarks, IMAGE_SIZE)

            align_face_img = age_gender_preprocessing(align_face_img)
            result = request_server(align_face_img, SERVING_URL)
            age, gender = age_gender_postprocessing(result)

            ages.append(age)
            genders.append(gender)
            face_crops.append([start_x, start_y, end_x, end_y])
        self.finish({'ages': ages, 'genders': genders, 'face_crops': face_crops})


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/age_gender", AgeGenderHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(HOST_PORT)
    tornado.ioloop.IOLoop.current().start()
