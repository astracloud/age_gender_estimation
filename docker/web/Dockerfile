FROM python:3.6-slim

RUN apt update && apt upgrade -y
RUN apt install build-essential cmake pkg-config -y
RUN apt install libsm-dev libxrender1 libxext-dev libgif-dev -y

RUN pip install opencv-python tornado numpy scikit-image tensorflow-serving-api dlib face_recognition

EXPOSE  8888
