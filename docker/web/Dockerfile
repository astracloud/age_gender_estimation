FROM python:3.6-slim

RUN apt update && apt upgrade -y
RUN apt install build-essential cmake pkg-config -y
RUN apt install libsm-dev libxrender1 libxext-dev libgif-dev -y

RUN pip install opencv-python tornado numpy scikit-image tensorflow-serving-api dlib face_recognition

COPY web_server.py /app/web_server.py
COPY web_utils.py /app/web_utils.py
COPY align_dataset.py /app/align_dataset.py
COPY inference_grpc.py /app/inference_grpc.py

EXPOSE  8888
