import cv2
import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def request_server(img_resized, server_url):
    # Request.
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "age_gender"
    request.model_spec.signature_name = "serving_default"
    request.inputs["input_1"].CopyFrom(
        tf.make_tensor_proto(img_resized, shape=[1, ] + list(img_resized.shape)))
    response = stub.Predict(request, 5.0)  # 5 secs timeout
    return np.asarray(response.outputs["age"].float_val), np.asarray(response.outputs["gender"].float_val)


def main():
    img = cv2.imread('test_images/9_m_s3fb100019682865086.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img - 127.5
    img = img * 0.0078125
    img = img.astype(np.float32)
    server_url = 'localhost:8500'
    age, gender = request_server(img, server_url)
    print(age, gender)


if __name__ == '__main__':
    main()
