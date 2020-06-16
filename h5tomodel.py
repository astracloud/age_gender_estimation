from train_tf2 import get_model

VERSION = 2
IMG_SIZE = 128
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
H5_PATH = 'checkpoints/102-3.93.h5'


def main():
    model = get_model(INPUT_SHAPE)

    model.load_weights(H5_PATH, by_name=True)

    model.save(f'saved_model/face_detail/{VERSION}/', include_optimizer=False)


if __name__ == '__main__':
    main()
