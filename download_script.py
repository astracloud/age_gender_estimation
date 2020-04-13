import hashlib
import os
import pathlib
import shutil
import zipfile
from os import listdir

import requests

URL = 'https://genesis-ap-southeast-1-qa.s3-ap-southeast-1.amazonaws.com/cv_models/age_gender_v2.zip'
MD5_SUM = 'a47f63270e72c2d31bc97dcbc2550378'
NAME = 'age_gender'
VERSION = '2'


def check_md5(save_path):
    m = hashlib.md5()
    with open(save_path, "rb") as f:
        buf = f.read()
        m.update(buf)
    h = m.hexdigest()

    is_already_exist = MD5_SUM == h
    if is_already_exist:
        print("Model {} ver={} already exist.".format(NAME, VERSION))

    return MD5_SUM == h


def download_model(save_path):
    try:
        with open(save_path, "wb") as f:
            print("Downloading model: {}, ver={}".format(NAME, VERSION))
            response = requests.get(URL,
                                    stream=True,
                                    timeout=5)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                print('Error: file total length is 0')
            else:
                for data in response.iter_content(chunk_size=4096):
                    if data:
                        f.write(data)
                print('download complete!')
    except KeyboardInterrupt:
        print('\nstop download by KeyboardInterrupt... ')
        if os.path.exists(save_path):
            os.remove(save_path)
    except Exception:
        print("Error when downloading model, remove failed file ... Abort")
        if os.path.exists(save_path):
            os.remove(save_path)
        raise


def main():
    save_path = os.path.join('tmp', NAME, VERSION)
    zip_path = os.path.join(save_path, 'model.zip')
    model_path = os.path.join('saved_model', NAME, VERSION)
    if not os.path.isdir(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(model_path):
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    is_already_exist = False

    if os.path.exists(zip_path):
        is_already_exist = check_md5(zip_path)

    if not is_already_exist:
        download_model(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(save_path)

    for filename in listdir(os.path.join(save_path, NAME)):
        shutil.move(os.path.join(save_path, NAME, filename), os.path.join(model_path, filename))


if __name__ == '__main__':
    main()
