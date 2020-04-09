import timeit

import requests


def main():
    for _ in range(0, 10):
        start = timeit.default_timer()

        url = 'http://localhost:8888/age_gender'
        files = {'media': open('test_images/fb/1_female_and_3_male.jpg', 'rb')}
        response = requests.post(url, files=files)
        print(response.json())

        print(f'cost:{timeit.default_timer() - start}')


if __name__ == '__main__':
    main()
