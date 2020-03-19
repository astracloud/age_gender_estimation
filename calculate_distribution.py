import glob
import os
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument("path", help="path folder. example: "
                                 "\'/Volumes/Astra2TB/age_gender_estimation/faces_hawk/origin_train_set\'")

MAX_AGE = 70
FEMALE_INDEX = 0
MALE_INDEX = 1


def main():
    args = parser.parse_args()
    distribution = np.zeros((2, MAX_AGE+1), dtype=np.int)

    for jpg_path in glob.glob(os.path.join(args.path, '*.jpg')):
        file_name = jpg_path.split('/')[-1]
        age = int(file_name.split('_')[0])
        gender = file_name.split('_')[1]

        if gender == 'f':
            gender_index = FEMALE_INDEX
        elif gender == 'm':
            gender_index = MALE_INDEX
        else:
            raise RuntimeError(f'error: got gender string = {gender}')

        distribution[gender_index, age] += 1

    print(f'female count {sum(distribution[FEMALE_INDEX, :])}')
    print(f'male count {sum(distribution[MALE_INDEX, :])}')

    print('female age distribution:')
    for idx, age_count in enumerate(distribution[FEMALE_INDEX, :]):
        print(f'{idx} : {age_count}')

    print('male age distribution:')
    for idx, age_count in enumerate(distribution[MALE_INDEX, :]):
        print(f'{idx} : {age_count}')


if __name__ == '__main__':
    main()
