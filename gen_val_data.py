import glob
import random
import shutil


def main():
    gender_list = ['f', 'm']
    age_list = range(1, 71)

    for gender in gender_list:
        for age in age_list:
            all_paths = glob.glob(
                f'/Volumes/Astra2TB/age_gender_estimation/faces_hawk/origin_train_set/{age}_{gender}*')
            val_count = round(len(all_paths) / 5.0)
            val_list = random.sample(all_paths, val_count)
            for val in val_list:
                filename = val.split('/')[-1]
                shutil.move(val, f"/Volumes/Astra2TB/age_gender_estimation/faces_hawk/origin_val_set/{filename}")


if __name__ == '__main__':
    main()
