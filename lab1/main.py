from pathlib import Path
import cv2
import numpy as np
import random


def main():
    root_data_folder = Path('D:/Programming/bsuir/sem4/MO/data')
    small_data = root_data_folder / 'notMNIST_small'
    large_data = root_data_folder / 'notMNIST_large'

    show_rand_image_matrix(small_data)

    # img = cv2.imread(str(small_data / 'A' / 'MDEtMDEtMDAudHRm.png'), 0)
    # img1 = cv2.imread(str(small_data / 'A' / 'MDRiXzA4LnR0Zg==.png'), 0)
    # img2 = cv2.imread(str(small_data / 'A' / 'MlRvb24gU2hhZG93LnR0Zg==.png'), 0)
    # img3 = cv2.imread(str(small_data / 'A' / 'Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png'), 0)
    #
    # big_image = cv2.resize(img, (0, 0), fx=2, fy=2)
    # big_image1 = cv2.resize(img1, (0, 0), fx=2, fy=2)
    # big_image2 = cv2.resize(img2, (0, 0), fx=2, fy=2)
    # big_image3 = cv2.resize(img3, (0, 0), fx=2, fy=2)
    #
    # row1 = np.concatenate((big_image, big_image1), axis=1)
    # row2 = np.concatenate((big_image2, big_image3), axis=1)
    # image_matrix = np.concatenate((row1, row2), axis=0)
    # cv2.imshow('image', image_matrix)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    prepare_images(small_data)
    pass


def show_rand_image_matrix(image_path):
    rows = 4
    cols = 4

    rand_files = get_random_files(image_path, rows * cols)
    rand_files_shape = rand_files.shape

    image_matrix = np.concatenate(rand_files.reshape((rows, cols, rand_files_shape[1], rand_files_shape[2])), axis=2)
    image_matrix = np.concatenate(image_matrix, axis=0)

    cv2.imshow('image', image_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def get_random_files(root_dir, files_number):
    file_list = list(root_dir.glob('**/*.png'))

    rand_files_names = set()
    rand_files = []

    while len(rand_files) < files_number:
        rand = random.randint(0, len(file_list) - 1)
        file_name = str(file_list[rand])

        if file_name in rand_files_names:
            continue

        img = cv2.imread(file_name, 0)
        img_scaled_2 = cv2.resize(img, (0, 0), fx=3, fy=3)
        rand_files_names.add(file_name)
        rand_files.append(img_scaled_2)

    return np.array(rand_files)


def prepare_images(data_path):
    pass


def show_images():
    pass


if __name__ == '__main__':
    main()
