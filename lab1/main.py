from pathlib import Path
import cv2
import numpy as np
import random


def main():
    root_data_folder = Path('D:/Programming/bsuir/sem4/MO/data')
    small_data = root_data_folder / 'notMNIST_small'
    large_data = root_data_folder / 'notMNIST_large'

    # task 1
    # show_rand_image_matrix(small_data, 5, 5, 3)

    # task 2
    print_classes_count(small_data)
    pass


def print_classes_count(root_dir):
    classes = list(root_dir.glob('*/'))

    class_count = {}

    for cl in classes:
        files_in_class = list(cl.glob('*.png'))
        class_name = str(cl).rsplit('\\', 1)[-1]
        class_count[class_name] = len(files_in_class)

    max_file_count = class_count[max(class_count, key=class_count.get)]

    for cl, cnt in class_count.items():
        print(f'{cl}: {cnt} files. {cnt / max_file_count}')
    pass


def show_rand_image_matrix(root_dir, rows, cols, scale):
    rand_files = get_random_files(root_dir, rows * cols, scale)
    rand_files_shape = rand_files.shape

    image_matrix = rand_files.reshape((cols, rows, rand_files_shape[1], rand_files_shape[2]))
    image_matrix = np.concatenate(image_matrix, axis=2)
    image_matrix = np.concatenate(image_matrix, axis=0)

    cv2.imshow('image', image_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_random_files(root_dir, files_number, scale):
    file_list = list(root_dir.glob('**/*.png'))

    rand_files_names = set()
    rand_files = []

    while len(rand_files) < files_number:
        rand = random.randint(0, len(file_list) - 1)
        file_name = str(file_list[rand])

        if file_name in rand_files_names:
            continue

        img = cv2.imread(file_name, 0)
        img_scaled_2 = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        rand_files_names.add(file_name)
        rand_files.append(img_scaled_2)

    return np.array(rand_files)


if __name__ == '__main__':
    main()
