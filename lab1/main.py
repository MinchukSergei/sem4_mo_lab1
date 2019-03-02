from pathlib import Path
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import sys


def main():
    np.set_printoptions(threshold=sys.maxsize)

    root_data_folder = Path('D:/Programming/bsuir/sem4/MO/data')
    small_data = root_data_folder / 'notMNIST_small'
    large_data = root_data_folder / 'notMNIST_large'

    get_unique_data(large_data)
    # prepare_data(small_data)

    # task 1
    # show_rand_image_matrix(large_data, 5, 5, 3)

    # files = get_random_files(small_data, 1000)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # task 2
    # print_classes_count(large_data)

    # task 3

    pass


def prepare_data(root_dir):
    files = list(root_dir.glob('**/*.png'))

    prepared_data_path = Path(str(root_dir) + '_prepared.npz')

    if prepared_data_path.exists():
        prepared_data = np.load(prepared_data_path)

        files_data = prepared_data['files_data']
        classes = prepared_data['classes']
    else:
        files_data = []
        classes = []

        for f in tqdm(files):
            file_name = str(f)
            class_name = file_name.rsplit('\\', 2)[-2]
            img = cv2.imread(file_name, 1)

            if img is not None:
                files_data.append(img)
                classes.append(class_name)

        files_data = np.array(files_data)
        classes = np.array(classes)

        np.savez(prepared_data_path, files_data=files_data, classes=classes)

    return files_data, classes


def get_unique_data(root_dir):
    prepared_unique_data_path = Path(str(root_dir) + '_prepared_unique.npz')

    if prepared_unique_data_path.exists():
        prepared_data = np.load(prepared_unique_data_path)

        files_data = prepared_data['files_data']
        classes = prepared_data['classes']
    else:
        files_data, labels = prepare_data(root_dir)
        unique_data = set()
        unique_indexes = []

        st = time.time()
        for i, (d, l) in enumerate(tqdm(zip(files_data, labels))):
            len_before = len(unique_data)
            unique_data.add(d.tobytes())
            len_after = len(unique_data)

            if len_after > len_before:
                unique_indexes.append(True)
            else:
                unique_indexes.append(False)
        print(time.time() - st)

        print(len(files_data))
        files_data = files_data[unique_indexes]
        classes = labels[unique_indexes]
        print(len(files_data))

        np.savez(prepared_unique_data_path, files_data=files_data, classes=classes)

    return files_data, classes


def split_data(data, data_labels, train, test, validation, seed, percent=True):
    random.seed(seed)

    if percent:
        train = len(data) * train
        test = len(data) * test
        validation = len(data) - train - test

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

    rand_files_reshaped = rand_files.reshape(
        (cols, rows, rand_files_shape[1], rand_files_shape[2], rand_files_shape[3]))
    image_matrix = np.concatenate(rand_files_reshaped, axis=2)
    image_matrix = np.concatenate(image_matrix, axis=0)

    cv2.imshow('image', image_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_random_files(root_dir, files_number, scale=1):
    file_list = list(root_dir.glob('**/*.png'))

    rand_files_names = set()
    rand_files = []

    while len(rand_files) < files_number:
        rand = random.randint(0, len(file_list) - 1)
        file_name = str(file_list[rand])

        if file_name in rand_files_names:
            continue

        img = cv2.imread(file_name, 1)
        img_scaled_2 = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        rand_files_names.add(file_name)
        rand_files.append(img_scaled_2)

    return np.array(rand_files)


if __name__ == '__main__':
    main()
