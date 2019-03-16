import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

root_data_folder = Path('D:/Programming/bsuir/sem4/MO/data')
small_data = root_data_folder / 'notMNIST_small'
large_data = root_data_folder / 'notMNIST_large'


def main():
    random.seed(42)
    one_hot_encoded_labels = generate_one_hot_encoded_class()
    # prepare_data(small_data)

    # task 1
    # show_rand_image_matrix(small_data, 5, 5, 3)

    # files = get_random_files(small_data, 1000)

    # task 2
    # print_classes_count(small_data)

    # task 3
    images, labels = get_unique_data(small_data)
    labels = encode_classes(labels)
    # labels = replace_classes_with_numbers(labels)
    tr_x, tr_y, te_x, te_y, v_x, v_y = split_data(images, labels, 0.7, 0.2, 0.1, 42)

    # task 4
    # get_unique_data(large_data)

    # task 5
    log_reg(tr_x, tr_y, te_x, te_y, v_x, v_y)

    pass


def log_reg(tr_x, tr_y, te_x, te_y, v_x, v_y):
    logreg = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs', max_iter=30)

    tr_x = tr_x.reshape(tr_x.shape[0], -1)
    te_x = te_x.reshape(te_x.shape[0], -1)
    v_x = v_x.reshape(v_x.shape[0], -1)

    logreg.fit(tr_x, tr_y)

    print(f'Logistic Regression TRAINING set accuracy: {logreg.score(tr_x, tr_y)}')
    print(f'Logistic Regression TEST set accuracy: {logreg.score(te_x, te_y)}')
    print(f'Logistic Regression VALIDATION set accuracy: {logreg.score(v_x, v_y)}')


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
            img = cv2.imread(file_name, 0)

            if img is not None:
                # norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                files_data.append(img)
                classes.append(class_name)

        files_data = np.array(files_data)
        classes = np.array(classes)

        np.savez(prepared_data_path, files_data=files_data, classes=classes)

    return files_data, classes


def generate_one_hot_encoded_class():
    a_char = 65
    classes = 10
    labels = {}

    for i, class_name in enumerate(range(a_char, a_char + classes)):
        z = np.zeros((1, classes), dtype=np.int8)
        z[0, i] = 1
        labels[chr(class_name)] = z

    return labels


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

        for i, (d, l) in enumerate(tqdm(zip(files_data, labels))):
            len_before = len(unique_data)
            unique_data.add(d.tobytes())
            len_after = len(unique_data)

            if len_after > len_before:
                unique_indexes.append(True)
            else:
                unique_indexes.append(False)

        files_data = files_data[unique_indexes]
        classes = labels[unique_indexes]

        np.savez(prepared_unique_data_path, files_data=files_data, classes=classes)

    return files_data, classes


def split_data(data, data_labels, train, test, validation, seed, percent=True):
    if seed is not None:
        np.random.seed(seed)

    if percent:
        data_len = len(data)
        train = int(data_len * train)
        test = int(data_len * test)
        validation = int(data_len - train - test)

    data, data_labels = data_shuffle(data, data_labels, seed)

    train_data = data[: train]
    train_data_labels = data_labels[: train]

    test_data = data[train: train + test]
    test_data_labels = data_labels[train: train + test]

    validation_data = data[train + test: train + test + validation]
    validation_data_labels = data_labels[train + test: train + test + validation]

    return train_data, train_data_labels, test_data, test_data_labels, validation_data, validation_data_labels


def replace_classes_with_numbers(data_labels):
    labels = list(set(data_labels))
    labels_dict = {l: i for (i, l) in enumerate(labels)}

    data_labels_numbers = []

    for l in data_labels:
        data_labels_numbers.append(labels_dict.get(l))

    return np.array(data_labels_numbers)


def encode_classes(data_labels, one_hot_dict):
    data_labels_numbers = []

    for l in data_labels:
        data_labels_numbers.append(one_hot_dict[l].flatten())

    return np.array(data_labels_numbers)


def data_shuffle(x, y, seed):
    if seed is not None:
        np.random.seed(seed)

    permutation = np.random.permutation(y.shape[0])
    x_shuffled = x[permutation, :]
    y_shuffled = y[permutation, :]

    return x_shuffled, y_shuffled


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
        (cols, rows, rand_files_shape[1], rand_files_shape[2]))
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

        img = cv2.imread(file_name, 0)

        if img is not None:
            img_scaled_2 = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            rand_files_names.add(file_name)
            rand_files.append(img_scaled_2)

    return np.array(rand_files)


if __name__ == '__main__':
    main()
