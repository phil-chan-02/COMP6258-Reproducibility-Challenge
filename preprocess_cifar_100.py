import os
import cv2
import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar100_img(file_dir, loc_train, loc_test):
    # 加载meta文件获取类别名称
    meta = unpickle(os.path.join(file_dir, 'meta'))
    fine_label_names = [label.decode('utf-8') for label in meta[b'fine_label_names']]

    # 加载训练数据
    train_dict = unpickle(os.path.join(file_dir, 'train'))
    print('train is processing')

    for i in range(len(train_dict[b'data'])):
        img = np.reshape(train_dict[b'data'][i], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = train_dict[b'fine_labels'][i]
        class_folder = os.path.join(loc_train, f"{label}_{fine_label_names[label]}")
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        img_name = os.path.join(class_folder, f"img_{i}.jpg")
        cv2.imwrite(img_name, img)
    print('train is done')

    # 加载测试数据
    test_dict = unpickle(os.path.join(file_dir, 'test'))
    print('test is processing')

    for i in range(len(test_dict[b'data'])):
        img = np.reshape(test_dict[b'data'][i], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = test_dict[b'fine_labels'][i]
        class_folder = os.path.join(loc_test, f"{label}_{fine_label_names[label]}")
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        img_name = os.path.join(class_folder, f"img_{i}.jpg")
        cv2.imwrite(img_name, img)
    print('test is done')
    print('Finish transforming to image')

if __name__ == '__main__':
    # 输出目录
    loc_train = './autodl-tmp/data/cifar100/train/'
    loc_test = './autodl-tmp/data/cifar100/test/'

    # 创建目录
    os.makedirs(loc_train, exist_ok=True)
    os.makedirs(loc_test, exist_ok=True)

    # CIFAR-100 路径
    file_dir = './autodl-tmp/data/cifar-100-python'
    cifar100_img(file_dir, loc_train, loc_test)
