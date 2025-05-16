import os
import cv2
import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# CIFAR-10的类别名称
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# 训练集有五个批次，每个批次10000个图片，测试集有10000张图片
def cifar10_img(file_dir):
    # 处理训练数据
    for i in range(1, 6):
        data_name = file_dir + '/' + 'data_batch_' + str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')
 
        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
            img = np.transpose(img, (1, 2, 0))
            # 通道顺序为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 获取当前图像的类别标签
            label = data_dict[b'labels'][j]
            
            # 创建每个类别的子文件夹
            class_folder = os.path.join(loc_1, str(label) + '_' + cifar10_classes[label])
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            
            # 保存图像到对应类别的文件夹中
            img_name = os.path.join(class_folder, f"img_{(i)*10000+j}.jpg")
            print(img_name)
            cv2.imwrite(img_name, img)
 
        print(data_name + ' is done')
 
    # 处理测试数据
    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)
 
    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        # 通道顺序为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取当前图像的类别标签
        label = test_dict[b'labels'][m]
        
        # 创建每个类别的子文件夹
        class_folder = os.path.join(loc_2, str(label) + '_' + cifar10_classes[label])
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        
        # 保存图像到对应类别的文件夹中
        img_name = os.path.join(class_folder, f"img_{m}.jpg")
        cv2.imwrite(img_name, img)
        
    print(test_data_name + ' is done')
    print('Finish transforming to image')
 
if __name__ == '__main__':
    loc_1 = './autodl-tmp/data/cifar10/train/'
    loc_2 = './autodl-tmp/data/cifar10/test/'
    # 判断文件夹是否存在，不存在的话创建文件夹
    if os.path.exists(loc_1) == False:
        os.makedirs(loc_1)
    if os.path.exists(loc_2) == False:
        os.makedirs(loc_2)
    file_dir = './autodl-tmp/data/cifar-10-batches-py'
    cifar10_img(file_dir)