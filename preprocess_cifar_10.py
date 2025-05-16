import os
import cv2
import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

def cifar10_img(file_dir):
    for i in range(1, 6):
        data_name = file_dir + '/' + 'data_batch_' + str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')
 
        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            label = data_dict[b'labels'][j]
            
            class_folder = os.path.join(loc_1, str(label) + '_' + cifar10_classes[label])
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            
            img_name = os.path.join(class_folder, f"img_{(i)*10000+j}.jpg")
            print(img_name)
            cv2.imwrite(img_name, img)
 
        print(data_name + ' is done')
 
    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)
 
    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = test_dict[b'labels'][m]
        
        class_folder = os.path.join(loc_2, str(label) + '_' + cifar10_classes[label])
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        
        img_name = os.path.join(class_folder, f"img_{m}.jpg")
        cv2.imwrite(img_name, img)
        
    print(test_data_name + ' is done')
    print('Finish transforming to image')
 
if __name__ == '__main__':
    loc_1 = './autodl-tmp/data/cifar10/train/'
    loc_2 = './autodl-tmp/data/cifar10/test/'
    if os.path.exists(loc_1) == False:
        os.makedirs(loc_1)
    if os.path.exists(loc_2) == False:
        os.makedirs(loc_2)
    file_dir = './autodl-tmp/data/cifar-10-batches-py'
    cifar10_img(file_dir)