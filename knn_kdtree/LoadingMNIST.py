#encoding:utf-8
import numpy as np
import struct
#import matplotlib.pyplot as plt

train_images_path = 'D:/study/大三上/机器学习/matlab project/python/train-images.idx3-ubyte'
train_labels_path = 'D:/study/大三上/机器学习/matlab project/python/train-labels.idx1-ubyte'
test_images_path = 'D:/study/大三上/机器学习/matlab project/python/t10k-images.idx3-ubyte'
test_labels_path = 'D:/study/大三上/机器学习/matlab project/python/t10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file,'rb').read()

    #解析文件头信息
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    #print '魔数：%d, 图片数量：%d，图片大小：%d*%d' %(magic_number, num_images, num_rows, num_cols)
    print('图片数量：%d，图片大小：%d*%d' %(num_images, num_rows, num_cols))

    #解析数据集
    image_size = num_cols * num_rows
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i+1)%10000 == 0:
            print("已解析 %d" %(i + 1) + "张")
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):

    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    #print '魔数:%d, 图片数量: %d张' % (magic_number, num_images)
    print('图片数量: %d张' % (num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file = train_images_path):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_train_labels(idx_ubyte_file = train_labels_path):
    return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(idx_ubyte_file = test_images_path):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(idx_ubyte_file = test_labels_path):
    return decode_idx1_ubyte(idx_ubyte_file)

def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    for i in range(10):
        print(train_labels[i])
        #plt.imshow(train_images[i], cmap = 'gray')
        #plt.show()
    print('Loading ... Done!')

if __name__ == "__main__":
    run()
