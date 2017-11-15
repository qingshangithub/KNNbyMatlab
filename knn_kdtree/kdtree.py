import numpy as np
import itertools
from scipy.stats import mode
import sys
sys.setrecursionlimit(1000000)
import copy
#encoding:utf-8
import struct

train_images_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

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


class kd_node:
    '''
    KD树的节点
    '''

    def __init__(self, point, dim):
        self.point = point  # kd树储存的点
        self.split_dim = dim  # 分割维度
        self.left = None
        self.right = None


def zhwsh(data_lst, split_dim):
    '''找出data_lst的中位数'''
    d = len(data_lst) / 2
    l = 0
    h = int(d)
    while l < h:
        m = int((l + h) / 2)
        if data_lst[m][split_dim] < data_lst[h][split_dim]:
            l = m + 1
        else:
            h = m
    return data_lst[h], h


def get_split_dim(data_lst):
    """
    计算points在每个维度上的和, 选择在和最大的维度上进行切割
    """

    sum_lst = np.sum(data_lst, axis=0)
    split_dim = 0
    for v in range(1, len(sum_lst)):
        if sum_lst[v] > sum_lst[split_dim]:
            split_dim = v
    return split_dim


def build_kdtree(data_lst):
    '''构建kd树'''
    split_dim = get_split_dim(data_lst)
    data_lst = sorted(data_lst, key=lambda x: x[split_dim])
    point, m = zhwsh(data_lst, split_dim)
    tree_node = kd_node(point, split_dim)

    if m > 0:
        tree_node.left = build_kdtree(data_lst[:m])
    if len(data_lst) > m + 1:
        tree_node.right = build_kdtree(data_lst[m + 1:])

    return tree_node

def euclid_distance(d1, d2):
    dist = np.linalg.norm(np.array(d1) - np.array(d2))
    return dist


class NeiNode:
    '''neighbor node'''

    def __init__(self, p, d):
        self.__point = p
        self.__dist = d

    def get_point(self):
        return self.__point

    def get_dist(self):
        return self.__dist


class BPQ:
    '''优先队列'''

    def __init__(self, k):
        self.__K = k
        self.__pos = 0
        self.__bpq = [0] * (k + 2)

    def add_neighbor(self, neighbor):
        self.__pos += 1
        self.__bpq[self.__pos] = neighbor
        self.__swim_up(self.__pos)
        if self.__pos > self.__K:
            self.__exchange(1, self.__pos)
            self.__pos -= 1
            self.__sink_down(1)

    def get_knn_points(self):
        return [neighbor.get_point() for neighbor in self.__bpq[1:self.__pos + 1]]

    def get_max_distance(self):
        if self.__pos > 0:
            return self.__bpq[1].get_dist()
        return 0

    def get_knearest(self,k):
        if self.__pos > 0:
            tmp=[]
            while k > 0:
                tmp.append([self.__bpq[k].get_dist(),self.__bpq[k].get_point()])
                k = k-1
            return tmp
        return 0

    def is_full(self):
        return self.__pos >= self.__K



    def __swim_up(self, n):
        while n > 1 and self.__less(int(n / 2), n):
            self.__exchange(int(n / 2), n)
            n = n / 2

    def __sink_down(self, n):
        while 2 * n <= self.__pos:
            j = 2 * n
            if j < self.__pos and self.__less(j, j + 1):
                j += 1
            if not self.__less(n, j):
                break
            self.__exchange(n, j)
            n = j

    def __less(self, m, n):
        if m != 0:
            return self.__bpq[m].get_dist() < self.__bpq[n].get_dist()

    def __exchange(self, m, n):
        tmp = self.__bpq[m]
        self.__bpq[m] = self.__bpq[n]
        self.__bpq[n] = tmp


def knn_search_kd_tree_non_recursively(knn_bpq, tree, target, search_track):
    track_node = []
    node_ptr = tree
    while node_ptr:
        while node_ptr:
            track_node.append(node_ptr)
            search_track.append([node_ptr.point, knn_bpq.get_knn_points(), knn_bpq.get_max_distance()])
            dist = euclid_distance(node_ptr.point, target)
            knn_bpq.add_neighbor(NeiNode(node_ptr.point, dist))

            search_track.append([None, knn_bpq.get_knn_points(), knn_bpq.get_max_distance()])

            split_dim = node_ptr.split_dim
            if target[split_dim] < node_ptr.point[split_dim]:
                node_ptr = node_ptr.left
            else:
                node_ptr = node_ptr.right

        while track_node:
            iter_node = track_node[-1]
            del track_node[-1]

            split_dim = iter_node.split_dim
            if not knn_bpq.is_full() or \
                            abs(iter_node.point[split_dim] - target[split_dim]) < knn_bpq.get_max_distance():
                if target[split_dim] < iter_node.point[split_dim]:
                    node_ptr = iter_node.right
                else:
                    node_ptr = iter_node.left

            if node_ptr:
                break
    a = knn_bpq.get_knearest(k)
    return a

if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    T = train_images
    nn = 3999
    mm = 399
    TT = [[0]*784]*(nn+1)
    TST = [[0]*784]*(mm+1)
    n = nn
    m = mm
    while n >= 0:
        TT[n] = list(itertools.chain.from_iterable(T[n]))
        n = n-1
    while m >= 0:
        TST[m] = list(itertools.chain.from_iterable(test_images[m]))
        m = m-1
    print('开始建树')
    kd_tree = build_kdtree(TT)
    print('建树结束\n')

    k = 2
    m = mm
    labelK = [0]*k
    labelResult = [0]*(m+1)
    j = 0
    print('begin to search target point in kd-tree')
    while j <= m:
        knn_bpq = BPQ(k)
        search_track = []
        a = knn_search_kd_tree_non_recursively(knn_bpq, kd_tree, TST[j], search_track)
        tmp1 = 0
        while tmp1 < k:
            labelK[tmp1] = TT.index(a[tmp1][1])
            tmp1 = tmp1+1
        print(train_labels[labelK[:]])
        labelResult[j] = int(mode(train_labels[labelK[:]])[0][0])
        j = j+1
    lx = 0
    error = 0
    while lx <= m:
        if test_labels[lx] != labelResult[lx]:
            error = error+1
        lx = lx+1
    accuracy = 1-error/(m+1)
    print(accuracy)

