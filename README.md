# KNNbyMatlab
<<<<<<< HEAD
KNN算法的matlab实现。训练了两个数据集，分别是MNIST和CIFAR，

1. 数据库下载地址：

   1. mnist ： http://yann.lecun.com/exdb/mnist/	

   2. cifar10： https://www.kaggle.com/c/cifar-10/data

2. KNN_MNIST

   1. 运行环境：将MNIST四个数据集文件加入文件夹内即可运行。
   2. 在matlab2017b环境下编写测试
   3. 默认运行全部数据，大约耗时2000到3000秒
   4. 运行过程中，会输出分类错误的样本索引，可从输出的错误分类样本索引大概估计出准确率。
 3. KNN_CIFAR
   1. 运行环境：将CIFAR七个数据集文件加入文件夹内即可运行。
   2. 在matlab2017b环境下编写测试
   3. 默认运行全部数据，大约耗时2000到3000秒
   4. 运行过程中，会输出分类错误的样本索引，可从输出的错误分类样本索引大概估计出准确率。正确率大约在38%左右。
 4. KNN_KDTREE
   1. 运行环境：python3.6 需要导入scipy库、numpy库，将MNIST四个数据集文件加入文件夹内；
   2. 在文件夹中打开windowsPowerShell，输入命令python .\kdtree.py运行即可；
   3. 默认运行4000个训练数据，400个测试数据；
   4. 会输出查找的标签值，默认K=2；
   5. 运行时间几分钟，正确率89%；
   6. 建树过程用递归实现。跑MNIST全部数据集会内存报错，经测试上限大约20000~30000个数据（8G内存），搜索非递归；
   7. 参考了代码：https://github.com/richardxdh/ml_algorithms/blob/master/kd_tree.py进行了一些修改，源代码下下来会报错，另外，采用方差求最佳维度会非常耗时，改成了求和函数；



=======
KNN realization by matlab.
>>>>>>> parent of 5e2edec... First commit
