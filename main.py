import numpy as np  #引入numpy模块取别名为np
import time
import collections
from torch import optim #引入torch中的optim
import torch
from sklearn import metrics, preprocessing#用于导入一个模块中的某一个部分，比如一个函数或者一个类等。
import datetime
from generate_pic import ConstractiveLoss
from generate_pic import WeightedCrossEntropyLoss
import train
from DASNET import Finalmodel
import sys
from scipy.io import savemat
import matplotlib.pyplot as plt
sys.path.append('../global_module/')  #改变一下路径




from generate_pic import aa_and_each_accuracy, sampling,load_dataset, generate_png, generate_iter,applyPCA
import record, extract_samll_cubic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#####################################################

# for Monte Carlo runs
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]#产生随机种子意味着每次运行实验，产生的随机数都是相同的
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')  #当地时间

print('-----Importing Dataset-----')


global Dataset  # UP,IN,KSC   三张图片

Dataset = 'river'
lr, num_epochs, batch_size = 0.0005, 200, 32
T11, T22, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(Dataset)  #在genarate里面定义的一个函数 读取图片还有数目

print(T11.shape)

image_x, image_y, BAND1 = T11.shape


data1 = T11.reshape(np.prod(T11.shape[:2]), np.prod(T11.shape[2:]))  #连乘操作，将里面所有的元素相乘
data2 = T22.reshape(np.prod(T22.shape[:2]), np.prod(T22.shape[2:]))  #改变形状 三维矩阵变成二维矩阵
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),) #二维变一维
CLASSES_NUM = 2


print('-----Importing Setting Parameters-----')  #导入设置参数
ITER = 1

PATCH_LENGTH = 2
loss = torch.nn.CrossEntropyLoss()
loss1 = WeightedCrossEntropyLoss()  #加权交叉熵损失函数


img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
img_channels1 = T11.shape[2]    #155
img_channels2 = T22.shape[2]    #155
INPUT_DIMENSION1 = T11.shape[2]  #155
INPUT_DIMENSION2 = T22.shape[2]  #155
ALL_SIZE = T11.shape[0] * T22.shape[1]  #450*140
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

data1 = preprocessing.scale(data1)
data2 = preprocessing.scale(data2) #数据标准化

data1_ = data1.reshape(T11.shape[0], T11.shape[1], T11.shape[2])

whole_data1 = data1_
data2_ = data2.reshape(T22.shape[0], T22.shape[1], T22.shape[2])
whole_data2 = data2_


padded_data1 = np.lib.pad(whole_data1, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)  #对图像进行填充 得到（456，146，155）
padded_data2 = np.lib.pad(whole_data2, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)  #二维数组填充  把周围填黑   得到（456，146，155）

#指标
KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))#一行两列


for index_iter in range(ITER):  # iter=1
    net = Finalmodel()# x1  x2  out
    params = sum(p.numel() for p in list(net.parameters())) / 1e6  # numel()
    print('#Params: %.1fM' % (params))
    optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.0001)   lr=0.0005#######不理解################################
    #优化器

    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)#VALIDATION_SPLIT=0.787,gt=63000个#输入比例为0.787的样本集，进行训练 测试
    _, total_indices = sampling(1, gt)  #输入比例为全部的样本集，进行训练 测试

    # 打印相关数据
    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----') #从原始立方体数据中选择小块

    train_iter, valida_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                                                      whole_data1, whole_data2, PATCH_LENGTH, padded_data1,
                                                      padded_data2, INPUT_DIMENSION1, INPUT_DIMENSION2, batch_size, gt)

    train.train(net, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs)


    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        return -val_loss
    pred_test_fdssc = []  #测试的结果
    tic2 = time.perf_counter()#time.clock()是统计cpu时间的工具,这在统计某一程序或函数的执行速度最为合适。两次调用time.clock()函数的插值即为程序运行的cpu时间。
    newnet = torch.load("UPbest.pth")
    with torch.no_grad():
        for X1, X2, y in all_iter:
            X1 = X1.to(device)
            X2 = X2.to(device)
            newnet.eval()  # 评估模式, 这会关闭dropout
            out1, out2, y_hat = newnet(X1,X2)
            # print(net(X))
            pred_test_fdssc.extend(np.array(y_hat.cpu().argmax(axis=1))) #argmax取出a中元素最大值所对应的索引
    toc2 = time.perf_counter()

    collections.Counter(pred_test_fdssc) #显然代码更加简单了，也更容易读和维护了。
    gt_test = gt[total_indices]-1

    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test) #测试集 的结果和真值做对比
    confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test)
    label_name = ['cat','dog']
    confusion_matrix_fdssc1 = confusion_matrix_fdssc
    row_sums = np.sum(confusion_matrix_fdssc, axis=1)
    confusion_matrix_fdssc = confusion_matrix_fdssc / row_sums[:, np.newaxis]
    confusion_matrix_fdssc = confusion_matrix_fdssc.T
    plt.imshow(confusion_matrix_fdssc, cmap='Blues')
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.1f' % (confusion_matrix_fdssc[i, j] * 100)))
            value1 = str(value) + '%\n' + str(confusion_matrix_fdssc[i, j])
            plt.text(i, j, value1, verticalalignment='center', horizontalalignment='center', color=color)

    plt.show()
    plt.savefig("Confusion_Matrix.jpg", bbox_inches='tight', dpi=300)

    # print(confusion_matrix_fdssc)
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test)

    savemat("china.mat", mdict={'result': pred_test_fdssc})
    KAPPA.append(kappa)
    OA.append(overall_acc_fdssc)
    AA.append(average_acc_fdssc)
    ELEMENT_ACC[index_iter, :] = each_acc_fdssc

    print("-------- Training Finished-----------")
    print('OA:',OA)
    print('AA:',AA)
    print('Kappa:',KAPPA)
    print('OA_UN',each_acc_fdssc)
    print(confusion_matrix_fdssc)
    print(confusion_matrix_fdssc1)
    print(toc2 - time_1)
    print(tic2 - time_1)
    generate_png(pred_test_fdssc, gt_hsi-1, 'fyn_usa_m', total_indices)






