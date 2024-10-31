import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
import extract_samll_cubic
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

def load_dataset(Dataset):  # 有三张图片 china river usa
    if Dataset == 'china':
        T1_ori = sio.loadmat('F:/LNNU/data/CD数据集/数据集/Farm1.mat')
        T2_ori = sio.loadmat('F:/LNNU/data/CD数据集/数据集/Farm2.mat')
        mat_gt = sio.loadmat('F:/LNNU/data/CD数据集/数据集/GTChina1.mat')
        # print("test T2_ori:", T2_ori)
        # print("test mat_gt:", mat_gt)
        TT1 = T1_ori['imgh']
        TT2 = T2_ori['imghl']
        gt_hsi = mat_gt['label']
        TOTAL_SIZE = 63000
        VALIDATION_SPLIT = 0.7905
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)  #训练数目

    if Dataset =='river':
        T1_ori = sio.loadmat('F:/LNNU/data/CD数据集/数据集/River_before.mat')
        T2_ori = sio.loadmat('F:/LNNU/data/CD数据集/数据集/River_after.mat')
        mat_gt = sio.loadmat('F:/LNNU/data/CD数据集/数据集/Rivergt.mat')
        TT1 = T1_ori['river_before']
        TT2 = T2_ori['river_after']
        gt_hsi = mat_gt['gt']
        TOTAL_SIZE = 111583
        VALIDATION_SPLIT = 0.945
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'usa':
        T1_ori = sio.loadmat('F:/LNNU/data/CD数据集/数据集/Sa1.mat')
        T2_ori = sio.loadmat('F:/LNNU/data/CD数据集/数据集/Sa2.mat')
        mat_gt = sio.loadmat('F:/LNNU/data/CD数据集/数据集/SaGT.mat')
        TT1 = T1_ori['T1']
        TT2 = T2_ori['T2']
        gt_hsi = mat_gt['GT']
        TOTAL_SIZE = 73987
        VALIDATION_SPLIT = 0.902
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return TT1, TT2, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT

def save_cmap(img, cmap, fname):    #画的是什么表？？
    sizes = np.shape(img)  #读取矩阵的长度 返回 几行 几列
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()          #画图的 创建figure对象
    fig.set_size_inches(width / height, 1, forward=False)  #定义格式 英尺  指定画布大小
    ax = plt.Axes(fig, [0., 0., 1., 1.]) #######d对x,y轴进行设置，轴的范围##################################################
    ax.set_axis_off() #关闭轴
    fig.add_axes(ax)  #在figure上画图

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()



def sampling(proportion, ground_truth):  #样本 （比例，真值）
    train = {} #训练
    test = {}   #测试
    labels_loc = {}   #字典
    m = max(ground_truth)  #返回最大值
    for i in range(m): #m=63000
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]  #ravel  扁平化操作  tolist () 转化为列表形式
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        np.random.shuffle(indexes) #重新排序返回一个随机序列作用类似洗牌
        labels_loc[i] = indexes  # labels_loc是一个空的字典 上面几行有 定义完了
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)  #len 返回对象的长度或者项目个数
        else:
            nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]   #nb_val 项目的最大数
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)   #重新排序返回一个随机序列作用类似洗牌
    np.random.shuffle(test_indexes)   #重新排序返回一个随机序列作用类似洗牌
    return train_indexes, test_indexes   #返回了一堆数 不明白这些数是干嘛的



def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)  #默认为对输入参数中的所有元素进行求和
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))#＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃truediv＃＃＃
    average_acc = np.mean(each_acc)  #求均值 详见JAY
    return each_acc, average_acc



def classification_map(map, ground_truth, dpi, save_path):  #######savepath dpi 什么意思#################################
    fig = plt.figure(frameon=False)  #plt.figure()用来画图，自定义画布大小  将轴的可见性设置为false
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)  #隐藏框架
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)  #绘图完成 保存图片

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):#用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列,同时列出数据和数据下标,一般用在 for 循环当中
        if item == 1:
            y[index] = np.array([255, 255, 255]) / 255.
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
    return y




def generate_iter(TRAIN_SIZE, train_indices,  TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data1, whole_data2, PATCH_LENGTH, padded_data1, padded_data2, INPUT_DIMENSION1,
                  INPUT_DIMENSION2, batch_size, gt):
    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1

    all_data1 = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data1,
                                                       PATCH_LENGTH, padded_data1, INPUT_DIMENSION1)
    all_data2 = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data2,
                                                       PATCH_LENGTH, padded_data2, INPUT_DIMENSION2)

    train_data1 = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data1,
                                                         PATCH_LENGTH, padded_data1, INPUT_DIMENSION1)
    train_data2 = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data2,
                                                         PATCH_LENGTH, padded_data2, INPUT_DIMENSION2)

    x_train1 = train_data1.reshape(train_data1.shape[0], train_data1.shape[1], train_data1.shape[2], INPUT_DIMENSION1)

    x_train2 = train_data2.reshape(train_data2.shape[0], train_data2.shape[1], train_data2.shape[2], INPUT_DIMENSION2)
    all_data1.reshape(all_data1.shape[0], all_data1.shape[1], all_data1.shape[2], INPUT_DIMENSION1)
    all_data2.reshape(all_data2.shape[0], all_data2.shape[1], all_data2.shape[2], INPUT_DIMENSION2)
    x_val1 = all_data1[7000:9000]
    x_val2 = all_data2[7000:9000]
    y_val = gt_all[7000:9000]


    x1_tensor_train = torch.from_numpy(x_train1).type(torch.FloatTensor)
    x2_tensor_train = torch.from_numpy(x_train2).type(torch.FloatTensor)
    y_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, x2_tensor_train, y_tensor_train)


    x1_tensor_valida = torch.from_numpy(x_val1).type(torch.FloatTensor)
    x2_tensor_valida = torch.from_numpy(x_val2).type(torch.FloatTensor)
    y_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, x2_tensor_valida, y_tensor_valida)
    #


    all_tensor_data1 = torch.from_numpy(all_data1).type(torch.FloatTensor)
    all_tensor_data2 = torch.from_numpy(all_data2).type(torch.FloatTensor)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data1, all_tensor_data2, all_tensor_data_label)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valiada_iter = Data.DataLoader(
         dataset=torch_dataset_valida,
         batch_size=batch_size,
         shuffle=True,
         num_workers=0,
     )

    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_iter,  valiada_iter, all_iter

def generate_png(pred_test, gt_hsi, Dataset, total_indices):

    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)

    x_label[total_indices] = pred_test
    x = np.ravel(x_label)

    # print('-------Save the result in mat format--------')
    # x_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
    # sio.savemat('mat/' + Dataset + '_' + '.mat', {Dataset: x_re})

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    name = 'CDCNN'
    path = '../' + name
    classification_map(y_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_' + name + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_gt.png')
    print('------Get classification maps successful-------')

class ConstractiveLoss(nn.Module):

    def __init__(self,margin =2.0,dist_flag='l2'):    #参数自动传递？？
        super(ConstractiveLoss, self).__init__()
        self.margin = margin
        self.dist_flag = dist_flag

    def various_distance(self,out_vec_t0,out_vec_t1):   #######计算距离 像素之间的距离？？ 是要干什么呢？？？

        if self.dist_flag == 'l2':
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)  #F.pairwise_distance特征图之间像素级的距离
        if self.dist_flag == 'l1':
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=1)
        if self.dist_flag == 'cos':
            similarity = F.cosine_similarity(out_vec_t0,out_vec_t1)
            distance = 1 - 2 * similarity/np.pi
        return distance

    def forward(self,out_vec_t0,out_vec_t1,label):

        #distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        distance = self.various_distance(out_vec_t0,out_vec_t1)
        #distance = 1 - F.cosine_similarity(out_vec_t0,out_vec_t1)
        constractive_loss = torch.sum((1-label)*torch.pow(distance,2) + label * torch.pow(torch.clamp(self.margin - distance, min=0.0),2))
        return constractive_loss

def applyPCA(X, numComponents=75):   #numComponents 转换特征的数目
    newX = np.reshape(X, (-1, X.shape[2]))  #在不改变矩阵的数值的前提下修改矩阵X的形状
    pca = PCA(n_components=numComponents, whiten=True) #降维
    newX = pca.fit_transform(newX)  #用于从训练数据生成学习模型参数
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca




class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
   """
   Network has to have NO NONLINEARITY!
   """
   def __init__(self, weight=None):
       super(WeightedCrossEntropyLoss, self).__init__()
       self.weight = weight

   def forward(self, inp, target):
       target = target.long()
       num_classes = inp.size()[1]

       i0 = 1
       i1 = 2

       while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
           inp = inp.transpose(i0, i1)
           i0 += 1
           i1 += 1

       inp = inp.contiguous()
       inp = inp.view(-1, num_classes)

       target = target.view(-1,)
       wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

       return wce_loss(inp, target)



