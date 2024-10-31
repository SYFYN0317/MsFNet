import torch.nn as nn
from torch.nn import functional as F
import torch


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(in_planes, 128 // 16, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(128 // 16, in_planes, 1, bias=False)
        )

        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        x = max_out + avg_out
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.contiguous().view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

#---------------------------------------------------------------------------------------------------------------------------------------------------------
class deeplab_V2(nn.Module):
    def __init__(self):
        super(deeplab_V2, self).__init__()
        self.convdd1 = nn.Sequential(
            Dynamic_conv2d(in_planes=198, out_planes=64, kernel_size=3, ratio=0.25, padding=1, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        '''
            nn.Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
            同时以神经网络模块为元素的有序字典也可以作为传入参数。
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=198, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )
        inter_channels = 512 // 4
        self.conv5a = nn.Sequential(nn.Conv2d(512, inter_channels, 3, padding=1, bias=False),
                                    nn.ReLU())
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))
        ####### multi-scale contexts #######
        ####### dialtion = 6 ##########
        self.fc6_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=6, padding=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 12 ##########
        self.fc6_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=12, padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        #可以看到torch.nn.Dropout对所有元素中每个元素按照概率0.5更改为零
        #而torch.nn.Dropout2d是对每个通道按照概率0.5置为0
        self.fc7_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 18 ##########
        self.fc6_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=18, padding=18),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 24 ##########
        self.fc6_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=24, padding=24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.embedding_layer = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)



    def forward(self, x):
        x = self.convdd1(x)
        x = self.conv2(x)
        feat1 = self.conv5a(x)
        sa_conv = self.conv51(feat1)
        sasc_output = self.conv8(sa_conv)
        return sa_conv , feat1, sasc_output
class SiameseNet(nn.Module):
    def __init__(self, norm_flag='l2'):
        super(SiameseNet, self).__init__()
        self.CNN = deeplab_V2()
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))
        self.conv9 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))
        #  self.sco = attention.CoAM_Module()

        if norm_flag == 'l2':
            self.norm = F.normalize  #F.normalize对输入的数据（tensor）进行指定维度的L2_norm运算
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()

    def forward(self, t0, t1):
        t0 = t0.float()
        t1 = t1.float()


        nwhC0, nwhD0, out_t0_embedding = self.CNN(t0)
        nwhC1, nwhD1, out_t1_embedding = self.CNN(t1)

        return out_t0_embedding,out_t1_embedding
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, 2, padding=1, padding_mode='reflect', bias=False)  # [64, 24, 24]
        self.bat1 = nn.BatchNorm2d(64)#在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.reli1 = nn.LeakyReLU(0.2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect', bias=False)
        self.bat2 = nn.BatchNorm2d(32)
        self.reli2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(64, 256, 3, padding=1, padding_mode='reflect', bias=False)
        self.bat3 = nn.BatchNorm2d(256)
        self.reli3 = nn.LeakyReLU(0.2)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        con1 = self.conv1(x)
        ba1 = self.bat1(con1)
        re1 = self.reli1(ba1)
        po1 = self.pool1(re1)
        con2 = self.conv2(po1)
        ba2 = self.bat2(con2)
        re2 = self.reli2(ba2)
        return re2


"""

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        # 3x3 卷积融合特征
        self.MtoP = nn.Conv2d(256, 128, 1, 1, 1)
        # 横向连接, 使用1x1卷积降维
        self.C2toM2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.C3toM3 = nn.Conv2d(512, 256, 1, 1, 0)


        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )
    # 特征融合方法
    def _upsample_add(self, in_C, in_M):
        H = in_M.shape[2]
        W = in_M.shape[3]
        # 最邻近上采样方法
        return F.upsample_bilinear(in_C, size=(H, W)) + in_M

    def forward(self, x):
        # 自下而上
        C1 = self.conv1(x)
        C2 = self.conv2(C1)
        C3 = self.conv3(C2)
        # 自上而下+横向连接
        M5 = self.C3toM3(C3)
        P5 = self.MtoP(M5)
        # 返回的是多尺度特征
        return P5
"""


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )
        # self.conv4=nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        # )
        # self.conv5=nn.Sequential(
        #     nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        # )

        # 3x3 卷积融合特征
        self.MtoP = nn.Conv2d(128, 128, 3, 1, 1)
        self.MtoP1 = nn.Conv2d(128, 128, 2, 1, 1)
        # 横向连接, 使用1x1卷积降维
        self.C2toM2 = nn.Conv2d(128, 128, 1, 1, 0)
        self.C3toM3 = nn.Conv2d(256, 128, 1, 1, 0)
        self.C4toM4 = nn.Conv2d(512, 128, 1, 1, 0)
        self.C5toM5 = nn.Conv2d(1024, 128, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    # 特征融合方法
    def _upsample_add(self, in_C, in_M):
        H = in_M.shape[2]
        W = in_M.shape[3]
        # 最邻近上采样方法
        return F.upsample_bilinear(in_C, size=(H, W)) + in_M

    def forward(self, x):
        # 自下而上
        C1 = x
        C2 = self.conv2(x)
        C3 = self.conv3(C2)
        C4 = self.conv4(C3)
        # C5 = self.conv4(C4)

        # C1 =
        # C2 = self.conv2(C1)
        # C3 = self.conv3(C2)
        # C4 = self.conv4(C3)
        # C5 = self.conv5(C4)
        # 自上而下+横向连接
        M4 = self.C5toM5(C4)
        MM3= self.C4toM4(C3)
        M3 = self._upsample_add(M4, MM3)
        MM2=self.C3toM3(C2)
        M2 = self._upsample_add(M3,  MM2)
        MM1=self.C2toM2(C1)
        M1 = self._upsample_add(M2, MM1)
        # MM1=self.C2toM2(C1)
        # M2 = self._upsample_add(M3,MM2)
        # 卷积融合
        # P5 = self.MtoP(MM5)
        P5=  self.MtoP(M1)
        P4 = self.MtoP(M4)
        P3 = self.MtoP(M3)
        P2 = self.MtoP1(M2)
        # 返回的是多尺度特征
        return P2

#
#
# import torch
# input1 = torch.rand([32, 128, 5, 5])  #随机生成函数
#
# model =FPN()
#
# # print(model)
# output = model(input1)
# print(output)

class ChangeNet(nn.Module):
    def __init__(self):
        super(ChangeNet, self).__init__()
        self.singlebrach = Classifier()# re2
        self.fc = nn.Sequential(        #一个有序的容器
            nn.Linear(32, 16),#32和16是维度
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.maxpool = nn.MaxPool2d(2)
        self.fpn = FPN()

    def forward(self, t0, t1):
        t0 = self.fpn(t0)
        t1 = self.fpn(t1)
        indata = t0 - t1
        c3 = self.singlebrach(indata)
        return c3


class Finalmodel(nn.Module):#######################nn,Module################################################################
    def __init__(self):
        super(Finalmodel, self).__init__()#子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西。
        self.siamesnet = SiameseNet()
        self.chnet = ChangeNet() #c3
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=1)

        # self.FCN = FPN(layers=[3, 4, 6, 3])
    def forward(self, t0, t1):
        t0 = t0.permute(0, 3, 1, 2) #换个顺序 0123-----0312
        t1 = t1.permute(0, 3, 1, 2)

        x1, x2 = self.siamesnet(t0, t1)
        out5 = self.chnet(x1, x2)
        # out = torch.cat((out2, out3, out4, out5), 1)
        out = self.maxpool(out5)
        out= out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.sigmoid(out)
        return x1, x2, out

#
# import torch
# input1 = torch.rand([32, 9, 9, 155])  #随机生成函数
# input2 = torch.rand([32, 9, 9, 155])
# model =Finalmodel()
#
# # print(model)
# output = model(input1,input2)
# # print(output)
