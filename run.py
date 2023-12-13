# @Time    : 2023/11/22 
# @Author  : Shuaijie Zhao
# @FileName: run.py

import torch
from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset
from dataset_process.brain4cars_dataset import Brain4carDataset
import torch.optim as optim
from time import time
from tqdm import tqdm
import os
import random

from module.transformer import Transformer
from module.transformer_20frame import Transformer_20frame
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization
from utils.predictions import predictManeuver
# from mytest.gather.main import draw

setup_seed(30)  # 设置随机数种子
reslut_figure_path = 'result_figure'  # 结果图像保存路径

# 数据集路径选择
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\AUSLAN\\AUSLAN.mat'  # lenth=1140  input=136 channel=22 output=95
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\CharacterTrajectories\\CharacterTrajectories.mat'
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\CMUsubject16\\CMUsubject16.mat'  # lenth=29,29  input=580 channel=62 output=2
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\ECG\\ECG.mat'  # lenth=100  input=152 channel=2 output=2
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/ECG/ECG.mat"
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\JapaneseVowels\\JapaneseVowels.mat'  # lenth=270  input=29 channel=12 output=9
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\Libras\\Libras.mat'  # lenth=180  input=45 channel=2 output=15
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\UWave\\UWave.mat'  # lenth=4278  input=315 channel=3 output=8
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\KickvsPunch\\KickvsPunch.mat'  # lenth=10  input=841 channel=62 output=2
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\NetFlow\\NetFlow.mat'  # lenth=803  input=997 channel=4 output=只有1和13
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\ArabicDigits\\ArabicDigits.mat'  # lenth=6600  input=93 channel=13 output=10
# path = '/home/ubuntu/zsj/GTN-master/MTS_dataset/ArabicDigits/ArabicDigits.mat'
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\PEMS\\PEMS.mat'
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\Wafer\\Wafer.mat'
# path = 'E:\PyCharmWorkSpace\dataset\\MTS_dataset\\WalkvsRun\\WalkvsRun.mat'
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/AUSLAN/AUSLAN.mat"
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/CMUsubject16/CMUsubject16.mat"
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/CharacterTrajectories/CharacterTrajectories.mat"
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/JapaneseVowels/JapaneseVowels.mat"
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/KickvsPunch/KickvsPunch.mat"
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/Libras/Libras.mat"
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/NetFlow/NetFlow.mat"
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/PEMS/PEMS.mat"
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/UWave/UWave.mat"
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/Wafer/Wafer.mat"
# path = "/home/ubuntu/zsj/GTN-master/MTS_dataset/WalkvsRun/WalkvsRun.mat"

"""brain4car"""
brain4car_train_path = "/home/ubuntu/zsj/GTN-master/dataset/annotations/trainset/brain4cars_train_dataset_random_20frame.pik"
brain4car_test_path = "/home/ubuntu/zsj/GTN-master/dataset/annotations/validset/brain4cars_valid_dataset_random_20frame.pik"
file_name = 'brain4cars'
train_dataset = Brain4carDataset(brain4car_train_path)
test_dataset = Brain4carDataset(brain4car_test_path)
"""brain4car"""

test_interval = 5  # 测试间隔 单位：epoch
draw_key = 1  # 大于等于draw_key才会保存图像
# file_name = path.split('/')[-1][0:path.split('/')[-1].index('.')]  # 获得文件名字


# 超参数设置
EPOCH = 150
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(f'use device: {DEVICE}')

d_model = 512
d_hidden = 1024
q = 8
v = 8
h = 8
N = 8
dropout = 0.2
pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask
# 优化器选择.
optimizer_name = 'Adagrad'

# train_dataset = MyDataset(path, 'train')
# test_dataset = MyDataset(path, 'test')
# For brain4cars

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

"""original data"""
DATA_LEN = train_dataset.dataset_len  # 训练集样本数量
d_input = train_dataset.input_len  # 时间部数量
d_channel = train_dataset.channel_len  # 时间序列维度
d_output = train_dataset.output_len  # 分类类别
"""brain4car"""
# DATA_LEN = train_dataset.dataset_len  # 训练集样本数量
# d_input = train_dataset.input_len  # 时间部数量
# d_channel = train_dataset.channel_len  # 时间序列维度
# d_output = train_dataset.output_len  # 分类类别

# 维度展示
print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{test_dataset.dataset_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')


# ckpt = torch.load("/home/ubuntu/zsj/PFL-Non-IID-master/saved_model/ArabicDigits batch=64_statedict.pkl", map_location=DEVICE)

# 创建Transformer模型
# net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
#                   q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
net = Transformer_20frame(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)

loss_function = Myloss()
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

# CosineAnnealingWarmRestarts
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=EPOCH, eta_min=1e-6)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=50, T_mult=2, eta_min=1e-6)

# 用于记录准确率变化
correct_on_train = []
correct_on_test = []
# 用于记录损失变化
loss_list = []
time_cost = 0


# 测试函数
def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)

# this is for 32 6 7
def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    anticipation_time = 0 
    prediction_list = []
    anticipation_list = []
    labels = []
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # obtain the final truth
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=1)
            total += label_index.shape[0]
            """best prediction for anticipation_time"""
            # y[y > 0] -= 1
            # prediction_list_batch, anticipation_list_batch = predictManeuver(label_index)
            # actual = y[:, -1].tolist()
            # prediction_list += prediction_list_batch
            # anticipation_list += anticipation_list_batch
            # labels += actual
            """best prediction for anticipation_time"""
            correct_per_row = torch.sum(label_index == y, dim=1)
            correct += torch.sum(correct_per_row > 3).item()

            # correct += (label_index == y.long()).sum().item()
        # if len(prediction_list) == len(labels):
        #     pairs = zip(prediction_list, labels)
        #     correct = sum(int(x) == int(y) for x, y in pairs)
        # non_zero_time = [x for x in anticipation_list if x != 0]
        # anticipation_time = sum(non_zero_time) / len(non_zero_time) if len(non_zero_time) > 0 else 0

        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)


# 训练函数
def train():
    net.train()
    max_accuracy = 0
    pbar = tqdm(total=EPOCH)
    begin = time()
    for index in range(EPOCH):
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')

            loss = loss_function(y_pre, y.to(DEVICE))

            print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            loss_list.append(loss.item())

            loss.backward()

            optimizer.step()

        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader)
            test(train_dataloader, 'train_set')
            print(f'当前最大准确率\t测试集:{max(correct_on_test)}%\t 训练集:{max(correct_on_train)}%, 学习率：{LR}')

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')
        scheduler.step()
        LR = scheduler.get_last_lr()[0]
        pbar.update()

    os.rename(f'saved_model/{file_name} batch={BATCH_SIZE}.pkl',
              f'saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl')

    end = time()
    time_cost = round((end - begin) / 60, 2)

    # 结果图
    result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
                         test_interval=test_interval,
                         d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
                         time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
                         file_name=file_name,
                         optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask)


if __name__ == '__main__':
    train()
    # test(test_dataloader)
