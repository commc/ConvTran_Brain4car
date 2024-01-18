# @Time    : 2023/11/22 
# @Author  : Shuaijie Zhao
# @FileName: run.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from time import time
from tqdm import tqdm
import argparse

import random
from dataset_process.dataset_process import MyDataset
from dataset_process.brain4cars_dataset import Brain4carDataset
from module.transformer import Transformer
from module.transformer_multilabel import Transformer_multilabel
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization, plot_confusion_matrix, plot_result
from utils.predictions import predictManeuver, confusionMat
# from mytest.gather.main import draw

setup_seed(16)  # 设置随机数种子
reslut_figure_path = 'result_figure'  # 结果图像保存路径

# 数据集路径选择
"""brain4car"""
brain4car_train_path = "./brain4cars_pipeline/temp_data/folder5/brain4cars_train_dataset_random.json"
brain4car_test_path = "./brain4cars_pipeline/temp_data/folder5/brain4cars_valid_dataset_random.json"
file_name = 'brain4cars'
train_dataset = Brain4carDataset(brain4car_train_path)
test_dataset = Brain4carDataset(brain4car_test_path)

test_interval = 5  # 测试间隔 单位：epoch
draw_key = 1  # 大于等于draw_key才会保存图像
# file_name = path.split('/')[-1][0:path.split('/')[-1].index('.')]  # 获得文件名字
action_name = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']

# 超参数设置
EPOCH = 150
BATCH_SIZE = 32
LR = 0.00017358693285391162
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(f'use device: {DEVICE}')

d_model = 512
d_hidden = 1024
q = 8
v = 8
h = 8
N = 4
dropout = 0.1
pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask
# 优化器选择.
optimizer_name = 'Adagrad'

# train_dataset = MyDataset(path, 'train')
# test_dataset = MyDataset(path, 'test')
# For brain4cars

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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


# 创建Transformer模型
# net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
#                   q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
net = Transformer_multilabel(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)

loss_function = Myloss()
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

# CosineAnnealingWarmRestarts
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=EPOCH, eta_min=1e-6)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=50, T_mult=2, eta_min=1e-6)

# 用于记录准确率变化
correct_on_train = []
correct_on_test = []
# 用于记录损失变化
loss_list = []
time_cost = 0

# # 用于多标签的测试
def test(net, dataloader, flag='test_set'):
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
            _, label_pre = torch.max(y_pre.data, dim=1)
            label_pre[label_pre > 0] -= 1
            y[y > 0] -= 1
            total += label_pre.shape[0]
            """best prediction for anticipation_time"""
            prediction_list_batch, anticipation_list_batch = predictManeuver(label_pre)
            actual = y[:, -1].tolist()
            actual = [int(x) for x in actual]
            prediction_list += prediction_list_batch
            anticipation_list += anticipation_list_batch
            labels += actual

        confMat, precision_confMat, recall_confMat, TimeMat = confusionMat(prediction_list, labels, anticipation_list)
        avg_precision = np.mean(np.diag(precision_confMat)[1:])
        lane_changing_precision = np.mean(np.diag(precision_confMat)[1::2])
        turns_precision = np.mean(np.diag(precision_confMat)[2::2])
        avg_recall = np.mean(np.diag(recall_confMat)[1:])
        avg_anticipation_time = np.mean(np.divide(np.diag(TimeMat)[1:], np.diag(confMat)[1:]))

        if flag == 'test_set':
            correct_on_test.append(round((100 * avg_precision), 2))
        elif flag == 'train_set':
            correct_on_train.append(round((100 * avg_precision), 2))
        print(f'All maneuver Precision on {flag}: %.2f %%' % (100 * avg_precision))
        print(f'Lane changing maneuver Precision on {flag}: %.2f %%' % (100 * lane_changing_precision))
        print(f'Turn maneuver Precision on {flag}: %.2f %%' % (100 * turns_precision))
        print(f'Anticipation time on {flag}: %.2f ' % (avg_anticipation_time))

    return round((100 * avg_precision), 2), confMat, precision_confMat, recall_confMat, avg_anticipation_time

# 训练函数
def train():
    net.train()
    max_accuracy = 0
    best_confMat, best_precision_confMat, best_recall_confMat = None, None, None
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
            current_accuracy, current_confMat, current_precision_confMat, current_recall_confMat, current_anticipation_time = test(net, test_dataloader)
            test(net, train_dataloader, 'train_set')
            print(f'当前最大准确率\t测试集:{max(correct_on_test)}%\t 训练集:{max(correct_on_train)}%, 学习率：{LR}')

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                best_confMat = current_confMat
                best_precision_confMat = current_precision_confMat
                best_recall_confMat = current_recall_confMat
                best_anticipation_time = current_anticipation_time
                torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')
        # scheduler.step()
        # LR = scheduler.get_last_lr()[0]
        pbar.update()

    os.rename(f'saved_model/{file_name} batch={BATCH_SIZE}.pkl',
              f'saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl')

    end = time()
    time_cost = round((end - begin) / 60, 2)

    # 结果图
    # result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
    #                      test_interval=test_interval,
    #                      d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
    #                      time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
    #                      file_name=file_name, optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask)
    plot_result(loss_list, correct_on_test, correct_on_train, test_interval, reslut_figure_path, file_name)
    plot_confusion_matrix(best_confMat, action_name, reslut_figure_path, "confMat")
    plot_confusion_matrix(best_precision_confMat, action_name, reslut_figure_path, "precision")
    plot_confusion_matrix(best_recall_confMat, action_name, reslut_figure_path, "recall")
    
    return max_accuracy, best_confMat, best_precision_confMat, best_recall_confMat, best_anticipation_time


def kFolderValidation(folder_path, k=5):
    accuracy_list = []
    confMat_list = []
    precision_confMat_list = []
    avg_precision_list = []
    recall_confMat_list = []
    avg_recall_list = []
    anticipation_time_list = []

    for i in range(k):
        correct_on_train = []
        correct_on_test = []
        # 用于记录损失变化
        loss_list = []
        print("-"*15,f"Folder{i+1}开始验证", "-"*15)
        brain4car_train_path = os.path.join(folder_path, f"folder{i+1}", "brain4cars_train_dataset_random.json")
        brain4car_test_path = os.path.join(folder_path, f"folder{i+1}", "brain4cars_valid_dataset_random.json")
        train_dataset = Brain4carDataset(brain4car_train_path)
        test_dataset = Brain4carDataset(brain4car_test_path) 
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        DATA_LEN = train_dataset.dataset_len  # 训练集样本数量
        d_input = train_dataset.input_len  # 时间部数量
        d_channel = train_dataset.channel_len  # 时间序列维度
        d_output = train_dataset.output_len  # 分类类别
        # 维度展示
        print('data structure: [lines, timesteps, features]')
        print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
        print(f'mytest data size: [{test_dataset.dataset_len, d_input, d_channel}]')
        print(f'Number of classes: {d_output}')
        # 模型创建
        net = Transformer_multilabel(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
        optimizer_name = 'Adagrad'
        loss_function = Myloss()
        if optimizer_name == 'Adagrad':
            optimizer = optim.Adagrad(net.parameters(), lr=LR)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=LR)

        max_accuracy, best_confMat, best_precision_confMat, best_recall_confMat, best_anticipation_time = train()
        accuracy_list.append(max_accuracy)
        confMat_list.append(best_confMat)
        precision_confMat_list.append(best_precision_confMat)
        avg_precision_list.append(np.mean(np.diag(best_precision_confMat)[1:]))
        recall_confMat_list.append(best_recall_confMat)
        avg_recall_list.append(np.mean(np.diag(best_recall_confMat)[1:]))
        anticipation_time_list.append(best_anticipation_time)
    
    accuracy_list = np.array(accuracy_list)
    precision_confMat_list = np.array(precision_confMat_list)
    avg_precision_list = np.array(avg_precision_list)
    recall_confMat_list = np.array(avg_recall_list)
    avg_recall_list = np.array(avg_recall_list)
    anticipation_time_list = np.array(anticipation_time_list)
    
    print("*"*20)
    print(f"Precision = {np.mean(avg_precision_list)} {np.std(avg_precision_list)}")
    print(f"Recall = {np.mean(avg_recall_list)} {np.std(avg_recall_list)}")
    print(f"Anticipation time = {np.mean(anticipation_time_list)} {np.std(anticipation_time_list)}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, default="./brain4cars_pipeline/temp_data/folder5/brain4cars_train_dataset_random.json", help="train dataset path")
    parser.add_argument('--test-path', type=str, default=".brain4cars_pipeline/temp_data/folder5/brain4cars_valid_dataset_random.json", help="train dataset path")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32, help="total batch size for GPUs")
    parser.add_argument('--lr', type=float, default=0.00017358693285391162, help="learning rate")
    parser.add_argument('--optimizer', type=str, default="Adagrad", help="optimizer for back propogation")
    parser.add_argument('--device', type=str, default="0", help="cuda device, i.e. 0,1,2,3 or cpu")
    parser.add_argument('--d-model', type=int, default=512, help="embedding dimension")
    parser.add_argument('--d-hidden', type=int, default=1024, help="fully connected layer hidden dimension")
    parser.add_argument('--q', type=int, default=8, help="querry ")
    parser.add_argument('--v', type=int, default=8, help="value")
    parser.add_argument('--h', type=int, default=4, help="n heads")
    parser.add_argument('--N', type=int, default=4, help="n encoders")
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pe', type=bool, default=True, help=" positional embefding")
    parser.add_argument('--mask', type=bool, default=True, help="mask for step encoder")
    parser.add_argument('--reslut-figure-path', type=str, default="result_figure", help="path for saving figures")
    parser.add_argument('--file-name', type=str, default="brain4cars", help="figure prefix")
    parser.add_argument('--test-interval', type=int, default=5, help="after test interval epochs training, test")
    parser.add_argument('--draw-key', type=int, default=1, help="1")
    parser.add_argument('--random-seed', type=int, default=16)

    # train()
    folder_path = "./brain4cars_pipeline/temp_data/"
    kFolderValidation(folder_path)
    # test(test_dataloader)

