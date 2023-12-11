import torch
from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset
from dataset_process.brain4cars_dataset import Brain4carDataset
import torch.optim as optim
from time import time
from tqdm import tqdm
import os

from module.MLSTM_FCN import MLSTM_FCN
from module.transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization


"""brain4car"""
brain4car_train_path = "/home/ubuntu/zsj/GTN-master/dataset/annotations/trainset/brain4cars_train_dataset_random.json"
brain4car_test_path = "/home/ubuntu/zsj/GTN-master/dataset/annotations/validset/brain4cars_valid_dataset_random.json"
file_name = 'brain4cars'
train_dataset = Brain4carDataset(brain4car_train_path)
test_dataset = Brain4carDataset(brain4car_test_path)
"""brain4car"""

test_interval = 5  # 测试间隔 单位：epoch
draw_key = 1  # 大于等于draw_key才会保存图像

EPOCH = 250
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(f'use device: {DEVICE}')

# 优化器选择.
optimizer_name = 'Adagrad'


train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""original data"""
DATA_LEN = train_dataset.dataset_len  # 训练集样本数量
d_input = train_dataset.input_len  # 时间部数量
d_channel = train_dataset.channel_len  # 时间序列维度
d_output = train_dataset.output_len  # 分类类别

print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{test_dataset.dataset_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

net = MLSTM_FCN(Batch_Size=BATCH_SIZE, N_Features=d_channel, N_ClassesOut=d_output).to(DEVICE)

loss_function = Myloss()
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)


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
            y_pre = net(x)
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
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

            y_pre= net(x.to(DEVICE))

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
    #                      file_name=file_name,
    #                      optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask)
if __name__ == '__main__':
    # train()
    print(torch.backends.cudnn.version())