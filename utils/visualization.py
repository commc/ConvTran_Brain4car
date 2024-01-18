import os
import time
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as fp  # 1、引入FontProperties
import math

logger = logging.getLogger(__name__)

def result_visualization(loss_list: list,
                         correct_on_test: list,
                         correct_on_train: list,
                         test_interval: int,
                         d_model: int,
                         q: int,
                         v: int,
                         h: int,
                         N: int,
                         dropout: float,
                         DATA_LEN: int,
                         BATCH_SIZE: int,
                         time_cost: float,
                         EPOCH: int,
                         draw_key: int,
                         reslut_figure_path: str,
                         optimizer_name: str,
                         file_name: str,
                         LR: float,
                         pe: bool,
                         mask: bool):
    my_font = fp(fname=r"font/simsun.ttc")  # 2、设置字体路径

    # 设置风格
    plt.style.use('seaborn')

    fig = plt.figure()  # 创建基础图
    ax1 = fig.add_subplot(311)  # 创建两个子图
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)  # 添加折线
    ax2.plot(correct_on_test, color='red', label='on Test Dataset')
    ax2.plot(correct_on_train, color='blue', label='on Train Dataset')

    # 设置坐标轴标签 和 图的标题
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{test_interval}')
    ax2.set_ylabel('correct')
    ax1.set_title('LOSS')
    ax2.set_title('CORRECT')

    plt.legend(loc='best')

    # 设置文本
    fig.text(x=0.13, y=0.4, s=f'最小loss：{min(loss_list)}' '    '
                              f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}' '    '
                              f'最后一轮loss:{loss_list[-1]}' '\n'
                              f'最大correct：测试集:{max(correct_on_test)}% 训练集:{max(correct_on_train)}%' '    '
                              f'最大correct对应的已训练epoch数:{(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}' '    '
                              f'最后一轮correct：{correct_on_test[-1]}%' '\n'
                              f'd_model={d_model}   q={q}   v={v}   h={h}   N={N}  drop_out={dropout}'  '\n'
                              f'共耗时{round(time_cost, 2)}s', fontproperties=my_font)

    # 保存结果图   测试不保存图（epoch少于draw_key）
    if EPOCH >= draw_key:
        plt.savefig(
            f'{reslut_figure_path}/{file_name} {max(correct_on_test)}% {optimizer_name} epoch={EPOCH} batch={BATCH_SIZE} lr={LR} pe={pe} mask={mask} [{d_model},{q},{v},{h},{N},{dropout}].png')

    # 展示图
    plt.show()

    print('正确率列表', correct_on_test)

    print(f'最小loss：{min(loss_list)}\r\n'
          f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}\r\n'
          f'最后一轮loss:{loss_list[-1]}\r\n')

    print(f'最大correct：测试集:{max(correct_on_test)}\t 训练集:{max(correct_on_train)}\r\n'
          f'最correct对应的已训练epoch数:{(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}\r\n'
          f'最后一轮correct:{correct_on_test[-1]}')

    print(f'共耗时{round(time_cost, 2)} min')


def plot_result(loss_list, correct_on_test, correct_on_train, test_interval, reslut_figure_path, file_name):
    plt.clf()
    loss_color = (0.01, 0.72, 0.77)
    test_acc_color = (0.99, 0.49, 0.00)
    best_test_acc = max(correct_on_test)
    best_test_acc_idx = correct_on_test.index(best_test_acc)

    fig = plt.figure()
    fig.suptitle('Brain4car')
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(loss_list, color=loss_color, label='Train loss')
    ax2.plot(correct_on_test, color=test_acc_color, label='Test acc')
    ax2.plot(correct_on_train, color='blue', label='Train acc')
    ax2.scatter(best_test_acc_idx, best_test_acc, color='red', marker='o')
    ax2.axhline(y=best_test_acc, color='gray', linestyle='--')
    ax2.text(-55, best_test_acc, f'{best_test_acc:.2f}', color='red' ,ha='right', va='center')
    # ax1.set_xlabel('Global_round')
    # 设置坐标轴标签 和 图的标题
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{test_interval}')
    ax2.set_ylabel('correct')
    ax2.set_ylabel('accuracy')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'{reslut_figure_path}/accuracy_loss/{file_name} {time_now} {max(correct_on_test)}.png')
    # plt.show()

def plot_confusion_matrix(confusionMat, classes, reslut_figure_path, file_name):
    plt.clf()
    logger.info(f"{file_name}混淆矩阵结果：")
    logger.info(confusionMat)
    # plt.figure()
    plt.imshow(confusionMat, interpolation='none', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(classes)-0.5, -0.5)
    fmt = '.2f'
    thresh = confusionMat.max() / 2
    for i in range(confusionMat.shape[0]):
        for j in range(confusionMat.shape[1]):
            plt.text(j, i, format(confusionMat[i, j], fmt), horizontalalignment="center",
                     color="white" if confusionMat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("Predicted Label")
    plt.xlabel("True label")
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"{reslut_figure_path}/confMat/{file_name} {time_now} .png")
    # plt.show()

def plot_intermediate_result(x, action, save_folder):
    plt.clf()
    action_list = ["end_action", "lchange", "lturn", "rchange", "rturn"]
    x = np.array(x)
    x[x == 0] = np.nan
    timestep, channels = x.shape
    plt.close('current')
    fig, axes = plt.subplots(channels, 1, figsize=(16, 2*channels), sharex=True)
    if channels == 1:
        axes = [axes]
    for channel in range(channels):
        axes[channel].plot(range(timestep), x[:, channel])
        axes[channel].set_ylabel(f"channel {channel + 1}")
        axes[channel].grid(True)
    axes[channels - 1].set_xlabel("Timestep")
    plt.tight_layout()
    idx = 0
    for i in range(len(action_list)):
        if action_list[i] in action:
            idx = i
            break
    file_name = action.split(os.sep)[-1]
    file_name = file_name.split('.')[0] + '.jpg'
    plt.savefig(os.path.join(save_folder, action_list[idx], file_name))
    
def plot_eyegaze_intermediate_result(origin_data, filter_data, action,save_folder):
    plt.clf()
    action_list = ["end_action", "lchange", "lturn", "rchange", "rturn"]
    plt.subplot(1,2,1)
    plt.title("Before filter")
    plt.plot(range(origin_data.shape[0]), origin_data)
    plt.plot(range(filter_data.shape[0]), filter_data, 'r')
    plt.subplot(1, 2, 2)
    plt.title("After filter")
    idx = 0
    for i in range(len(action_list)):
        if action_list[i] in action:
            idx = i
            break
    file_name = action.split(os.sep)[-1]
    file_name = file_name.split('.')[0] + '.jpg'
    
    plt.savefig(os.path.join(save_folder, action_list[idx], file_name))

def barplot(dataset_path):
    activity_list = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']
    data = [[] for i in range(5)]
    with open(dataset_path, 'r') as f:
        dataset_dict = json.load(f)
    person_keys = list(dataset_dict.keys())
    activity_keys = list(dataset_dict['person1']['face_camera'].keys())
    for i, pk in enumerate(person_keys):
        for j, ak in enumerate(activity_keys):
            activity_len = len(dataset_dict[pk]['face_camera'][ak])
            data[j].append(activity_len)
    data_np = np.array(data)
    # plot堆叠柱状图
    for idx, row in enumerate(data_np):
        plt.bar(range(len(row)), row, bottom=data_np[:idx].sum(axis=0), label=f"{activity_list[idx]}")
    plt.xlabel('Person')
    plt.ylabel('Data Length')
    plt.title("Brain4cars Dataset")
    plt.legend()
    plt.savefig('./brain4cars_dataset.png')
    plt.show()

# 绘制matlab compass图
def compass(angles, radii, arrowprops=None):
    """
    Compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.
    """
    # angles, radii = cart2pol(u, v)
    # angles, radii = pol2cart(u, v)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    kw = dict(arrowstyle="->", color='b')
    if arrowprops:
        kw.update(arrowprops)
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0),
                 arrowprops=kw) for
     angle, radius in zip(angles, radii)]

    ax.set_ylim(0, np.max(radii))

    return fig, ax


if __name__ == "__main__":
      cnf_matrix = np.array([[8707, 64, 731, 164, 45],
                              [1821, 5530, 79, 0, 28],
                              [266, 167, 1982, 4, 2],
                              [691, 0, 107, 1930, 26],
                              [30, 0, 111, 17, 42]])
      cnf_matrix = np.array([[0.74418605, 0.09302326, 0,         0.09302326, 0.06976744],
                             [0,         0.83333333, 0,         0,         0.16666667],
                             [0,         0,         1,         0,         0,        ],
                             [0.15384615, 0,         0.15384615, 0.69230769, 0,        ],
                             [0,         0,         0,         0,         1,        ]])
      action_name = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']
      reslut_figure_path = 'result_figure'
      plot_confusion_matrix(cnf_matrix, action_name, reslut_figure_path, "test")

