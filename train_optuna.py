# @Time    : 2023/11/22 
# @Author  : Shuaijie Zhao
# @FileName: train.py
import os
import sys
import random
import argparse
import logging
import time
from tqdm import tqdm
from art import text2art

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn import metrics
import optuna
from optuna.trial import TrialState

from dataset_process.dataset_process import MyDataset
from dataset_process.brain4cars_dataset import Brain4carDataset
from module.transformer import Transformer
from module.transformer_multilabel import Transformer_multilabel
from module.convtran import ConvTran
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization, plot_confusion_matrix, plot_result
from utils.predictions import predictManeuver, confusionMat
from utils.model_utils import count_parameters

logger = logging.getLogger(__name__)

# setup_seed(16)

def set_logging():
    logging.basicConfig(filename='runs/results.log' ,format="%(message)s",level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default="./brain4cars_pipeline/temp_data/folder5/brain4cars_train_dataset_mediapipe_random4.json", help="train dataset path")
    parser.add_argument('--test_path', type=str, default="./brain4cars_pipeline/temp_data/folder5/brain4cars_valid_dataset_mediapipe_random4.json", help="train dataset path")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32, help="total batch size for GPUs")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate 0.00017358693285391162")
    parser.add_argument('--optimizer', type=str, default="Adagrad", help="optimizer for back propogation")
    parser.add_argument('--device', type=str, default="0", help="cuda device, i.e. 0,1,2,3 or cpu")
    parser.add_argument('--model', type=str, default='ConvTran', help="model name")
    parser.add_argument('--d-model', type=int, default=512, help="embedding dimension")
    parser.add_argument('--d-hidden', type=int, default=1024, help="fully connected layer hidden dimension")
    parser.add_argument('--q', type=int, default=8, help="querry ")
    parser.add_argument('--v', type=int, default=8, help="value")
    parser.add_argument('--h', type=int, default=8, help="n heads")
    parser.add_argument('--N', type=int, default=4, help="n encoders")
    parser.add_argument('--dropout', type=float, default=0.01, help="0.1")
    parser.add_argument('--pe', type=bool, default=True, help=" positional embefding")
    parser.add_argument('--mask', type=bool, default=True, help="mask for step encoder")
    parser.add_argument('--reslut_figure_path', type=str, default="result_figure", help="path for saving figures")
    parser.add_argument('--filename', type=str, default="brain4cars", help="figure prefix")
    parser.add_argument('--logdir', type=str, default="runs/", help="logging directory")
    parser.add_argument('--kfolder_path', type=str, default="D:/ML/Code_project/brain4cars_video/brain4cars_pipeline/temp_data/", help="k folder cross validation path")
    parser.add_argument('--kfolder_filename', type=str, default="brain4cars_train_dataset_mediapipe_random5.pik")
    parser.add_argument('--test-interval', type=int, default=1, help="after test interval epochs training, test")
    parser.add_argument('--draw-key', type=int, default=1, help="1")
    parser.add_argument('--random-seed', type=int, default=1234)
    
    opt = parser.parse_args()
    return opt

def test(net, dataloader, DEVICE, flag='test_set'):
    action_name = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']
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
            x = x.permute(0, 2, 1)
            # obtain the final truth
            # y_pre, _, _, _, _, _, _ = net(x, 'test')
            y_pre = net(x)
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
        # sklearn_confMatrix = metrics.confusion_matrix(np.array(labels), np.array(prediction_list))
        sklearn_precision, sklearn_recall, sklearn_f1, sklearn_support = metrics.precision_recall_fscore_support(np.array(labels), np.array(prediction_list), zero_division=1)
        # metrics_report = metrics.classification_report(np.array(labels), np.array(prediction_list), target_names=action_name)
        # logger.info(metrics_report)
        avg_precision = np.mean(np.diag(precision_confMat)[1:])
        lane_changing_precision = np.mean(np.diag(precision_confMat)[1::2])
        turns_precision = np.mean(np.diag(precision_confMat)[2::2])
        avg_recall = np.mean(np.diag(recall_confMat)[1:])
        avg_f1 = np.mean(sklearn_f1[1:])
        avg_anticipation_time = np.mean(np.divide(np.diag(TimeMat)[1:], np.diag(confMat)[1:]))
        # if flag == 'test_set':
        #     logger.info(f'All maneuver Precision on {flag}: %.2f %%' % (100 * avg_precision))
        #     logger.info(f'Lane changing maneuver Precision on {flag}: %.2f %%' % (100 * lane_changing_precision))
        #     logger.info(f'Turn maneuver Precision on {flag}: %.2f %%' % (100 * turns_precision))
        #     logger.info(f'All maneuver Recall on {flag}: %.2f %%' % (100 * avg_recall))
        #     logger.info(f'All maneuver F1 on {flag}: %.2f %%' % (100 * avg_f1))
        #     logger.info(f'Anticipation time on {flag}: %.2f ' % (avg_anticipation_time))
            

    return round((100 * avg_precision), 2), confMat, precision_confMat, recall_confMat, avg_anticipation_time, avg_f1

def train(opt, DEVICE):
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S")
    # logger.info(time_now)
    # train config
    EPOCH, BATCH_SIZE, LR, dropout, optimizer_name = opt.epochs, opt.batch_size, opt.lr, opt.dropout, opt.optimizer
    # model config
    d_model, d_hidden, q, v, h, N, pe, mask = opt.d_model, opt.d_hidden, opt.q, opt.v, opt.h, opt.N, opt.pe, opt.mask
    # path config
    brain4car_train_path, brain4car_test_path, reslut_figure_path, file_name = opt.train_path, opt.test_path, opt.reslut_figure_path, opt.filename
    # other config
    random_seed, test_interval, draw_key = opt.random_seed, opt.test_interval, opt.draw_key
    # setup_seed(random_seed)

    # dataset setup
    train_dataset = Brain4carDataset(brain4car_train_path)
    test_dataset = Brain4carDataset(brain4car_test_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    DATA_LEN = train_dataset.dataset_len  # 训练集样本数量
    d_input = train_dataset.input_len  # 时间部数量
    d_channel = train_dataset.channel_len  # 时间序列维度
    d_output = train_dataset.output_len  # 分类类别
    # logger.info('data structure: [lines, timesteps, features]')
    # logger.info(f'train data size: [{DATA_LEN, d_input, d_channel}]')
    # logger.info(f'mytest data size: [{test_dataset.dataset_len, d_input, d_channel}]')
    # logger.info(f'Number of classes: {d_output}')
    action_name = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']
    # Model
    if opt.model == 'GTN':
        net = Transformer_multilabel(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                    q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
    elif opt.model == 'ConvTran':
        net = ConvTran(d_model=d_model, d_input=d_input, d_channel=d_channel, d_hidden=256,  heads=h, d_dropout=dropout, num_classes=d_output).to(DEVICE)
    
    # logger.info("Total number of parameters: {}".format(count_parameters(net)))
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

    net.train()
    max_accuracy = 0
    best_confMat, best_precision_confMat, best_recall_confMat, best_f1 = None, None, None, None
    pbar = tqdm(range(EPOCH))
    begin = time.time()
    print(('\n' + '%15s'*3) %("Global Round", "Gpu mem", "Train loss"))
    for index in pbar:
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.permute(0, 2, 1)
            net.train()
            optimizer.zero_grad()
            # y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')
            y_pre = net(x)
            lane_info = x[:, -4:-1, -1]
            lane_info_com = x[:, -4:-1, -2]
            if not torch.equal(lane_info, lane_info_com):
                raise ValueError("Lane info confusion!")
            loss = loss_function(y_pre, y, lane_info)
            mem = '%.3gG'%(torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0)
            str_loss = '%10.4g' %(loss.item())
            s = ('%15s'*3) %('%g/%g' %(index + 1, EPOCH), mem, str_loss)
            pbar.set_description(s)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        if ((index + 1) % test_interval) == 0:
            current_precision_test, current_confMat_test, current_precision_confMat_test, current_recall_confMat_test, current_anticipation_time_test, avg_f1 = test(net, test_dataloader, DEVICE)
            current_accuracy_train, _, _, _, _, _ = test(net, train_dataloader, DEVICE, 'train_set')
            correct_on_train.append(current_accuracy_train)
            correct_on_test.append(current_precision_test)
            # logger.info(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            # logger.info(f'The Max precision on the testset: {max(correct_on_test)}%, on trainset:{max(correct_on_train)}%, learning rate: {LR}')

            if avg_f1 > max_accuracy:
                best_precision = current_precision_test
                best_confMat = current_confMat_test
                best_precision_confMat = current_precision_confMat_test
                best_recall_confMat = current_recall_confMat_test
                best_anticipation_time = current_anticipation_time_test
                best_f1 = avg_f1
                max_accuracy = avg_f1
                # torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')
        # scheduler.step()
        # LR = scheduler.get_last_lr()[0]
        # pbar.update()
    # os.rename(f'saved_model/{file_name} batch={BATCH_SIZE}.pkl',
    #           f'saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl')

    end = time.time()
    time_cost = round((end - begin) / 60, 2)

    # 结果图
    # result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
    #                      test_interval=test_interval,
    #                      d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
    #                      time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
    #                      file_name=file_name, optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask)
    # plot_result(loss_list, correct_on_test, correct_on_train, test_interval, reslut_figure_path, file_name)
    # plot_confusion_matrix(best_confMat, action_name, reslut_figure_path, "confMat")
    # plot_confusion_matrix(best_precision_confMat, action_name, reslut_figure_path, "precision")
    # plot_confusion_matrix(best_recall_confMat, action_name, reslut_figure_path, "recall")
    
    return best_precision, best_confMat, best_precision_confMat, best_recall_confMat, best_anticipation_time, best_f1


def kFolderValidation(opt, DEVICE, k=5):
    accuracy_list = []
    confMat_list = []
    precision_confMat_list = []
    avg_precision_list = []
    recall_confMat_list = []
    avg_recall_list = []
    anticipation_time_list = []
    f1_list = []
    folder_path = opt.kfolder_path
    random_seed = opt.random_seed
    jsonfile_name = opt.kfolder_filename
    setup_seed(random_seed)
    # k折交叉验证
    for i in range(k):
        # logger.info(text2art(f"Folder{i+1}", font="small"))
        # logger.info("-"*15+f"Folder{i+1} Start validation"+"-"*15)
        opt.train_path = os.path.join(folder_path, f"folder{i+1}", jsonfile_name)
        opt.test_path = os.path.join(folder_path, f"folder{i+1}", jsonfile_name.replace("train", "valid"))
        max_accuracy, best_confMat, best_precision_confMat, best_recall_confMat, best_anticipation_time, f1 = train(opt, DEVICE)
        accuracy_list.append(max_accuracy)
        confMat_list.append(best_confMat)
        precision_confMat_list.append(best_precision_confMat)
        avg_precision_list.append(np.mean(np.diag(best_precision_confMat)[1:]))
        recall_confMat_list.append(best_recall_confMat)
        avg_recall_list.append(np.mean(np.diag(best_recall_confMat)[1:]))
        anticipation_time_list.append(best_anticipation_time)
        f1_list.append(f1)
    
    accuracy_list = np.array(accuracy_list)
    precision_confMat_list = np.array(precision_confMat_list)
    avg_precision_list = np.array(avg_precision_list)
    recall_confMat_list = np.array(avg_recall_list)
    avg_recall_list = np.array(avg_recall_list)
    anticipation_time_list = np.array(anticipation_time_list)
    f1_list = np.array(f1_list)
    
    # logger.info("\n"+"-"*15+"Original Result"+"-"*15)
    # logger.info(f"precision list = {avg_precision_list}")
    # logger.info(f"recall list = {avg_recall_list}")
    # logger.info(f"F1 list = {f1_list}")
    # logger.info(f"Anticipation time list = {anticipation_time_list}")

    # logger.info("\n"+"-"*15+"Result"+"-"*15)
    # logger.info(f"Precision = {np.mean(avg_precision_list)} {np.std(avg_precision_list)}")
    # logger.info(f"Recall = {np.mean(avg_recall_list)} {np.std(avg_recall_list)}")
    logger.info(f"F1 = {np.mean(f1_list)} {np.std(f1_list)}")
    # logger.info(f"Anticipation time = {np.mean(anticipation_time_list)} {np.std(anticipation_time_list)}")
    f1_avg = np.mean(f1_list).item()
    return f1_avg

def objective(trial):
    opt = parse_args()
    f1 = 0.0
    # 设定要调整的参数
    opt.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    opt.bath_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    opt.d_model = trial.suggest_categorical("d_model", [16, 32, 64])
    # opt.h = trial.suggest_categorical("heads", [2, 4, 6, 8])
    opt.dropout = trial.suggest_float("dropout", 1e-3, 1e-2, step=0.001)
    opt.random_seed = trial.suggest_int("seed", 0, 2000)
    setup_seed(opt.random_seed)
    logger.info(opt)
    device = torch.device('cuda:0' if torch.cuda.is_available() and opt.device != "cpu" else "cpu")
    f1 = kFolderValidation(opt, device)
    return f1

if __name__ == '__main__':

    set_logging()
    # logger.info(opt)
    # device = torch.device('cuda:0' if torch.cuda.is_available() and opt.device != "cpu" else "cpu")
    # logging.info(f"Use DEVICE {device}")

    # train(opt, device)
    storage_name = "sqlite:///optuna_convtran.db"
    study = optuna.create_study(direction='maximize', study_name="OPTUNA_CONVTRAN", storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=2000)
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
