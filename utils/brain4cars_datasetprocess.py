#-*- coding : utf-8-*-
import os
import re
import sys
import time
sys.path.append("/home/dulab/ML/Code_project/")
sys.path.append('/home/dulab/ML/Code_project/GTN-master')
import csv
import tsaug
import random
import json
import shutil
import cv2
import pickle

import numpy as np
import pandas as pd
import mediapipe as mp

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from tsaug.visualization import plot
from utils.random_seed import setup_seed
from utils.visualization import plot_intermediate_result, barplot, compass, plot_eyegaze_intermediate_result
# eye gaze
import argparse
import pathlib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
from PIL import Image
from PIL import Image, ImageOps
from l2cs import select_device, draw_gaze, getArch, Pipeline, render
from face_detection import RetinaFace
CWD = pathlib.Path.cwd()

# 准备mediapipe facemesh api
# 可以参照：https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md
# static_image_mode True:适合静态图片，False：适合视频和动态图片
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                           max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.7)

"""
一些总体流程中用到的函数
"""

# eye gaze所用到的args
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='/home/dulab/ML/Code_project/L2CS-Net/models/L2CSNet_gaze360.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

"""----------------------数据过滤插值等处理函数--------------------"""
#插值函数 对缺失的人脸关键点进行插值处理
def interpolate_col(posex_list, posey_list):
    posex_array = np.array(posex_list)
    posey_array = np.array(posey_list)
    row, col = posex_array.shape
    start_idx = 0
    posex_array_copy = np.copy(posex_array)
    posey_array_copy = np.copy(posey_array)
    for i in range(row):
        if np.all(posex_array[i, :] == 0):
            start_idx += 1
        else:
            break
    
    # 对y坐标利用前面的值进行填充，对x坐标进行插值处理
    # 为什么对y进行前值填充呢？因为在头部转动到人脸检测不出来时候，一般来说是水平转动，因此假设y不变，仅x变化
    for idx in range(start_idx, row):
        if np.all(posey_array[idx, :] == 0):
            posey_array_copy[idx, :] = posey_array_copy[idx - 1, :]

    for j in range(col):
        column_x = posex_array[start_idx:, j]
        non_zero_indice = np.nonzero(column_x)[0]
        f_x = interp1d(non_zero_indice, column_x[non_zero_indice], kind="quadratic", fill_value="extrapolate")
        posex_array_copy[start_idx:, j] = f_x(np.arange(row - start_idx))

    return posex_array_copy, posey_array_copy
        
# 以matfile中frame的数量以及非0欧拉角占比的数量决定是否添加到数据集中
def filter_by_mat(thred, action):
    if 'face_camera' not in action and 'video_' not in action:
        print("Path Error!, Please check action path!")
        sys.exit(0)
    action = action.replace("face_camera", "new_params")
    action = action.replace("video_", "new_param_")
    mat_file = action.split('.')[0] + '.mat'
    try:
        m = loadmat(mat_file)
    except FileNotFoundError as e:
        print(e)
        sys.exit(0)
    # process mat file
    _, _, _, datakey = m
    Data = m[datakey]
    frame_data = Data['frame_data'].item()
    frame_strat = Data['frame_start'].item().item()
    frame_end = Data['frame_end'].item().item()
    total_valid_frame = frame_end - frame_strat + 1
    Euler_count = 0
    for i in range(frame_strat - 1, frame_end):
        keys = frame_data[0][i].dtype.names
        if 'Euler' in keys:
            Euler_count += 1

    return total_valid_frame, (total_valid_frame >= thred) and (Euler_count >= thred)

# 过滤掉时长少于thred秒的视频
def filter_by_video(thred, video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        video_length = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        # print(video_fps, end='  ')
        video_time = video_length / video_fps
        cap.release()
        return video_time, video_time > thred
    else:
        return None, False

# 对利用tsaug数据增强后的数据进行处理
def aug_filter(Euler_features_aug, Brain4cars_Face_aug, Brain4cars_Speed_aug, zero_rows_index):
    # 判断speed和face_features(hist部分)小于0的元素设置为0
    Brain4cars_Speed_aug[Brain4cars_Speed_aug < 0] = 0
    _, face_col = Brain4cars_Face_aug.shape
    Brain4cars_Face_aug[:, :face_col - 1][Brain4cars_Face_aug[:, :face_col - 1] < 0] = 0
    # 根据zero_row_index保证那些原本整行为0行在数据增强之后仍然为0
    if zero_rows_index.shape[0] != 0:
        for i in zero_rows_index:
            Euler_features_aug[i, :] = 0
            Brain4cars_Face_aug[i, :] = 0
            Brain4cars_Speed_aug[i,:] = 0
        return Euler_features_aug, Brain4cars_Face_aug, Brain4cars_Speed_aug
    else:
        return Euler_features_aug, Brain4cars_Face_aug, Brain4cars_Speed_aug

# eyegaze butterworth lowband path filter
def filter_eyedata(action, data=None, order=4, cutoff=1.667):
    if data is None:
        eyegaze_angle_path = action.replace("video_", "eyegaze_angles_")
        eyegaze_angle_path = eyegaze_angle_path.split('.')[0] + '.csv'
        df = pd.read_csv(eyegaze_angle_path)
        eyegaze_data = df.values
        dx = eyegaze_data[:, 0]
        dy = eyegaze_data[:, 1]
    else:
        eyegaze_data = data
        dx = data[:, 0]
        dy = data[:, 1]
    
    # 采样频率为30
    fs = 30
    normal_f = 2*np.array(cutoff) / fs
    # data shape: time x channel!
    b, a = butter(order, normal_f, btype="lowpass")
    filter_data_dx = filtfilt(b, a, dx).reshape(-1, 1)
    filter_data_dy = filtfilt(b , a, dy).reshape(-1, 1)
    filter_eyegaze_data = np.column_stack((filter_data_dx, filter_data_dy))
    
    return eyegaze_data, filter_eyegaze_data

# 对视觉注视进行二次提取，主要是除去哪些变化幅度较大的视觉方向
def eliminate_eyegaze_err(action):
    # 主要思想：假定相邻两帧之间的视觉注视之差不能大于1，因为视觉注视的最大值是1，差大于1即认为下一帧检测不准确
    eyegaze_angle_path = action.replace("video_", "eyegaze_angles_")
    eyegaze_angle_path = eyegaze_angle_path.split('.')[0] + '.csv'
    df = pd.read_csv(eyegaze_angle_path)
    eyegaze_data = df.values
    eyegaze_diff = np.diff(eyegaze_data, axis=0)
    # 获取差值大于1的mask
    eyegaze_diff_mask = np.abs(eyegaze_diff) > 1.0
    
    if np.all(eyegaze_diff_mask == 0):
        return False, eyegaze_data
    # eyegaze_diff_mask = eyegaze_diff_mask.astype(int)
    mask_indices = np.where(eyegaze_diff_mask)
    while True:
        if np.all(mask_indices):
            break
        idx = mask_indices[0][0]
        eyegaze_data[idx + 1, :] = eyegaze_data[idx, :]
        eyegaze_diff = np.diff(eyegaze_data, axis=0)
        eyegaze_diff_mask = np.abs(eyegaze_diff) > 1.0
        mask_indices = np.where(eyegaze_diff_mask)
        
    return True, eyegaze_data

def aggregate_features(aggregate_frame, Euler_features, Brain4cars_Face, Brain4cars_Speed):
    # Euler angle 7 x 3
    # Calculate the average of features from 20 frames. 获得每20帧数据取平均的部分
    Euler_features_20frame = np.zeros((7, Euler_features.shape[1]))
    # Face Features: 7 x 9
    Brain4cars_Face_20frame = np.zeros((7, Brain4cars_Face.shape[1]))
    # Speed: 7 x 3
    # Brain4cars_Speed = np.concatenate((Mean_speed, Max_speed, Min_speed), axis=1)
    Brain4cars_Speed_20frame = np.zeros((7,Brain4cars_Speed.shape[1]))


    aggregated = int(150 / aggregate_frame)
    for i in range(aggregated):
        start_idx = i * aggregate_frame + 10
        end_idx = start_idx + aggregate_frame
        # 求平均时要忽略那些为0的行！
        Euler_window_data = Euler_features[start_idx: end_idx]
        Face_window_data = Brain4cars_Face[start_idx: end_idx]
        Speed_window_data = Brain4cars_Speed[start_idx: end_idx]
        Euler_all_row_zero = np.all(Euler_window_data == 0, axis=1)
        Face_all_row_zero = np.all(Face_window_data == 0, axis=1)
        Speed_all_row_zero = np.all(Speed_window_data == 0, axis=1)

        # 这里为什么要写三个判断？按理来说欧拉角为0时整行都为0，但是有欧拉角不为0时，速度和脸部信息整行为0的情况
        # 写三个if为了防止出现对nan求平均得到nan
        if np.any(Euler_all_row_zero):
            # 全为0，以下这样写为了debug好看信息
            current_mean_euler_features = np.mean(Euler_window_data, axis=0).reshape((1, -1))
            Euler_features_20frame[i, :] = current_mean_euler_features
        else:
            Euler_non_zero_rows = Euler_window_data[~Euler_all_row_zero]
            Euler_features_20frame[i, :] = np.mean(Euler_non_zero_rows, axis=0).reshape((1, -1))
        if np.any(Face_all_row_zero):
            current_mean_face_features = np.mean(Face_window_data, axis=0).reshape((1, -1))
            Brain4cars_Face_20frame[i, :] = current_mean_face_features
        else:
            Face_non_zero_rows = Face_window_data[~Face_all_row_zero]
            Brain4cars_Face_20frame[i, :] = np.mean(Face_non_zero_rows, axis=0).reshape((1, -1))
        if np.any(Speed_all_row_zero):
            current_mean_speed_features = np.mean(Speed_window_data, axis=0).reshape((1, -1))
            Brain4cars_Speed_20frame[i, :] = current_mean_speed_features
        else:
            Speed_non_zero_rows = Speed_window_data[~Speed_all_row_zero]
            Brain4cars_Speed_20frame[i, :] = np.mean(Speed_non_zero_rows, axis=0).reshape((1, -1))
    return Euler_features_20frame, Brain4cars_Face_20frame, Brain4cars_Speed_20frame


"""---------------------csv文件处理---------------------"""
def action2csvpath(action, prefix=None):
    if prefix == None:
        csv_list = action.split(os.sep)
        csv_extension = csv_list[-1].replace('video_', '').replace('.avi', '.csv')
        csv_list[-1] = csv_extension
        csv_path = os.sep.join(csv_list)
    else:
        csv_list = action.split(os.sep)
        csv_extension = csv_list[-1].replace('video_', prefix + '_').replace('.avi', '.csv')
        csv_list[-1] = csv_extension
        csv_path = os.sep.join(csv_list)
    return csv_path

def ndarray2csvfile(csv_path, np_array, header):
    np.savetxt(csv_path, np_array, delimiter=',', header=','.join(header), comments='', fmt='%.8f')
    print(".csv save at {} successfully!".format(csv_path))

def clear_csv(dir_path, csv_name):
    for root, dir, files in sorted(os.walk(dir_path)):
        for file in files:
            if file.endswith('.csv'):
                if csv_name in file:
                    os.remove(os.path.join(root, file))
                    print("已删除文件：", os.path.join(root, file))

"""-------------------------特征提取函数------------------------"""
def getEyeGzae(action, gaze_pipeline, prefix="eyegaze_angles_"):
    csv_path = action.split('.')[0] + '.csv'
    csv_path = csv_path.split('video_')[0] + prefix + csv_path.split('video_')[1]
    if os.path.exists(csv_path):
        print(f"{csv_path} 文件已经存在，跳过生成!")
        return None, None
    try:
        video1 = cv2.VideoCapture(action)
    except FileNotFoundError:
        print("无法打开{}路径下文件,请检查路径！".format(action))
    video_width = video1.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = 0
    pitch_list = []
    yaw_list = []
    while True:
        ret, frame = video1.read()
        if not ret:
            break
        results = gaze_pipeline.step(frame)
        # frame = render(frame, results)
        if not np.any(results.yaw):
            results.pitch = [0]
            results.yaw = [0]
        pitch = results.pitch[0]
        yaw = results.yaw[0]
        dx = -1 * np.sin(pitch) * np.cos(yaw)
        dy = -1 * np.sin(pitch)
        pitch_list.append(dx)
        yaw_list.append(dy)
        frame_count += 1
        frame_new = cv2.resize(frame, (int(video_width / 2), int(video_height /2)))
        # cv2.imshow('face_mesh', frame_new)
        # key = cv2.waitKey(30) & 0xFF
        # if key == ord('q'):
        #     break
    video1.release()
    
    return np.array(pitch_list).reshape(-1, 1), np.array(yaw_list).reshape(-1, 1)

def get_landmarks(video_path, prefix="mediapipe_landmarks_"):
    """
    descriptions: 传入每个视频的地址返回视频的各帧图像的关键点坐标
    """
    # video_path = "/home/ubuntu/zsj/brain4cars_video/brain4cars_data/face_camera/end_action/20141019_132535_1548_1698/video_20141019_132535_1548_1698.avi"
    csv_path = video_path.split('.')[0] + '.csv'
    csv_path = csv_path.split('video_')[0] + prefix + csv_path.split('video_')[1]
    if os.path.exists(csv_path):
        print(f"{csv_path} 文件已经存在，跳过生成!")
        return None, None
    landmark_refine_list = [1,2,4,5,195,197,6,168,8,9,151,10,
                            70, 63, 105, 66, 107,46,53,52,65,55,
                            336,296,334,293,300,285,295,282,283,276,
                            113,225,224,223,222,221,189,247,30,29,27,28,56,190,
                            31,228,229,230,231,232,233,
                            413,441,442,443,444,445,342,414,286,258,257,259,260,467,
                            453,452,451,450,449,448,261,
                            26,121,47,126,209,49,
                            256,350,277,355,429,279,
                            48,115,220,45,275,440,344,278,
                            123,50,101,100,
                            352,280,330,329]
    
    x_list, y_list = [], []
    prev_x, prev_y = [0 for i in range(len(landmark_refine_list))], [0 for i in range(len(landmark_refine_list))]
    action = os.path.normpath(video_path)
    # video properties
    try:
        video = cv2.VideoCapture(action)
    except FileNotFoundError:
        print("无法打开{}路径下文件,请检查路径！".format(action))
    video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_length = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print("The current video path is {}".format(video_path))
    print("The current video length is {}".format(video_length))
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # 获取视频的总长度便于后续padding
        # 对视频进行裁减，聚焦于驾驶员，让人脸关键点检测更加准确
        frame_new = frame[:, 350:]
        height, width, _ = frame_new.shape
        img_RGB = cv2.cvtColor(frame_new, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_RGB)
        x_per_frame, y_per_frame = [], []
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            for l_idx, landmark in enumerate(face_landmarks.landmark): 
                lx, ly = landmark.x * width, landmark.y * height
                if l_idx in landmark_refine_list:
                    x_per_frame.append(lx+350)
                    y_per_frame.append(ly)
                    cv2.circle(frame, (int(lx+350), int(ly)), 2, (0, 255, 0), -5)
            # prev_x, prev_y = x_per_frame, y_per_frame
            x_list.append(x_per_frame), y_list.append(y_per_frame)
        else:
            x_list.append(prev_x)
            y_list.append(prev_y)
        frame_count += 1
        frame = cv2.resize(frame, (int(video_width / 2), int(video_height / 2)))
    #     cv2.imshow('face_mesh', frame)
    #     key = cv2.waitKey(30) & 0xFF
    #     if key == ord('q'):
    #         break
    video.release()
    x_array, y_array = interpolate_col(x_list, y_list)
    # return np.array(x_list), np.array(y_list)
    return x_array, y_array

def landmarks_writer(action, posex_list,posey_list, prefix):
    """
    pos_x_array shape: [video_length, num of landmarks]
    pos_y_array shape: [video_length, num of landmarks]
    """
    pos_x_array = np.array(posex_list)
    pos_y_array = np.array(posey_list)
    landmark_num = pos_x_array.shape[1]
    print("The shapes of landmarks x and landmarks y are {}, {}".format(pos_x_array.shape, pos_y_array.shape))
    csv_path = action.split('.')[0] + '.csv'
    csv_path = csv_path.split('video_')[0] + prefix + csv_path.split('video_')[1]
    x_is_empty = not np.any(pos_x_array)
    y_is_empty = not np.any(pos_y_array)
    if x_is_empty or y_is_empty:
        sys.exit("关键点x或关键点y数组为空!")
    x_videolength = pos_x_array.shape[0]
    y_videolength = pos_y_array.shape[0]
    if x_videolength != y_videolength:
        sys.exit("关键点坐标x的行数不等于坐标y的行数!")
    landmarks_csv = open(csv_path, 'a', encoding='utf-8', newline='')
    writer = csv.writer(landmarks_csv)
    header = []
    for index in range(pos_x_array.shape[1]):
        header.append(f'x_{index}')
        header.append(f'y_{index}')
    pos_xy_array = np.column_stack((pos_x_array.flatten(), pos_y_array.flatten()))
    pos_xy_array = pos_xy_array.reshape((x_videolength, landmark_num*2))
    ndarray2csvfile(csv_path, pos_xy_array, header)
    np.savetxt(csv_path, pos_xy_array, delimiter=',', header=','.join(header), comments='', fmt='%.8f')
    print(".csv save at {} successfully!".format(csv_path))

def EyeFeaturesExtract(action):
    """
    descriptions: 传入视频地址, 转为csv地址再进行关键点处理
    """
    # action = "/home/ubuntu/zsj/brain4cars_video/brain4cars_data/face_camera/lturn/20141019_091035_1542_1689/video_20141019_091035_1542_1689.avi"
    hist_angle_values = [-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi]
    hist_distance_x = [-1e3, -2.0, 0, 2.0, 1e3]
    hist_distance_values = [-1e3, 2, 5, 8, 10, 1e3]
    print(action)
    csv_path = action.split('.')[0] + '.csv'
    csv_path = csv_path.split('video_')[0] + "mediapipe_landmarks_" + csv_path.split('video_')[1]
    df = pd.read_csv(csv_path)
    landmarks = df.values[:, 946:] / 2
    points_move_vec = np.diff(landmarks, axis=0)
    # 限制最大值
    points_move_vec[np.abs(points_move_vec) > 20] = 0
    frame = points_move_vec.shape[0]
    target_shape = (150, 10)
    if frame < 150:
        pad_with = [(target_shape[0]- frame, 0), (0, 0)]
        points_move_vec = np.pad(points_move_vec, pad_with, mode='constant', constant_values=0)
    
    points_move_in_x = points_move_vec[:, 0::2]
    points_move_in_y = points_move_vec[:, 1::2]
    points_distance = np.sqrt(points_move_in_x **2 + points_move_in_y**2)
    points_angle = np.arctan2(points_move_in_y, points_move_in_x)
    # histogram
    features_hist_angle = np.empty((points_angle.shape[0], 4))
    features_hist_move_in_x = np.empty((points_angle.shape[0], 4))
    features_hist_distance = np.empty((points_angle.shape[0], 5))
    for i in range(points_angle.shape[0]):
        features_hist_angle[i, :], angle_bins = np.histogram(points_angle[i], hist_angle_values)
        features_hist_distance[i, :], distance_bins = np.histogram(points_distance[i], hist_distance_values)
        features_hist_move_in_x[i, :], move_in_x_bins = np.histogram(points_move_in_x[i], hist_distance_x)
    # shape: 150x1
    features_mean_movement = np.mean(points_distance, axis=1).reshape((points_distance.shape[0], 1))
    features_mean_movement_x = np.mean(points_move_in_x, axis=1).reshape((points_distance.shape[0], 1))
    # shape: 150x4
    """feature norms: angle"""
    features_hist_angle_2norms = np.linalg.norm(features_hist_angle, axis=1, keepdims=True)
    features_hist_angle_2norms[features_hist_angle_2norms < 1] = 1
    features_hist_angle_normalize = features_hist_angle / features_hist_angle_2norms
    # shape: 150x4
    """feature norms: move_in_x"""
    features_hist_move_in_x_2norms = np.linalg.norm(features_hist_move_in_x, axis=1, keepdims=True)
    features_hist_move_in_x_2norms[features_hist_move_in_x_2norms < 1] = 1
    features_hist_move_in_x_normalize = features_hist_move_in_x / features_hist_move_in_x_2norms
    
    # all eye features shape: 150x9
    all_eye_features = np.concatenate((features_hist_angle_normalize, features_hist_move_in_x_normalize, features_mean_movement_x), axis=1)
    all_eye_features_Qunatize = tsaug.Quantize(n_levels=4).augment(all_eye_features.transpose((1, 0))).transpose((1,0))
    all_eye_features_Convolve = tsaug.Convolve(window="flattop", size=11).augment(all_eye_features_Qunatize.transpose((1, 0))).transpose((1,0))
    plot(all_eye_features.transpose(1,0))
    plot(all_eye_features_Convolve.transpose(1,0))
    """
    画图显示结果, 可以解除comment
    """
    # plt.hist(features_hist_angle[5, :], 4)
    # plt.show()
    # aaa = points_angle[28, :]
    # bbb = points_distance[28, :]
    # fig, ax = compass(points_angle[91, :], points_distance[91, :])
    # plt.show()

    return all_eye_features

def Brain4carFeatureExtract(action):
    """
    descriptions: 传入视频地址, 转为mat地址再进行除眼部之外的人脸关键点处理
    """
    if 'face_camera' not in action and 'video_' not in action:
        print("Path Error!, Please check action path!")
        sys.exit(0)
    # action = "/home/dulab/ML/Code_project/brain4cars_video/brain4cars_data/face_camera/rturn/20141115_101346_1_131/video_20141115_101346_1_131.avi"
    action_new = action
    action_new = action_new.replace("face_camera", "new_params")
    action_new = action_new.replace("video_", "new_param_")
    mat_file = action_new.split('.')[0] + '.mat'
    mediapipe_landmarks_path = action.replace("video_", "mediapipe_landmarks_")
    mediapipe_landmarks_path = mediapipe_landmarks_path.split('.')[0] + '.csv'
    eyegaze_angle_path = action.replace("video_", "eyegaze_angles_")
    eyegaze_angle_path = eyegaze_angle_path.split('.')[0] + '.csv'
    try:
        m = loadmat(mat_file)
    except FileNotFoundError as e:
        print(e)
        sys.exit(0)
    try:
        df_mediapipe = pd.read_csv(mediapipe_landmarks_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(0)
    try:
        df_eyegaze = pd.read_csv(eyegaze_angle_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(0)
           
    # process mat file 获取原始数据部分
    _, _, _, datakey = m
    Data = m[datakey]
    lane_info = Data['laneInfo'].item().item().split(',')
    lane_info = np.array([int(i) for i in lane_info])
    frame_data = Data['frame_data'].item()
    frame_strat = Data['frame_start'].item().item()
    frame_end = Data['frame_end'].item().item()
    # 处理csv file miedapiep_landmars形状是[video_length, 2*num of point] -->[x1,y1,x2,y2,.....]
    mediapipe_landmarks_data = df_mediapipe.values
    mediapipe_landmark_nums = mediapipe_landmarks_data.shape[-1]
    _, filter_eyegazedata = filter_eyedata(action, df_eyegaze.values)
    eyegaze_data_channels = filter_eyegazedata.shape[-1]
    # total_valid_frame = frame_end - frame_strat + 1
    # 保证形状是151
    landmarks = np.zeros((151, 68 * 2))
    mediapipe_landmarks = np.zeros((151, mediapipe_landmark_nums))
    eyegaze_data = np.zeros((151, eyegaze_data_channels))
    Euler = np.zeros((151, 3))
    Speed = np.zeros((151, 1))
    for i in range(frame_strat - 1, frame_end):
        keys = frame_data[0][i].dtype.names
        klt_points_per_frame = frame_data[0][i]['klt_points'].item()
        # reshape klts:[[x1,x2,x3,...]                ---> [x1,y1,x2,y2,.....]  shape:[1, 136]
        #                [y1,y2,y3,...]] shape:[2, 68]
        klt_points_per_frame = klt_points_per_frame.reshape((1, -1), order='F')
        mediapipe_points_per_frame = mediapipe_landmarks_data[i, :]
        eyegaze_data_per_frame = filter_eyegazedata[i, :]
        if klt_points_per_frame.size == 0:
            klt_points_per_frame = np.zeros((1, 68 * 2))
        if mediapipe_points_per_frame.size == 0:
            mediapipe_points_per_frame = np.zeros((1, mediapipe_landmark_nums))
        if eyegaze_data_per_frame.size == 0:
            eyegaze_data_per_frame = np.zeros((1, eyegaze_data_channels))
        speed_per_frame = frame_data[0][i]['speed'].item().item()
        if 'Euler' in keys:
            Euler_per_frame = frame_data[0][i]['Euler'].item().reshape(1,3)
        else:
            Euler_per_frame = np.zeros((1, 3))

        landmarks[151 - frame_end + i, :] = klt_points_per_frame
        Speed[151 - frame_end + i, :] = speed_per_frame
        Euler[151- frame_end + i, :] = Euler_per_frame
        mediapipe_landmarks[151 - frame_end + i, :] = mediapipe_points_per_frame
        eyegaze_data[151 - frame_end + i, :] = eyegaze_data_per_frame

    # extract face features 获取特征部分
    hist_angle_values = [-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi]
    # hist_distance_x = [-1e3, -2.0, 0, 2.0, 1e3]
    hist_distance_x = [-1e3, -5.0, -2.5, 0, 2.5, 5.0, 1e3]
    hist_distance_values = [-1e3, 2, 5, 8, 10, 1e3]
    hist_eye_distance = [-1.0, -0.5, 0, 0.5, 1.0]
    
    points_in_x = (np.mean(landmarks[:, 0::2], axis=1) / 1920).reshape(-1, 1)
    mediapipe_points_in_x = (np.mean(mediapipe_landmarks[:, 0::2], axis=1) / 1920).reshape(-1, 1)
    # plot_intermediate_result(mediapipe_points_in_x, action, "./result_figure/intermediate_results/point_12_in_x")
    # plot(points_in_x.transpose((1, 0)))
    # plot(mediapipe_points_in_x.transpose((1, 0)))
    # landmarks: 151 rows -> 150 rows
    points_move_vec = np.diff(landmarks, axis=0)
    points_move_vec[np.abs(points_move_vec) > 30] = 0
    points_move_in_x = points_move_vec[:, 0::2]
    points_move_in_y = points_move_vec[:, 1::2]
    points_distance = np.sqrt(points_move_in_x **2 + points_move_in_y**2)
    points_angle = np.arctan2(points_move_in_y, points_move_in_x)
    mediapipe_points_move_vec = np.diff(mediapipe_landmarks, axis=0)
    mediapipe_points_move_vec[np.abs(mediapipe_points_move_vec) > 30] = 0
    mediapipe_points_move_in_x = mediapipe_points_move_vec[:, 0::2]
    mediapipe_points_move_in_y = mediapipe_points_move_vec[:, 1::2]
    mediapipe_points_distance = np.sqrt(mediapipe_points_move_in_x **2 + mediapipe_points_move_in_y**2)
    mediapipe_points_angle = np.arctan2(mediapipe_points_move_in_y, mediapipe_points_move_in_x)
    eyegaze_pitch = eyegaze_data[:, 0]
    eyegaze_yaw = eyegaze_data[:, 1]
    
    # histogram
    # original
    features_hist_angle = np.zeros((points_angle.shape[0], len(hist_angle_values) - 1))
    features_hist_move_in_x = np.zeros((points_angle.shape[0], len(hist_distance_x) - 1))
    features_hist_distance = np.zeros((points_angle.shape[0], len(hist_distance_values) - 1))
    # mediapipe
    features_hist_mediapipe_angle = np.zeros((points_angle.shape[0], len(hist_angle_values) - 1))
    features_hist_mediapipe_move_in_x = np.zeros((points_angle.shape[0], len(hist_distance_x) - 1))
    features_hist_mediapipe_distance = np.zeros((points_angle.shape[0], len(hist_distance_values) - 1))
    # eyegaze
    features_hist_eyegaze_move_in_x = np.zeros((points_angle.shape[0], len(hist_eye_distance) - 1))
    features_hist_eyegaze_move_in_y = np.zeros((points_angle.shape[0], len(hist_eye_distance) - 1))
    
    for i in range(points_angle.shape[0]):
        features_hist_angle[i, :], angle_bins = np.histogram(points_angle[i], hist_angle_values)
        features_hist_distance[i, :], distance_bins = np.histogram(points_distance[i], hist_distance_values)
        features_hist_move_in_x[i, :], move_in_x_bins = np.histogram(points_move_in_x[i], hist_distance_x)
        features_hist_mediapipe_angle[i, :], _ = np.histogram(mediapipe_points_angle[i], hist_angle_values)
        features_hist_mediapipe_distance[i, :], _ = np.histogram(mediapipe_points_distance[i], hist_distance_values)
        features_hist_mediapipe_move_in_x[i, :], _ = np.histogram(mediapipe_points_move_in_x[i], hist_distance_x)
        features_hist_eyegaze_move_in_x[i, :], _ = np.histogram(eyegaze_pitch[i], hist_eye_distance)
        features_hist_eyegaze_move_in_y[i, :], _ = np.histogram(eyegaze_yaw[i], hist_eye_distance)
        
    # shape: total_valid_frame x 1
    features_mean_movement = np.mean(points_distance, axis=1).reshape((points_distance.shape[0], 1))
    features_mean_movement_x = np.mean(points_move_in_x, axis=1).reshape((points_distance.shape[0], 1))
    features_mediapipe_mean_movement = np.mean(mediapipe_points_distance, axis=1).reshape((mediapipe_points_distance.shape[0], 1))
    features_mediapipe_mean_movement_x = np.mean(mediapipe_points_move_in_x, axis=1).reshape((mediapipe_points_move_in_x.shape[0], 1))
    # shape: total_valid_frame x 4
    """feature norms: angle"""
    features_hist_angle_2norms = np.linalg.norm(features_hist_angle, axis=1, keepdims=True)
    features_hist_angle_2norms[features_hist_angle_2norms < 1] = 1
    features_hist_angle_normalize = features_hist_angle / features_hist_angle_2norms
    features_hist_mediapipe_angle_2norms = np.linalg.norm(features_hist_mediapipe_angle, axis=1, keepdims=True)
    features_hist_mediapipe_angle_2norms[features_hist_mediapipe_angle_2norms < 1] = 1
    features_hist_mediapipe_angle_normalize = features_hist_mediapipe_angle / features_hist_mediapipe_angle_2norms
    # shape: total_valid_frame x 4 or 5
    """feature norms: move_in_x"""
    features_hist_move_in_x_2norms = np.linalg.norm(features_hist_move_in_x, axis=1, keepdims=True)
    features_hist_move_in_x_2norms[features_hist_move_in_x_2norms < 1] = 1
    features_hist_move_in_x_normalize = features_hist_move_in_x / features_hist_move_in_x_2norms
    feature_hist_mediapipe_move_in_x_2norms = np.linalg.norm(features_hist_mediapipe_move_in_x, axis=1, keepdims=True)
    feature_hist_mediapipe_move_in_x_2norms[feature_hist_mediapipe_move_in_x_2norms < 1] = 1
    features_hist_mediapipe_move_in_x_normalize = features_hist_mediapipe_move_in_x / feature_hist_mediapipe_move_in_x_2norms
    
    """Speed features 这里和论文中的不一样，这里的原则是每隔获取最大速度，而不是5s中共用一个最大速度"""
    num_intervals = int(Speed.shape[0] / 30)
    # 获取平均着histogram 用于打印展示判断histogram间隔是否正确！
    mean_hist_move_in_x_normalize = np.mean(features_hist_move_in_x_normalize, axis=0).reshape(1, -1)
    mean_hist_mediapipe_move_in_x_normalize = np.mean(features_hist_mediapipe_move_in_x_normalize, axis=0).reshape(1, -1)
    # Speed: 151 rows 
    Mean_speed = np.zeros_like(Speed)
    Max_speed = np.zeros_like(Speed)
    Min_speed = np.zeros_like(Speed)
    Speed_features = np.zeros_like(Speed)
    for i in range(Speed.shape[0]):
        if Speed[i, :] == -1:
            Speed_features[i, :] = 30 / 160
        else:
            Speed_features[i, :] = Speed[i, :] / 160
    # for i in range(num_intervals):
    #     start_idx = i * 30
    #     end_idx = (i + 1) * 30
    #     current_mean_speed = np.mean(Speed[: end_idx])
    #     Mean_speed[start_idx: end_idx, :] = current_mean_speed
    #     current_max_speed = np.max(Speed[: end_idx])
    #     Max_speed[start_idx: end_idx, :] = current_max_speed
    #     current_min_speed = np.min(Speed[: end_idx])
    #     Min_speed[start_idx: end_idx, :] = current_min_speed

    """Lane features"""
    lane_no = lane_info[0]
    total_lanes = lane_info[1]
    intersection = lane_info[2]
    if total_lanes > lane_no:
        left_action = 1
    else:
        left_action = 0
    if lane_no > 1:
        right_action = 1
    else:
        right_action = 0
    Lane_features = np.array([left_action, right_action, intersection]).reshape(1, 3)
    Lane_features = np.tile(Lane_features, (150, 1))
    # padding to 150 rows
    euler_angle_rows, Eeuler_angle_cols = Euler.shape
    hist_angle_rows, hist_angle_cols = features_hist_angle_normalize.shape
    hist_move_in_x_rows, hist_move_in_x_cols = features_hist_move_in_x_normalize.shape
    mean_movement_x_rows, mean_movement_x_cols = features_mean_movement_x.shape
    hist_mediapipe_angle_rows, hist_mediapipe_angle_cols = features_hist_mediapipe_angle_normalize.shape
    hist_mediapipe_move_in_x_rows, hist_mediapipe_move_in_x_cols = features_hist_mediapipe_move_in_x_normalize.shape
    features_mediapipe_mean_movement_x_rows, features_mediapipe_mean_movement_x_cols = features_mediapipe_mean_movement_x.shape
    _, lane_cols = Lane_features.shape
    speed_rows, speed_cols = Speed_features.shape

    min_rows = min(euler_angle_rows, hist_angle_rows, hist_move_in_x_rows, mean_movement_x_rows, 
                   hist_mediapipe_angle_rows, hist_mediapipe_move_in_x_rows, features_mediapipe_mean_movement_x_rows, 
                   speed_rows)
    # 150 x cols zeros
    euler_angle_target = Euler[1:, :]
    features_hist_angle_normalize_target = features_hist_angle_normalize
    features_hist_move_in_x_normalize_target = features_hist_move_in_x_normalize
    features_mean_movement_x_target = features_mean_movement_x
    features_hist_mediapipe_angle_normalize_target = features_hist_mediapipe_angle_normalize
    features_hist_mediapipe_move_in_x_normalize_target = features_hist_mediapipe_move_in_x_normalize
    features_mediapipe_mean_movement_x_target = features_mediapipe_mean_movement_x
    features_hist_eyegaze_move_in_x_target = features_hist_eyegaze_move_in_x
    features_hist_eyegaze_move_in_y_target = features_hist_eyegaze_move_in_y
    Lane_features_target = Lane_features
    Speed_features_target = Speed_features[1:, :]
    mediapipe_points_in_x_target = mediapipe_points_in_x[1:, :]
    eyegaze_data_target = eyegaze_data[1:, :]
    
    for i in range(min_rows):
        if np.all(euler_angle_target[i, :] == 0):
            euler_angle_target[i, :] = np.zeros_like(euler_angle_target[i, :])
            features_hist_angle_normalize_target[i, :] = np.zeros_like(features_hist_angle_normalize_target[i, :])
            features_hist_move_in_x_normalize_target[i, :] = np.zeros_like(features_hist_move_in_x_normalize_target[i, :])
            features_mean_movement_x_target[i, :] = np.zeros_like(features_mean_movement_x_target[i, :])
            features_hist_mediapipe_angle_normalize_target[i, :] = np.zeros_like(features_hist_mediapipe_angle_normalize_target[i, :])
            features_hist_mediapipe_move_in_x_normalize_target[i, :] = np.zeros_like(features_hist_mediapipe_move_in_x_normalize_target[i, :])
            features_mediapipe_mean_movement_x_target[i, :] = np.zeros_like(features_mediapipe_mean_movement_x_target[i, :])
            features_hist_eyegaze_move_in_x_target[i, :] = np.zeros_like(features_hist_eyegaze_move_in_x_target[i, :])
            features_hist_eyegaze_move_in_y_target[i, :] = np.zeros_like(features_hist_eyegaze_move_in_y_target[i, :])
            
            Lane_features_target[i, :] = np.zeros_like(Lane_features_target[i, :])
            Speed_features_target[i, :] = np.zeros_like(Speed_features_target[i, :])
            mediapipe_points_in_x_target[i, :] = np.zeros_like(mediapipe_points_in_x_target[i, :])
            # eye
            eyegaze_data_target[i, :] = np.zeros_like(eyegaze_data_target[i, :])
    # Euler: 150 x 3
    Euler_features = euler_angle_target
    # Euler_features = Euler[:150, :]
    # Face Features: 150 x 9
    Brain4cars_Face = np.concatenate((features_hist_angle_normalize_target, features_hist_move_in_x_normalize_target, features_mean_movement_x_target), axis=1)
    # Brain4cars_Mediapipe_Face = np.concatenate((features_hist_mediapipe_angle_normalize_target, features_hist_mediapipe_move_in_x_normalize_target, features_mediapipe_mean_movement_x_target), axis=1)
    Brain4cars_Mediapipe_Face = np.concatenate((features_hist_mediapipe_angle_normalize_target, features_hist_mediapipe_move_in_x_normalize_target, eyegaze_data_target), axis=1)
    # plot_intermediate_result(features_mean_movement_x_target, action, "./result_figure/intermediate_results/mean_movement_x")
    # Speed: 150 x3
    # Brain4cars_Speed = np.concatenate((Mean_speed, Max_speed, Min_speed), axis=1)
    Brain4cars_Speed = Speed_features_target
    # Lane: 150 x3
    Brain4cars_Lane = Lane_features_target
    # plot(euler_angle_target.transpose((1, 0)))
    # plot(Brain4cars_Face.transpose((1, 0)))
    # plot(Brain4cars_Speed.transpose((1, 0)))
    # plot(Brain4cars_Lane.transpose((1, 0)))
    # plot(features_mean_movement_x_target.transpose((1, 0)))
    """
    Data augmentation!数据增强部分
        利用tsaug进行数据增强；
        因为部分特征不能被增强，所以这里进行数据集准备阶段的数据增强.
        数据增强需要的数据形状是(channel, time-step)!
    """
    # 获取Euler_features行号为0的坐标，数据增强后这些行元素仍都置0
    zero_rows_index= np.where(np.all(Euler_features == 0, axis=1))[0]

    # AddNoise（可能造成hist<0需要处理）
    Euler_features_AddNoise = tsaug.AddNoise(scale=0.06).augment(euler_angle_target.transpose((1, 0))).transpose((1,0))
    Brain4cars_Face_AddNoise = tsaug.AddNoise(scale=0.06).augment(Brain4cars_Face.transpose((1, 0))).transpose((1,0))
    Brain4cars_Speed_AddNoise = tsaug.AddNoise(scale=0.06).augment(Brain4cars_Speed.transpose((1, 0))).transpose((1,0))

    # Convolve (可能造成hist<0需要处理)
    Euler_features_Convolve = tsaug.Convolve(window="flattop", size=11).augment(euler_angle_target.transpose((1, 0))).transpose((1,0))
    Brain4cars_Face_Convolve = tsaug.Convolve(window="flattop", size=11).augment(Brain4cars_Face.transpose((1, 0))).transpose((1,0))
    Brain4cars_Speed_Convolve = tsaug.Convolve(window="flattop", size=11).augment(Brain4cars_Speed.transpose((1, 0))).transpose((1,0))

    # Dropout
    Euler_features_Dropout = tsaug.Dropout(p=0.1, size=(1,5), fill=float(0), per_channel=True).augment(euler_angle_target.transpose((1, 0))).transpose((1,0))
    Brain4cars_Face_Dropout = tsaug.Dropout(p=0.1, size=(1,5), fill=float(0.1), per_channel=True).augment(Brain4cars_Face.transpose((1, 0))).transpose((1,0))
    Brain4cars_Speed_Dropout = tsaug.Dropout(p=0.1, size=(1,5), fill=float(0.1), per_channel=True).augment(Brain4cars_Speed.transpose((1, 0))).transpose((1,0))

    # Pool (可能造成hist<0需要处理)
    Euler_features_Pool = tsaug.Pool(size=2).augment(euler_angle_target.transpose((1, 0))).transpose((1,0))
    Brain4cars_Face_Pool = tsaug.Pool(size=2).augment(Brain4cars_Face.transpose((1, 0))).transpose((1,0))
    Brain4cars_Speed_Pool = tsaug.Pool(size=2).augment(Brain4cars_Speed.transpose((1, 0))).transpose((1,0))

    # Quantize 
    Euler_features_Quantize = tsaug.Quantize(n_levels=20).augment(euler_angle_target.transpose((1, 0))).transpose((1,0))
    Brain4cars_Face_Quantize = tsaug.Quantize(n_levels=20).augment(Brain4cars_Face.transpose((1, 0))).transpose((1,0))
    Brain4cars_Speed_Quantize = tsaug.Quantize(n_levels=20).augment(Brain4cars_Speed.transpose((1, 0))).transpose((1,0))

    # TimeWarp
    Euler_features_TimeWarp = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(euler_angle_target.transpose((1, 0))).transpose((1,0))
    Brain4cars_Face_TimeWarp = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(Brain4cars_Face.transpose((1, 0))).transpose((1,0))
    Brain4cars_Speed_TimeWarp = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(Brain4cars_Speed.transpose((1, 0))).transpose((1,0))    
    
    # Combnine method
    TimeWarp_Convlve = (tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3) + tsaug.Convolve(window="flattop", size=11))
    Euler_features_Combine = TimeWarp_Convlve.augment(euler_angle_target.transpose((1, 0))).transpose((1,0))
    Brain4cars_Face_Combine = TimeWarp_Convlve.augment(Brain4cars_Face.transpose((1, 0))).transpose((1,0))
    Brain4cars_Speed_Combine = TimeWarp_Convlve.augment(Brain4cars_Speed.transpose((1, 0))).transpose((1,0))       

    # 对增强后的数据进行处理，保证原来是0的地方还为0,同时如hist和spped等特征不能出现负数
    Euler_features_AddNoise, Brain4cars_Face_AddNoise, Brain4cars_Speed_AddNoise = aug_filter(Euler_features_AddNoise, Brain4cars_Face_AddNoise, Brain4cars_Speed_AddNoise, zero_rows_index)
    Euler_features_Convolve, Brain4cars_Face_Convolve, Brain4cars_Speed_Convolve = aug_filter(Euler_features_Convolve, Brain4cars_Face_Convolve, Brain4cars_Speed_Convolve, zero_rows_index)
    Euler_features_Dropout, Brain4cars_Face_Dropout, Brain4cars_Speed_Dropout = aug_filter(Euler_features_Dropout, Brain4cars_Face_Dropout, Brain4cars_Speed_Dropout, zero_rows_index)
    Euler_features_Pool, Brain4cars_Face_Pool, Brain4cars_Speed_Pool = aug_filter(Euler_features_Pool, Brain4cars_Face_Pool, Brain4cars_Speed_Pool, zero_rows_index)
    Euler_features_Quantize, Brain4cars_Face_Quantize, Brain4cars_Speed_Quantize = aug_filter(Euler_features_Quantize, Brain4cars_Face_Quantize, Brain4cars_Speed_Quantize, zero_rows_index)
    Euler_features_TimeWarp, Brain4cars_Face_TimeWarp, Brain4cars_Speed_TimeWarp = aug_filter(Euler_features_TimeWarp, Brain4cars_Face_TimeWarp, Brain4cars_Speed_TimeWarp, zero_rows_index)
    Euler_features_Combine, Brain4cars_Face_Combine, Brain4cars_Speed_Combine = aug_filter(Euler_features_Combine, Brain4cars_Face_Combine, Brain4cars_Speed_Combine, zero_rows_index)

    # 获取以20帧为界限累积的特征
    Euler_features_20frame, Brain4cars_Face_20frame, Brain4cars_Speed_20frame = aggregate_features(20, Euler_features, Brain4cars_Face, Brain4cars_Speed)
    # 数据增强
    Euler_features_20frame_AddNoise, Brain4cars_Face_20frame_AddNoise, Brain4cars_Speed_20frame_AddNoise = aggregate_features(20, Euler_features_AddNoise, Brain4cars_Face_AddNoise, Brain4cars_Speed_AddNoise)
    Euler_features_20frame_Convolve, Brain4cars_Face_20frame_Convolve, Brain4cars_Speed_20frame_Convolve = aggregate_features(20, Euler_features_Convolve, Brain4cars_Face_Convolve, Brain4cars_Speed_Convolve)
    Euler_features_20frame_Dropout, Brain4cars_Face_20frame_Dropout, Brain4cars_Speed_20frame_Dropout = aggregate_features(20, Euler_features_Dropout, Brain4cars_Face_Dropout, Brain4cars_Speed_Dropout)
    Euler_features_20frame_Pool, Brain4cars_Face_20frame_Pool, Brain4cars_Speed_20frame_Pool = aggregate_features(20, Euler_features_Pool, Brain4cars_Face_Pool, Brain4cars_Speed_Pool)
    Euler_features_20frame_Quantize, Brain4cars_Face_20frame_Quantize, Brain4cars_Speed_20frame_Quantize = aggregate_features(20, Euler_features_Quantize, Brain4cars_Face_Quantize, Brain4cars_Speed_Quantize)
    Euler_features_20frame_TimeWarp, Brain4cars_Face_20frame_TimeWarp, Brain4cars_Speed_20frame_TimeWarp = aggregate_features(20, Euler_features_TimeWarp, Brain4cars_Face_TimeWarp, Brain4cars_Speed_TimeWarp)
    Euler_features_20frame_Combine, Brain4cars_Face_20frame_Combine, Brain4cars_Speed_20frame_Combine = aggregate_features(20, Euler_features_Combine, Brain4cars_Face_Combine, Brain4cars_Speed_Combine)

    # 单独处理lane信息
    # Lane: 7 x 3
    Brain4cars_Lane_20frame = np.zeros((7, Brain4cars_Lane.shape[1]))
    for i in range(7):
        if np.all(Euler_features_20frame[i, :] == 0):
            Brain4cars_Lane_20frame[i, :] = np.zeros((1, 3))
        else:
            Brain4cars_Lane_20frame[i, :] = Lane_features[0, :]

    # 展示mat file中的landmark信息
    # video = cv2.VideoCapture(action)
    # count = 0
    # while True:
    #     ret, frame = video.read()
    #     if not ret:
    #         break
    #     image_height, image_width, _ = np.shape(frame)
    #     row, col = landmarks.shape
    #     for i in range(0, col, 2):
    #         mark_x = int(landmarks[count, i])
    #         mark_y = int(landmarks[count, i+1])
    #         cv2.circle(frame, (mark_x, mark_y), 4, (0, 255, 0), -5)
    #     count += 1
    #     frame_new = cv2.resize(frame, (int(image_width / 2), int(image_height /2)))
    #     cv2.imshow('face_mesh', frame_new)
    #     key = cv2.waitKey(30) & 0xFF
    #     if key == ord('q'):
    #         break
    # video.release()    
    
    return Brain4cars_Lane, Brain4cars_Lane_20frame, \
           Euler_features, Brain4cars_Face, Brain4cars_Mediapipe_Face, Brain4cars_Speed, \
           Euler_features_20frame, Brain4cars_Face_20frame, Brain4cars_Speed_20frame, \
           Euler_features_20frame_AddNoise, Brain4cars_Face_20frame_AddNoise, Brain4cars_Speed_20frame_AddNoise, \
           Euler_features_20frame_Convolve, Brain4cars_Face_20frame_Convolve, Brain4cars_Speed_20frame_Convolve, \
           Euler_features_20frame_Dropout, Brain4cars_Face_20frame_Dropout, Brain4cars_Speed_20frame_Dropout, \
           Euler_features_20frame_Pool, Brain4cars_Face_20frame_Pool, Brain4cars_Speed_20frame_Pool, \
           Euler_features_20frame_Quantize, Brain4cars_Face_20frame_Quantize, Brain4cars_Speed_20frame_Quantize, \
           Euler_features_20frame_TimeWarp, Brain4cars_Face_20frame_TimeWarp, Brain4cars_Speed_20frame_TimeWarp, \
           Euler_features_20frame_Combine, Brain4cars_Face_20frame_Combine, Brain4cars_Speed_20frame_Combine, mean_hist_move_in_x_normalize, mean_hist_mediapipe_move_in_x_normalize


"""
第一步：由数据集产生原始的json文件，这个json文件包含所有的原始数据路径。
由划分的excel文件产生最原始的数据集路径的json文件，传入的是excel路径，数据集路径以及保存json的路径
excel中保存的是每个驾驶员的不同行为的数据文件夹名称的范围，依靠这个范围对数据集进行划分，但是目前有bug    
"""
def splitbyperson(excel_path, source_path, save_path):
    table = pd.read_excel(excel_path)
    table_head = table.keys().to_list()
    table_value = table.values.tolist() 
    d = []
    person_dict = {'person1': {},'person2': {},'person3': {},'person4': {},'person5': {},'person6': {},'person7': {},'person8': {}}
    # os.walk遍历所有文件夹
    for root, dir, files in sorted(os.walk(source_path)):
        if dir:
            dir = sorted(dir)
            # 遍历每个activity的文件夹
            for idx, head in enumerate(table_head):
                if head in root:
                    print('\n')
                    # 每遍历一个activity，把这个activity按person归类
                    for person in range(len(table_value)):
                        person_file_list = []
                        person_file_path_list = []
                        # dir_name_list 每个person所占的片段，是excel中的片段
                        dir_name_list = re.split('[-|,]', str(table_value[person][idx]))
                        print(table_value[person][0], re.split('[-|,]', str(table_value[person][idx])))
                        # 片段应该是偶数：开始-结束, 对每个
                        if len(dir_name_list) != 1:
                            if len(dir_name_list) == 2:
                                start, end = dir.index(dir_name_list[0]), dir.index(dir_name_list[1])
                                person_file_list += dir[start: end + 1]
                            elif len(dir_name_list) == 4:
                                start1, end1 = dir.index(dir_name_list[0]), dir.index(dir_name_list[1])
                                start2, end2 = dir.index(dir_name_list[2]), dir.index(dir_name_list[3])
                                person_file_list += dir[start1: end1 + 1]
                                person_file_list += dir[start2: end2 + 1]
                            for i in person_file_list:
                                person_file_path = os.path.join(root, i)
                                person_file_path_list.append(person_file_path)

                        # person_action = {table_head[idx]: person_file_path_list}
                        if table_head[idx] not in  person_dict[table_value[person][0]]:
                            person_dict[table_value[person][0]][table_head[idx]] = person_file_path_list
                        else:
                            person_dict[table_value[person][0]][table_head[idx]] += person_file_path_list
                
    person_json_data = json.dumps(person_dict, indent=1)
    with open(save_path + 'person_origin.json', 'w') as f_origin:
        f_origin.write(person_json_data)
    f_origin.close()

"""
第二步：由原始的数据集json文件产生过滤之后的json文件 
传入 由splitbyperson产生的json地址，以及保存的路径
"""
def filter_dataset(original_dataset_json_path, save_path):
    with open(original_dataset_json_path) as f:
        face_dataset_original = json.load(f)
    f.close()
    dataset_dict = {f"person{i+1}": {"road_camera": {}, "face_camera": {}} for i in range(8)}
    face_filterd_files = []
    road_filterd_files = []
    person_list = ['person1', 'person2', 'person3', 'person4', 'person5', 'person6', 'person7', 'person8']
    activity_list = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']

    thred = 50
    face_total_files_count = 0
    road_total_files_count = 0
    face_filterd_files_count = 0
    road_filterd_files_count = 0
    remained_video = 0
    road_camera = "road_camera"
    face_camera = "face_camera"
    for pidx, person in enumerate(person_list):
        for aidx, activity in enumerate(activity_list):
            road_activity_list = []
            face_activity_list = []
            per_act_path_list = face_dataset_original[person][activity]
            for i, pap in enumerate(per_act_path_list):
                face_total_files_count += 1
                road_total_files_count += 1
                # from face_video dirpath to face_video filepath
                face_video_name_list = pap.split(os.sep)
                video_name = face_video_name_list[-1]
                face_video_path = os.path.join(pap, "video_"+video_name+'.avi')
                # from video_path to road_video_path
                road_video_name_list = [road_camera if i == face_camera else i for i in face_video_name_list]
                road_video_name_list[-1] += '.avi'
                road_video_path = os.path.normpath(os.sep.join(road_video_name_list))
                
                # filter face_video
                face_video_time, face_save_flag =  filter_by_video(thred / 30, face_video_path)
                # filter video_path
                # road_video_time, road_save_flag = filter_by_video(thred, road_video_path)
                face_video_frame, face_save_flag = filter_by_mat(thred, face_video_path)
                
                # 结论：视频的长度基本是长于mat file中frame_end - frame_start的长度的
                if face_video_time < (face_video_frame) / 30:
                    print("{}帧数长度为{} 视频长度为 {}".format(face_video_path, (face_video_frame) / 30, face_video_time))


                if not face_save_flag:
                    face_filterd_files.append(face_video_path)
                    face_filterd_files_count += 1
                    print("{}mat file长度为 {:.2f}frame, 少于{}frame，被过滤掉！".format(face_video_path, face_video_frame, thred))
                
                if not face_save_flag:
                    road_filterd_files.append(road_video_path)
                    road_filterd_files_count += 1
                    print("{}mat file长度为 {:.2f}frame, 少于{}frame，被过滤掉！".format(road_video_path, face_video_frame, thred))                
                   
                if face_save_flag:
                    remained_video += 1
                    face_activity_list.append(face_video_path)
                    road_activity_list.append(road_video_path)
            
            if activity not in dataset_dict[person][face_camera]:
                dataset_dict[person][face_camera][activity] = face_activity_list
                dataset_dict[person][road_camera][activity] = road_activity_list
            else:
                dataset_dict[person][face_camera][activity] += face_activity_list
                dataset_dict[person][road_camera][activity] += road_activity_list
    print("视频过滤的标准：视频长度少于{} frame的被过滤掉！".format(thred))
    print("face视频总数是{}, 被过滤掉视频总数是{}, 剩余视频数量{}".format(face_total_files_count, face_filterd_files_count, remained_video))
    print("road视频总数是{}, 被过滤掉视频总数是{}, 剩余视频数量{}".format(road_total_files_count, road_filterd_files_count, remained_video))
    
    dataset_dict_json = json.dumps(dataset_dict, indent=1)
    with open(save_path + f'Brain4cars_datasetbyperson_alldata_thred_{thred}.json', 'w') as f:
        f.write(dataset_dict_json)
    f.close()


"""
第三步：根据过滤后的json文件产生原始数据csv文件
"""
def generated_feature_csv(filterd_dataset_json_path):
    with open(filterd_dataset_json_path) as f:
        dataset_dict = json.load(f)
    f.close()
    args = parse_args()
    gaze_pipeline = Pipeline(
                weights='/home/dulab/ML/Code_project/L2CS-Net/models/L2CSNet_gaze360.pkl',
                arch='ResNet50',
                device = select_device(args.device, batch_size=1)
                )
    person_list = list(dataset_dict.keys())
    action_list = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']

    # total_video_nums = 587
    processed_video = 0
    action_50_100_dict = {}
    end_action_move_in_x = []
    lchange_move_in_x = []
    lturn_move_in_x = []
    rchange_move_in_x = []
    rturn_move_in_x = []
    mediapipe_end_action_move_in_x = []
    mediapipe_lchange_move_in_x = []
    mediapipe_lturn_move_in_x = []
    mediapipe_rchange_move_in_x = []
    mediapipe_rturn_move_in_x = []

    eyegaze_end_action_move_in_x = []
    eyegaze_lchange_move_in_x = []
    eyegaze_lturn_move_in_x = []
    eyegaze_rchange_move_in_x = []
    eyegaze_rturn_move_in_x = []
    eyegaze_end_action_move_in_y = []
    eyegaze_lchange_move_in_y = []
    eyegaze_lturn_move_in_y = []
    eyegaze_rchange_move_in_y = []
    eyegaze_rturn_move_in_y = []
    for p_idx, person in enumerate(person_list):
        person_video_count = 0
        print("-"*15,"Start processing video data for {}.".format(person) + "-"*15)
        person_dataset = dataset_dict[person]
        # 获取每位驾驶员的face_camera集合
        person_face_dataset = person_dataset['face_camera']
        for a_idx, action in enumerate(action_list):
            #获取每个行为的列表
            actions = person_face_dataset[action]
            #遍历每个列表的视频地址
            for video_idx, action in enumerate(actions):
                person_video_count += 1
                processed_video += 1
                # if processed_video < 230:
                #     continue
                # 从原始视频获取landmark坐标点函数
                # posex_list, posey_list = get_landmarks(action, prefix="mediapipe_landmarks_")                
                # 显示landmark以及eye gaze
                """ eye gaze"""
                start_gaze = time.time()
                pitch_list, yaw_list = getEyeGzae(action, gaze_pipeline, prefix="eyegaze_angles_")
                end_gaze = time.time()
                if pitch_list is not None and yaw_list is not None:
                    print(f"处理时间为 {end_gaze - start_gaze} s")
                    landmarks_writer(action, pitch_list, yaw_list, prefix="eyegaze_angles_")
                    origin_data, filtered_data = filter_eyedata(action)
                    non_zero_mask = ~np.all(filtered_data == 0, axis=1)
                    non_zero_rows = filtered_data[non_zero_mask]
                    column_mean = np.mean(non_zero_rows, axis=0)
                    plot_eyegaze_intermediate_result(origin_data, filtered_data, action, "./result_figure/intermediate_results/eye_gaze_butterworth")
                    action_50_100_dict[video_idx] = action
                    
                # try:
                #     video1 = cv2.VideoCapture(action)
                # except FileNotFoundError:
                #     print("无法打开{}路径下文件,请检查路径！".format(action))
                # video_width = video1.get(cv2.CAP_PROP_FRAME_WIDTH)
                # video_height = video1.get(cv2.CAP_PROP_FRAME_HEIGHT)
                # frame_count = 0
                # while True:
                #     ret, frame = video1.read()
                #     if not ret:
                #         break
                #     # rows, cols = posex_list.shape
                #     # for j in range(cols):
                #     #     mark_x = int(posex_list[frame_count][j])
                #     #     mark_y = int(posey_list[frame_count][j])
                #     #     cv2.circle(frame, (mark_x, mark_y), 4, (0, 255, 0), -5)
                #     results = gaze_pipeline.step(frame)
                #     frame = render(frame, results)
                #     frame_count += 1
                #     frame_new = cv2.resize(frame, (int(video_width / 2), int(video_height /2)))
                #     cv2.imshow('face_mesh', frame_new)
                #     key = cv2.waitKey(30) & 0xFF
                #     if key == ord('q'):
                #         break
                # video1.release()
                                
                # 将坐标点写入csv函数
                # if posex_list is not None and posey_list is not None:
                #     landmarks_writer(action, posex_list, posey_list, prefix="mediapipe_landmarks_")              
                
                # 提取瞳孔特征，shape: 150x9
                # mediapipe_features =  EyeFeaturesExtract(action)
                
                # 提取mat file特征 150x9, 150x3, 150x3
                # brain4cars_lane_features, Brain4cars_Lane_20frame, \
                # Euler_features, brain4cars_face_features, brain4cars_mediapipe_face_features, brain4cars_speed_features, \
                # Euler_features_20frame, Brain4cars_Face_20frame, Brain4cars_Speed_20frame, \
                # Euler_features_20frame_AddNoise, Brain4cars_Face_20frame_AddNoise, Brain4cars_Speed_20frame_AddNoise, \
                # Euler_features_20frame_Convolve, Brain4cars_Face_20frame_Convolve, Brain4cars_Speed_20frame_Convolve, \
                # Euler_features_20frame_Dropout, Brain4cars_Face_20frame_Dropout, Brain4cars_Speed_20frame_Dropout, \
                # Euler_features_20frame_Pool, Brain4cars_Face_20frame_Pool, Brain4cars_Speed_20frame_Pool, \
                # Euler_features_20frame_Quantize, Brain4cars_Face_20frame_Quantize, Brain4cars_Speed_20frame_Quantize, \
                # Euler_features_20frame_TimeWarp, Brain4cars_Face_20frame_TimeWarp, Brain4cars_Speed_20frame_TimeWarp, \
                # Euler_features_20frame_Combine, Brain4cars_Face_20frame_Combine, Brain4cars_Speed_20frame_Combine, mean_hist_move_in_x_normalize, mean_hist_mediapipe_move_in_x_normalize = Brain4carFeatureExtract(action)
                
                # # Euler_features, brain4cars_face_features, brain4cars_speed_features, brain4cars_lane_features, \
                # # Euler_features_20frame, Brain4cars_Face_20frame, Brain4cars_Speed_20frame, Brain4cars_Lane_20frame = Brain4carFeatureExtract(action)
                
                # All_features                  = np.concatenate((Euler_features, brain4cars_face_features, brain4cars_lane_features, brain4cars_speed_features), axis=1)
                # All_features_withmediapipe      = np.concatenate((Euler_features, brain4cars_mediapipe_face_features, brain4cars_lane_features, brain4cars_speed_features), axis=1)
                
                # All_features_20frame          = np.concatenate((Euler_features_20frame, Brain4cars_Face_20frame, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame), axis=1)
                # All_features_20frame_AddNoise = np.concatenate((Euler_features_20frame_AddNoise, Brain4cars_Face_20frame_AddNoise, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_AddNoise), axis=1)
                # All_features_20frame_Convolve = np.concatenate((Euler_features_20frame_Convolve, Brain4cars_Face_20frame_Convolve, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_Convolve), axis=1)
                # All_features_20frame_Dropout  = np.concatenate((Euler_features_20frame_Dropout, Brain4cars_Face_20frame_Dropout, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_Dropout), axis=1)
                # All_features_20frame_Pool     = np.concatenate((Euler_features_20frame_Pool, Brain4cars_Face_20frame_Pool, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_Pool), axis=1)
                # All_features_20frame_Quantize = np.concatenate((Euler_features_20frame_Quantize, Brain4cars_Face_20frame_Quantize, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_Quantize), axis=1)
                # All_features_20frame_TimeWarp = np.concatenate((Euler_features_20frame_TimeWarp, Brain4cars_Face_20frame_TimeWarp, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_TimeWarp), axis=1)
                # All_features_20frame_Combine  = np.concatenate((Euler_features_20frame_Combine, Brain4cars_Face_20frame_Combine, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_Combine), axis=1)

                All_features_header =['yaw', 'pitch', 'row'] +  [f'face_angle_hist_{i}' for i in range(4)] + [f'face_move_in_x_hist_{i}' for i in range(6)] \
                                    + ["face_mean_move"] + ['left_action', 'right_action', 'intersection'] + ['mean_speed'] 
                
                # All_features_withmediapipe_header = ['yaw', 'pitch', 'row'] +  [f'face_angle_hist_{i}' for i in range(4)] + [f'face_move_in_x_hist_{i}' for i in range(6)] + ["face_mean_move_x"] \
                #                                   + ['left_action', 'right_action', 'intersection'] + ['mean_speed']
                All_features_withmediapipe_header = ['yaw', 'pitch', 'row'] +  [f'face_angle_hist_{i}' for i in range(4)] + [f'face_move_in_x_hist_{i}' for i in range(6)] \
                                                  + [f'eyegaze_move_in_x_'] + [f'eyegaze_move_in_y'] \
                                                  + ['left_action', 'right_action', 'intersection'] + ['mean_speed']
                # All_features_header_20frame = ['yaw', 'pitch', 'row'] + [f'face_angle_hist_{i}' for i in range(4)] + [f'face_move_in_x_hist_{i}' for i in range(4)] \
                #                     + ["face_mean_move_x"] + ['left_action', 'right_action', 'intersection'] + ['mean_speed'] 

                All_features_csv_path =  action2csvpath(action, prefix='allfeatures_alone')
                All_features_withmediapipe_csv_path = action2csvpath(action, prefix='allfeatures_withmediapipeandeye')
                # Frame_csv_path = action2csvpath(action, prefix='20frame_features')
                # Frame_csv_path_AddNoise = action2csvpath(action, prefix='20frame_features_AddNoise')
                # Frame_csv_path_Convolve = action2csvpath(action, prefix='20frame_features_Convolve')
                # Frame_csv_path_Dropout = action2csvpath(action, prefix='20frame_features_Dropout')
                # Frame_csv_path_Pool = action2csvpath(action, prefix='20frame_features_Pool')
                # Frame_csv_path_Quantize = action2csvpath(action, prefix='20frame_features_Quantize')
                # Frame_csv_path_TimeWarp = action2csvpath(action, prefix='20frame_features_TimeWarp')
                # Frame_csv_path_Combine = action2csvpath(action, prefix='20frame_features_Combine')
                
                # ndarray2csvfile(All_features_csv_path, All_features, All_features_header)
                # ndarray2csvfile(All_features_withmediapipe_csv_path, All_features_withmediapipe, All_features_withmediapipe_header)
                # ndarray2csvfile(Frame_csv_path, All_features_20frame, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_AddNoise, All_features_20frame_AddNoise, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_Convolve, All_features_20frame_Convolve, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_Dropout, All_features_20frame_Dropout, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_Pool, All_features_20frame_Pool, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_Quantize, All_features_20frame_Quantize, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_TimeWarp, All_features_20frame_TimeWarp, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_Combine, All_features_20frame_Combine, All_features_header_20frame)

    #             if "end_action" in action:
    #                 end_action_move_in_x.append(mean_hist_move_in_x_normalize)
    #                 mediapipe_end_action_move_in_x.append(mean_hist_mediapipe_move_in_x_normalize)
    #                 eyegaze_end_action_move_in_x.append(column_mean[0])
    #                 eyegaze_end_action_move_in_y.append(column_mean[1])
    #             elif "lchange" in action:
    #                 lchange_move_in_x.append(mean_hist_move_in_x_normalize)
    #                 mediapipe_lchange_move_in_x.append(mean_hist_mediapipe_move_in_x_normalize)
    #                 eyegaze_lchange_move_in_x.append(column_mean[0])
    #                 eyegaze_lchange_move_in_y.append(column_mean[1])

    #             elif "lturn" in action:
    #                 lturn_move_in_x.append(mean_hist_move_in_x_normalize)
    #                 mediapipe_lturn_move_in_x.append(mean_hist_mediapipe_move_in_x_normalize)
    #                 eyegaze_lturn_move_in_x.append(column_mean[0])
    #                 eyegaze_lturn_move_in_y.append(column_mean[1])

    #             elif "rchange" in action:
    #                 rchange_move_in_x.append(mean_hist_move_in_x_normalize)
    #                 mediapipe_rchange_move_in_x.append(mean_hist_mediapipe_move_in_x_normalize)
    #                 eyegaze_rchange_move_in_x.append(column_mean[0])
    #                 eyegaze_rchange_move_in_y.append(column_mean[1])

    #             elif "rturn" in action:
    #                 rturn_move_in_x.append(mean_hist_move_in_x_normalize)
    #                 mediapipe_rturn_move_in_x.append(mean_hist_mediapipe_move_in_x_normalize)
    #                 eyegaze_rturn_move_in_x.append(column_mean[0])
    #                 eyegaze_rturn_move_in_y.append(column_mean[1])                    
    #             print("The {} is processed.\n".format(action))
    #             print("The number of videos processed is {}.\n".format(processed_video))
    #     print("person{}的视频数量为：{}".format(p_idx, person_video_count))
    # eyegaze_end_action_move_in_x = np.array(eyegaze_end_action_move_in_x)
    # eyegaze_lchange_move_in_x = np.array(eyegaze_lchange_move_in_x)
    # eyegaze_lturn_move_in_x = np.array(eyegaze_lturn_move_in_x)
    # eyegaze_rchange_move_in_x = np.array(eyegaze_rchange_move_in_x)
    # eyegaze_rturn_move_in_x = np.array(eyegaze_rturn_move_in_x)
    # print(f"end_action: EyeGazeX mean:{np.mean(eyegaze_rturn_move_in_y)}, EyeGazeY mean: {np.mean(eyegaze_end_action_move_in_y)}")
    # print(f"lchange: EyeGazeX mean: {np.mean(eyegaze_lchange_move_in_x)}, EyeGazeY mean: {np.mean(eyegaze_lchange_move_in_y)}")
    # print(f"lturn: EyeGazeX mean: {np.mean(eyegaze_lturn_move_in_x)}, EyeGazeY mean: {np.mean(eyegaze_lturn_move_in_y)}")
    # print(f"rchange: EyeGazeX mean: {np.mean(eyegaze_rchange_move_in_x)}, EyeGazeY mean: {np.mean(eyegaze_rchange_move_in_y)}")
    # print(f"rturn: EyeGazeX mean: {np.mean(eyegaze_rturn_move_in_x)}, EyeGazeY mean: {np.mean(eyegaze_rturn_move_in_y)}")
    action_50_100_json = json.dumps(action_50_100_dict, indent=1)
    with open("./brain4cars_pipeline/temp_data/action_50_100_list.json", 'w') as f:
        f.write(action_50_100_json)
    f.close()
    
"""
第四步划分数据集，按照驾驶员个体划分或者是随机划分
"""
def TrainTestSet_byperson(dataset_dict, person_list, action_list):
    train_dataset = {'annotations': []}
    test_dataset = {'annotations': []}
    action_list = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']
    test_person = ['person2', 'person4', 'person7']
    processed_video = 0
    for p_idx, person in enumerate(person_list):
        person_video_count = 0
        print("-"*15,"Start processing video data for {}.".format(person) + "-"*15)
        person_dataset = dataset_dict[person]
        # 获取每位驾驶员的face_camera集合
        person_face_dataset = person_dataset['face_camera']
        for a_idx, action in enumerate(action_list):
            #获取每个行为的列表
            actions = person_face_dataset[action]
            #遍历每个列表的视频地址
            for video_file in actions:
                person_video_count += 1
                csv_list = video_file.split(os.sep)
                csv_mark = csv_list[-1].replace('video_', 'allfeatures_').replace('.avi', '.csv')
                csv_list[-1] = csv_mark
                All_features_csv_path = os.sep.join(csv_list)
                if person in test_person:
                    test_dataset['annotations'].append({'filename': All_features_csv_path, 'label': action_list.index(action)})
                else:
                    train_dataset['annotations'].append({'filename': All_features_csv_path, 'label': action_list.index(action)})        
        print("person{}的视频数量为：{}".format(p_idx, person_video_count))

    trainset_json = json.dumps(train_dataset, indent=1)
    testset_json = json.dumps(test_dataset, indent=1)
    with open('./brain4cars_train_dataset.json', 'w') as f_train:
        f_train.write(trainset_json)
    f_train.close()
    with open('./brain4cars_test_dataset.json', 'w') as f_test:
        f_test.write(testset_json)
    f_test.close()


def TrainTestSetRandom(filterd_dataset_json_path, dataset_save_path, label_type = 'one'):
    with open(filterd_dataset_json_path) as f:
        dataset_dict = json.load(f)
    f.close()

    person_list = list(dataset_dict.keys())
    action_list = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']

    random_seed = 30
    random.seed(random_seed)
    test_ratio    = 0
    valid_ratio   = 0.15
    train_dataset = {'annotations': []}
    valid_dataset = {'annotations': []}
    test_dataset  = {'annotations': []}
    dataset_list  = []
    labels_list   = []
    train_list    = []
    train_labels  = []
    test_list     = []
    test_labels   = []
    valid_list    = []
    valid_labels  = []
    for p_idx, person in enumerate(person_list):
        person_video_count = 0
        print("-"*15,"Start processing video data for {}.".format(person) + "-"*15)
        person_dataset = dataset_dict[person]
        # 获取每位驾驶员的face_camera集合
        person_face_dataset = person_dataset['face_camera']
        for a_idx, action in enumerate(action_list):
            #获取每个行为的列表
            actions = person_face_dataset[action]
            #遍历每个列表的视频地址
            for video_path in actions:
                person_video_count += 1
                allfeatrues_csv_path = action2csvpath(video_path, prefix='allfeatures_alone')
                if label_type == 'one':
                    dataset_list.append(allfeatrues_csv_path)
                    labels_list.append(action_list.index(action))

                elif label_type == 'multi':
                    Frame_csv_path = action2csvpath(video_path, prefix='allfeatures_alone')
                    # Frame_csv_path_AddNoise = action2csvpath(video_path, prefix='20frame_features_AddNoise')
                    # Frame_csv_path_Convolve = action2csvpath(video_path, prefix='20frame_features_Convolve')
                    # Frame_csv_path_Dropout = action2csvpath(video_path, prefix='20frame_features_Dropout')
                    # Frame_csv_path_Pool = action2csvpath(video_path, prefix='20frame_features_Pool')
                    # Frame_csv_path_Quantize = action2csvpath(video_path, prefix='20frame_features_Quantize')
                    # Frame_csv_path_TimeWarp = action2csvpath(video_path, prefix='20frame_features_TimeWarp')
                    # Frame_csv_path_Combine = action2csvpath(video_path, prefix='20frame_features_Combine')
                
                    # dataset_list.append([Frame_csv_path, 
                    #                      Frame_csv_path_AddNoise, 
                    #                      Frame_csv_path_Convolve, 
                    #                      Frame_csv_path_Dropout, 
                    #                      Frame_csv_path_Pool,
                    #                      Frame_csv_path_Quantize, 
                    #                      Frame_csv_path_TimeWarp, 
                    #                      Frame_csv_path_Combine])
                    dataset_list.append(Frame_csv_path
                                        )
                    labels = []
                    try:
                        table = pd.read_csv(Frame_csv_path)
                        # table_value: channelx150
                        table_value = table.values
                    except Exception as e:
                        raise Exception(f'Error loading csv from {Frame_csv_path}') from e
                    table_value = np.array(table_value)
                    row_count_zero = 0
                    for idx, row in enumerate(table_value):
                        if np.all(row == 0):
                            row_count_zero += 1
                        if (idx + 1) % 25 == 0:
                            if row_count_zero > 10:
                                 labels.append(0)
                            else:
                                labels.append(action_list.index(action)+1)
                            row_count_zero = 0
                        # if np.all(row == 0):
                        #     labels.append(0)
                        #     row_count_zero += 1
                        # else:
                        #     labels.append(action_list.index(action)+1)
                    labels_list.append(labels)
                    
    dataset_indices = list(range(len(dataset_list)))
    random.shuffle(dataset_indices)
    
    test_split = int(test_ratio * len(dataset_indices))
    valid_split = int((test_ratio + valid_ratio) * len(dataset_indices))
    test_indices = dataset_indices[:test_split]
    valid_indices = dataset_indices[test_split: valid_split]
    train_indices = dataset_indices[valid_split: ]
    
    train_list = [dataset_list[i] for i in train_indices]
    train_labels = [labels_list[i] for i in train_indices]

    test_list = [dataset_list[i] for i in test_indices]
    test_labels = [labels_list[i] for i in test_indices]

    valid_list = [dataset_list[i] for i in valid_indices]
    valid_labels = [labels_list[i] for i in valid_indices]

    for i in range(len(train_list)):
        train_dataset['annotations'].append({'filename': train_list[i], 'label': train_labels[i]})
    for i in range(len(test_list)):
        test_dataset['annotations'].append({'filename': test_list[i], 'label': test_labels[i]})
    for i in range(len(valid_list)):
        valid_dataset['annotations'].append({'filename': valid_list[i], 'label': valid_labels[i]})
    
    trainset_json = json.dumps(train_dataset, indent=1)
    testset_json = json.dumps(test_dataset, indent=1)
    validset_json = json.dumps(valid_dataset, indent=1)
    with open(dataset_save_path + 'brain4cars_train_dataset_random.json', 'w') as f_train:
        f_train.write(trainset_json)
    f_train.close()
    with open(dataset_save_path + 'brain4cars_test_dataset_random.json', 'w') as f_test:
        f_test.write(testset_json)
    f_test.close()
    with open(dataset_save_path + 'brain4cars_valid_dataset_random.json', 'w') as f_test:
        f_test.write(validset_json)
    f_test.close()


def TrainTestSetKFolder(filterd_dataset_json_path, dataset_save_path, k=5, test_ratio=0.15):
    random.seed(30)
    with open(filterd_dataset_json_path) as f:
        dataset_dict = json.load(f)
    f.close()
    person_list = list(dataset_dict.keys())
    action_list = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']

    dataset_list  = []
    labels_list   = []
    train_list    = []
    train_labels  = []
    test_list     = []
    test_labels   = []
    valid_list    = []
    valid_labels  = []
    for p_idx, person in enumerate(person_list):
        person_video_count = 0
        print("-"*15,"Start processing video data for {}.".format(person) + "-"*15)
        person_dataset = dataset_dict[person]
        # 获取每位驾驶员的face_camera集合
        person_face_dataset = person_dataset['face_camera']
        for a_idx, action in enumerate(action_list):
            #获取每个行为的列表
            actions = person_face_dataset[action]
            #遍历每个列表的视频地址
            for video_path in actions:
                person_video_count += 1
                allfeatrues_csv_path = action2csvpath(video_path, prefix='allfeatures_withmediapipe')
                Frame_csv_path = action2csvpath(video_path, prefix='allfeatures_withmediapipeandeye')
                dataset_list.append(Frame_csv_path)
                labels = []
                try:
                    table = pd.read_csv(Frame_csv_path)
                    # table_value: channelx150
                    table_value = table.values
                except Exception as e:
                    raise Exception(f'Error loading csv from {Frame_csv_path}') from e
                table_value = np.array(table_value)
                row_count_zero = 0
                for idx, row in enumerate(table_value):
                    if np.all(row == 0):
                        row_count_zero += 1
                    if (idx + 1) % 25 == 0:
                        if row_count_zero > 10:
                                labels.append(0)
                        else:
                            labels.append(action_list.index(action)+1)
                        row_count_zero = 0
                labels_list.append(labels)
    
    # 通过数据集的下标号来实现随机划分
    dataset_indices = list(range(len(dataset_list)))
    random.shuffle(dataset_indices)
    
    q, r = divmod(len(dataset_indices), k)
    group_size = [q] * k
    for i in range(r):
        group_size[-1] += 1
    groups = []
    i = 0
    for size in group_size:
        group = dataset_indices[i: i+size]
        groups.append(group)
        i += size
    
    # 开始生成kfolder的json文件
    for idx in range(k):
        train_dataset = {'annotations': []}
        valid_dataset = {'annotations': []}
        valid_list    = groups[idx]
        num_valid_to_select = int(len(valid_list) * min(1.0, k * test_ratio))
        valid_list_final = valid_list[: num_valid_to_select]
        train_list_total = [group for j, group in enumerate(groups) if j != idx]
        train_list_flat = [] + valid_list[num_valid_to_select: ]
        for sub_list in train_list_total:
            train_list_flat += sub_list
        for i1 in train_list_flat:
            train_dataset['annotations'].append({'filename': dataset_list[i1], 'label': labels_list[i1]})
        for i2 in valid_list_final:
            valid_dataset['annotations'].append({'filename': dataset_list[i2], 'label': labels_list[i2]})

        trainset_json = json.dumps(train_dataset, indent=1)
        validset_json = json.dumps(valid_dataset, indent=1)
        if not os.path.exists(dataset_save_path + f"/folder{idx+1}"):
            os.mkdir(dataset_save_path + f"/folder{idx+1}")
        with open(dataset_save_path + f"/folder{idx+1}/" + 'brain4cars_train_dataset_mediapipe_random4.json', 'w') as f_train:
            f_train.write(trainset_json)
        f_train.close()
        with open(dataset_save_path + f"/folder{idx+1}/" + 'brain4cars_valid_dataset_mediapipe_random4.json', 'w') as f_valid:
            f_valid.write(validset_json)
        f_valid.close()

# 数据增强相关函数！
def readbyManeuvers(train_json_path, use_sample_ratio=True):
    actions = ["end_action", "lchange", "lturn", "rchange", "rturn"]
    features = {"end_action":[], "lchange":[], "lturn":[], "rchange":[], "rturn":[]}
    labels = {"end_action":[], "lchange":[], "lturn":[], "rchange":[], "rturn":[]}
    class_wise_count = {}
    sample_ratio = {}
    try:
        with open(train_json_path, 'r') as f:
            dataset_dict = json.load(f)
    except FileNotFoundError as e:
        print(e)
    filenames  = []
    labels_tmp = []
    for csv_info in dataset_dict['annotations']:
        filenames.append(csv_info['filename'])
        labels_tmp.append(csv_info['label'])
    for idx, csv_file in enumerate(filenames):
        try:
            table = pd.read_csv(csv_file)
            # table_value: 150xchannel
            table_value = table.values
        except Exception as e:
            raise Exception(f'Error loading csv from {csv_file}') from e
        action = actions[labels_tmp[idx][-1] - 1]
        features[action].append(table_value)
        labels[action].append(labels_tmp[idx])
    
    for action in actions:
        class_wise_count[action] = 1.0 * len(features[action])
    # sample_ratio确定方式：利用end_action数量/其他行为数量，后续根据这个数量对相应类别进行该倍数的数据增强以平衡类别数
    for action in actions:
        if use_sample_ratio:
            sample_ratio[action] = class_wise_count['end_action'] / class_wise_count[action]
        else:
            sample_ratio[action] = 1.0
    
    return features, labels, sample_ratio

def sampleSubSequences(length, num_samples=1, min_len=1, max_len=150):
    max_len = min(length, max_len)
    min_len = min(min_len, max_len)
    sequence = []
    for i in range(num_samples):
        # len代表随机选取片段的长度 50 < len < 150
        len = random.randint(min_len, max_len)
        # start_idx = random.randint(0, length - len)
        start_idx = length - len - 1
        end_idx = start_idx + len
        if not (start_idx, end_idx) in sequence:
            sequence.append((start_idx, end_idx))
    return sequence

def sample2Feature(sample, feature, label):
    assert len(sample) == 2
    feature_new = np.zeros_like(feature)
    feature_new[sample[0]: sample[1], :] = feature[sample[0]: sample[1], :]
    label_new = []
    row_count_zero = 0
    for idx, row in enumerate(feature_new):
        if np.all(row == 0):
            row_count_zero += 1
        if (idx + 1) % 25 == 0:
            if row_count_zero > 10:
                label_new.append(0)
            else:
                label_new.append(label[-1])
            row_count_zero = 0
    return feature_new, label_new
    
    
def multiplyData(features, labels, sample_train_ratio, extra_sample):
    # 利用features_train以及sample_train_ratio进行数据增强
    actions = ["end_action", "lchange", "lturn", "rchange", "rturn"]
    N = 0
    min_length_sequence = 50
    for action in actions:
        new_samples = []
        new_sample_labels = []
        original_action_len = len(features[action])
        for f, l in zip(features[action], labels[action]):
            N += 1
            samples = sampleSubSequences(f.shape[0], int(sample_train_ratio[action] * extra_sample), min_length_sequence)
            for s in samples:
                N += 1
                augmented_feature, augmented_label = sample2Feature(s, f, l)
                new_samples.append(augmented_feature)
                new_sample_labels.append(augmented_label)
        features[action] += new_samples
        labels[action] += new_sample_labels
        print(f"Original number of {action} data is {original_action_len}; After augmentation, the data of {action} is {len(features[action])}")
    return N, features, labels

def aggregateFeatures(features, labels):
    actions = ["end_action", "lchange", "lturn", "rchange", "rturn"]
    X = []
    Y = []
    N = 0
    for action in actions:
        for feature, label in zip(features[action], labels[action]):
            X.append(feature)
            Y.append(label)
            N += 1
    return np.array(X), np.array(Y), N

def dataAugmention(train_json_path, test_json_path, use_data_augmention=True, extra_sample=4, use_sample_ratio=True):
    # （1）一次性转化kfolder中的所有文件 
    # （2）基于类别占比的数据增强 每个类别都扩大四倍，类别数 + 4*类别数
    # （3）数据增强的裁剪范围确定 目前是 50 < len < 150
    # （4）数据增强后生成pickle文件
    
    params = {'use_data_augmentation': use_data_augmention,
              "min_length_sequence": 50,
              "extra_samples": extra_sample}
    actions = ["end_action", "lchange", "lturn", "rchange", "rturn"]
    # 经过readbyManeuvers处理后，features的结构为: features = {"end_action":[], "lchange":[], "lturn":[], "rchange":[], "rturn":[]}
    # train features
    features_train, labels_train, sample_train_ratio = readbyManeuvers(train_json_path)
    if use_data_augmention:
        N_train, features_train, labels_train = multiplyData(features_train, labels_train, sample_train_ratio, extra_sample)
    # test features
    features_test, labels_test, _ = readbyManeuvers(test_json_path)
    
    features_train, labels_train, num_train = aggregateFeatures(features_train, labels_train)
    features_test, labels_test, num_test = aggregateFeatures(features_test, labels_test)
    
    train_data = {"params": params, "labels": labels_train, "features": features_train, "actions": actions}
    test_data = {"labels": labels_test, "features":features_test, "actions": actions}
    train_data_path = train_json_path.split('json')[0] + 'pik'
    test_data_path = test_json_path.split('json')[0] + 'pik'
    with open(train_data_path, 'wb') as f:
        pickle.dump(train_data, f)
    f.close()
    print(f"Saving train data as {train_data_path}")
    with open(test_data_path, 'wb') as f:
        pickle.dump(test_data, f)
    f.close()
    print(f"Saving test data as {test_data_path}")

def kfolerAugmentation(kfolder_path, jsonfile_name, k=5):
    for i in range(k):
        train_json_path = os.path.join(kfolder_path, f"folder{i + 1}", jsonfile_name)
        test_json_path = os.path.join(kfolder_path, f"folder{i + 1}", jsonfile_name.replace('train', 'valid'))
        dataAugmention(train_json_path, test_json_path)

if __name__ == '__main__':
    # excel_path = "/home/ubuntu/zsj/brain4cars_video/time.xlsx"
    # source_path = os.path.normpath("/home/ubuntu/zsj/brain4cars_video/brain4cars_data")
    # save_path = "/home/ubuntu/zsj/GTN-master/brain4cars_process/"
    # splitbyperson(excel_path, source_path, save_path)
    # filter_dataset("/home/ubuntu/zsj/GTN-master/brain4cars_process/person_all_data.json", "/home/ubuntu/zsj/GTN-master/brain4cars_process/")
    # dataAugmention("./brain4cars_pipeline/temp_data/folder1/brain4cars_train_dataset_mediapipe_random.json", "./brain4cars_pipeline/temp_data/folder1/brain4cars_valid_dataset_mediapipe_random.json")
    # kfolerAugmentation("./brain4cars_pipeline/temp_data", "brain4cars_train_dataset_mediapipe_random4.json")
    pass