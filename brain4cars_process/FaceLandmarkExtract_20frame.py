import os
import sys
import cv2
import json
import csv
import tsaug
import random
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tsaug.visualization import plot

json_file = '/home/ubuntu/zsj/GTN-master/brain4cars_process/Brain4cars_datasetbyperson_alldata.json'
with open(json_file) as f:
    dataset_dict = json.load(f)
f.close()

# 获取8位驾驶员的名字
person_list = list(dataset_dict.keys())
action_list = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']
pupil_left_idx = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
pupil_right_idx = []
# 准备mediapipe facemesh api
# 可以参照：https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md
# static_image_mode True:适合静态图片，False：适合视频和动态图片
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                           max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.1,
                           min_tracking_confidence=0.1)


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


def get_landmarks(video_path):
    """
    descriptions: 传入每个视频的地址返回视频的各帧图像的关键点坐标
    """
    x_list, y_list = [], []
    prev_x, prev_y = [0 for i in range(478)], [0 for i in range(478)]
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
        img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_RGB)
        x_per_frame, y_per_frame = [], []
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            for l_idx, landmark in enumerate(face_landmarks.landmark): 
                lx, ly = landmark.x * video_width, landmark.y * video_height
                x_per_frame.append(lx)
                y_per_frame.append(ly)
                cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 0), -5)
            prev_x, prev_y = x_per_frame, y_per_frame
            x_list.append(x_per_frame), y_list.append(y_per_frame)
        else:
            x_list.append(prev_x)
            y_list.append(prev_y)
        frame_count += 1
        # frame_new = cv2.resize(frame, (int(video_width / 2), int(video_height / 2)))
        # cv2.imshow('face_mesh', frame_new)
        # key = cv2.waitKey(30) & 0xFF
        # if key == ord('q'):
        #     break
    video.release()

    return x_list, y_list

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


def landmarks_writer(action, posex_list,posey_list):
    """
    pos_x_array shape: [video_length, 478]
    pos_y_array shape: [video_length, 478]
    """
    pos_x_array = np.array(posex_list)
    pos_y_array = np.array(posey_list)
    print("The shapes of landmarks x and landmarks y are {}, {}".format(pos_x_array.shape, pos_y_array.shape))
    csv_path = action.split('.')[0] + '.csv'
    csv_path = csv_path.split('video_')[0] + "eye_landmarks_" + csv_path.split('video_')[1]
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
    pos_xy_array = pos_xy_array.reshape((x_videolength, 478*2))
    ndarray2csvfile(csv_path, pos_xy_array, header)
    # np.savetxt(csv_path, pos_xy_array, delimiter=',', header=','.join(header), comments='', fmt='%.8f')
    print(".csv save at {} successfully!".format(csv_path))

def EyeFeaturesExtract(action):
    """
    descriptions: 传入视频地址, 转为csv地址再进行关键点处理
    """
    action = "/home/ubuntu/zsj/brain4cars_video/brain4cars_data/face_camera/lturn/20141019_091035_1542_1689/video_20141019_091035_1542_1689.avi"
    hist_angle_values = [-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi]
    hist_distance_x = [-1e3, -2.0, 0, 2.0, 1e3]
    hist_distance_values = [-1e3, 2, 5, 8, 10, 1e3]
    print(action)
    csv_path = action.split('.')[0] + '.csv'
    csv_path = csv_path.split('video_')[0] + "eye_landmarks_" + csv_path.split('video_')[1]
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
    # action = "/home/ubuntu/zsj/brain4cars_video/brain4cars_data/face_camera/rchange/20141019_141831_1436_1586/video_20141019_141831_1436_1586.avi"
    action = action.replace("face_camera", "new_params")
    action = action.replace("video_", "new_param_")
    mat_file = action.split('.')[0] + '.mat'
    try:
        m = loadmat(mat_file)
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
    total_valid_frame = frame_end - frame_strat + 1
    landmarks = np.zeros((total_valid_frame, 68 * 2))
    Euler = np.zeros((total_valid_frame, 3))
    Speed = np.zeros((total_valid_frame, 1))
    for i in range(total_valid_frame):
        keys = frame_data[0][i].dtype.names
        klt_points_per_frame = frame_data[0][i]['klt_points'].item()
        # reshape klts:[[x1,x2,x3,...]                ---> [x1,y1,x2,y2,.....]  shape:[1, 136]
        #                [y1,y2,y3,...]] shape:[2, 68]
        klt_points_per_frame = klt_points_per_frame.reshape((1, -1), order='F')
        if klt_points_per_frame.size == 0:
            klt_points_per_frame = np.zeros((1, 136))
        speed_per_frame = frame_data[0][i]['speed'].item().item()
        if 'Euler' not in keys:
            Euler_per_frame = np.zeros((1, 3))
        else:
            Euler_per_frame = frame_data[0][i]['Euler'].item().reshape(1,3)
        
        landmarks[i, :] = klt_points_per_frame
        Speed[i, :] = speed_per_frame
        Euler[i, :] = Euler_per_frame

    # extract face features 获取特征部分
    hist_angle_values = [-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi]
    hist_distance_x = [-1e3, -2.0, 0, 2.0, 1e3]
    hist_distance_values = [-1e3, 2, 5, 8, 10, 1e3]
    # landmarks: 151 rows -> 150 rows
    points_move_vec = np.diff(landmarks, axis=0)
    points_move_vec[np.abs(points_move_vec) > 50] = 0
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
    # shape: total_valid_frame x 1
    features_mean_movement = np.mean(points_distance, axis=1).reshape((points_distance.shape[0], 1))
    features_mean_movement_x = np.mean(points_move_in_x, axis=1).reshape((points_distance.shape[0], 1))
    # shape: total_valid_frame x 4
    """feature norms: angle"""
    features_hist_angle_2norms = np.linalg.norm(features_hist_angle, axis=1, keepdims=True)
    features_hist_angle_2norms[features_hist_angle_2norms < 1] = 1
    features_hist_angle_normalize = features_hist_angle / features_hist_angle_2norms
    # shape: total_valid_frame x 4
    """feature norms: move_in_x"""
    features_hist_move_in_x_2norms = np.linalg.norm(features_hist_move_in_x, axis=1, keepdims=True)
    features_hist_move_in_x_2norms[features_hist_move_in_x_2norms < 1] = 1
    features_hist_move_in_x_normalize = features_hist_move_in_x / features_hist_move_in_x_2norms
    """Speed features 这里和论文中的不一样，这里的原则是每隔获取最大速度，而不是5s中共用一个最大速度"""
    num_intervals = int(Speed.shape[0] / 30)
     
    # Speed: 151 rows -> 150 rows 将特征统一为150的时间长度
    Mean_speed = np.zeros_like(Speed)
    Max_speed = np.zeros_like(Speed)
    Min_speed = np.zeros_like(Speed)
    Speed_features = np.zeros_like(Speed)
    for i in range(Speed.shape[0]):
        if Speed[i, :] == -1:
            Speed_features[i, :] = 30 / 160
        else:
            Speed_features[i, :] = Speed[i, :] / 160
    for i in range(num_intervals):
        start_idx = i * 30
        end_idx = (i + 1) * 30
        current_mean_speed = np.mean(Speed[: end_idx])
        Mean_speed[start_idx: end_idx, :] = current_mean_speed
        current_max_speed = np.max(Speed[: end_idx])
        Max_speed[start_idx: end_idx, :] = current_max_speed
        current_min_speed = np.min(Speed[: end_idx])
        Min_speed[start_idx: end_idx, :] = current_min_speed

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
    _, lane_cols = Lane_features.shape
    speed_rows, speed_cols = Speed_features.shape

    min_rows = min(euler_angle_rows, hist_angle_rows, hist_move_in_x_rows, mean_movement_x_rows, speed_rows)
    # 150 x cols zeros
    euler_angle_target = np.zeros((150, Eeuler_angle_cols))
    features_hist_angle_normalize_target = np.zeros((150, hist_angle_cols))
    features_hist_move_in_x_normalize_target = np.zeros((150, hist_move_in_x_cols))
    features_mean_movement_x_target = np.zeros((150, mean_movement_x_cols))
    Lane_features_target = np.zeros((150, lane_cols))
    Speed_features_target = np.zeros((150, speed_cols))
    for i in range(min_rows):
        if np.all(Euler[i, :] == 0):
            euler_angle_target[150 - min_rows + i :, :] = np.zeros_like(Euler[i, :])
            features_hist_angle_normalize_target[150 - min_rows + i :, :] = np.zeros_like(features_hist_angle_normalize[i, :])
            features_hist_move_in_x_normalize_target[150 - min_rows + i :, :] = np.zeros_like(features_hist_move_in_x_normalize[i, :])
            Lane_features_target[150 - min_rows + i :, :] = np.zeros_like(Lane_features[i, :])
            Speed_features_target[150 - min_rows + i :, :] = np.zeros_like(Speed_features[i, :])
        else: 
            euler_angle_target[150 - min_rows + i: , :] = Euler[i, :]
            features_hist_angle_normalize_target[150 - min_rows + i: , :] = features_hist_angle_normalize[i, :]
            features_hist_move_in_x_normalize_target[150 - min_rows + i: , :] = features_hist_move_in_x_normalize[i, :]
            features_mean_movement_x_target[150 - min_rows + i: , :] = features_mean_movement_x[i, :]
            Lane_features_target[150 - min_rows + i: , :] = Lane_features[i, :]
            Speed_features_target[150 - min_rows + i: , :] = Speed_features[i, :]

    # Euler: 150 x 3
    Euler_features = euler_angle_target
    # Euler_features = Euler[:150, :]
    # Face Features: 150 x 9
    Brain4cars_Face = np.concatenate((features_hist_angle_normalize_target, features_hist_move_in_x_normalize_target, features_mean_movement_x_target), axis=1)
    # Speed: 150 x3
    # Brain4cars_Speed = np.concatenate((Mean_speed, Max_speed, Min_speed), axis=1)
    Brain4cars_Speed = Speed_features_target
    # Lane: 150 x3
    Brain4cars_Lane = Lane_features_target

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
    return Brain4cars_Lane, Brain4cars_Lane_20frame, \
           Euler_features, Brain4cars_Face, Brain4cars_Speed, \
           Euler_features_20frame, Brain4cars_Face_20frame, Brain4cars_Speed_20frame, \
           Euler_features_20frame_AddNoise, Brain4cars_Face_20frame_AddNoise, Brain4cars_Speed_20frame_AddNoise, \
           Euler_features_20frame_Convolve, Brain4cars_Face_20frame_Convolve, Brain4cars_Speed_20frame_Convolve, \
           Euler_features_20frame_Dropout, Brain4cars_Face_20frame_Dropout, Brain4cars_Speed_20frame_Dropout, \
           Euler_features_20frame_Pool, Brain4cars_Face_20frame_Pool, Brain4cars_Speed_20frame_Pool, \
           Euler_features_20frame_Quantize, Brain4cars_Face_20frame_Quantize, Brain4cars_Speed_20frame_Quantize, \
           Euler_features_20frame_TimeWarp, Brain4cars_Face_20frame_TimeWarp, Brain4cars_Speed_20frame_TimeWarp, \
           Euler_features_20frame_Combine, Brain4cars_Face_20frame_Combine, Brain4cars_Speed_20frame_Combine
           
                                 

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


def TrainTestSetRandom(dataset_dict, person_list, action_list, label_type = 'one'):
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
                allfeatrues_csv_path = action2csvpath(video_path, prefix='allfeatures_withmediapipe')
                if label_type == 'one':
                    dataset_list.append(allfeatrues_csv_path)
                    labels_list.append(action_list.index(action))

                elif label_type == 'multi':
                    Frame_csv_path = action2csvpath(video_path, prefix='20frame_features')
                    Frame_csv_path_AddNoise = action2csvpath(video_path, prefix='20frame_features_AddNoise')
                    Frame_csv_path_Convolve = action2csvpath(video_path, prefix='20frame_features_Convolve')
                    Frame_csv_path_Dropout = action2csvpath(video_path, prefix='20frame_features_Dropout')
                    Frame_csv_path_Pool = action2csvpath(video_path, prefix='20frame_features_Pool')
                    Frame_csv_path_Quantize = action2csvpath(video_path, prefix='20frame_features_Quantize')
                    Frame_csv_path_TimeWarp = action2csvpath(video_path, prefix='20frame_features_TimeWarp')
                    Frame_csv_path_Combine = action2csvpath(video_path, prefix='20frame_features_Combine')
                
                    dataset_list.append([Frame_csv_path, 
                                         Frame_csv_path_AddNoise, 
                                         Frame_csv_path_Convolve, 
                                         Frame_csv_path_Dropout, 
                                         Frame_csv_path_Pool,
                                         Frame_csv_path_Quantize, 
                                         Frame_csv_path_TimeWarp, 
                                         Frame_csv_path_Combine])
                    # dataset_list.append([Frame_csv_path
                    #                     ])
                    labels = []
                    try:
                        table = pd.read_csv(Frame_csv_path)
                        # table_value: channelx150
                        table_value = table.values
                    except Exception as e:
                        raise Exception(f'Error loading csv from {Frame_csv_path}') from e
                    table_value = np.array(table_value)
                    for row in table_value:
                        if np.all(row == 0):
                            labels.append(0)
                        else:
                            labels.append(action_list.index(action)+1)
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
    with open('./brain4cars_train_dataset_random_withmediapipe.json', 'w') as f_train:
        f_train.write(trainset_json)
    f_train.close()
    with open('./brain4cars_test_dataset_random_with_mediapipe.json', 'w') as f_test:
        f_test.write(testset_json)
    f_test.close()
    with open('./brain4cars_valid_dataset_random_withmediapipe.json', 'w') as f_test:
        f_test.write(validset_json)
    f_test.close()


def main_loop(dataset_dict, person_list, action_list):
    # total_video_nums = 587
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
            for action in actions:
                # 从原始视频获取landmark坐标点函数
                # posex_list, posey_list = get_landmarks(action)
                
                # 将坐标点写入csv函数
                # landmarks_writer(action, posex_list, posey_list)              
                
                # 提取瞳孔特征，shape: 150x9
                mediapipe_features =  EyeFeaturesExtract(action)
                
                # 提取mat file特征 150x9, 150x3, 150x3
                brain4cars_lane_features, Brain4cars_Lane_20frame, \
                Euler_features, brain4cars_face_features, brain4cars_speed_features, \
                Euler_features_20frame, Brain4cars_Face_20frame, Brain4cars_Speed_20frame, \
                Euler_features_20frame_AddNoise, Brain4cars_Face_20frame_AddNoise, Brain4cars_Speed_20frame_AddNoise, \
                Euler_features_20frame_Convolve, Brain4cars_Face_20frame_Convolve, Brain4cars_Speed_20frame_Convolve, \
                Euler_features_20frame_Dropout, Brain4cars_Face_20frame_Dropout, Brain4cars_Speed_20frame_Dropout, \
                Euler_features_20frame_Pool, Brain4cars_Face_20frame_Pool, Brain4cars_Speed_20frame_Pool, \
                Euler_features_20frame_Quantize, Brain4cars_Face_20frame_Quantize, Brain4cars_Speed_20frame_Quantize, \
                Euler_features_20frame_TimeWarp, Brain4cars_Face_20frame_TimeWarp, Brain4cars_Speed_20frame_TimeWarp, \
                Euler_features_20frame_Combine, Brain4cars_Face_20frame_Combine, Brain4cars_Speed_20frame_Combine = Brain4carFeatureExtract(action)
                
                # # Euler_features, brain4cars_face_features, brain4cars_speed_features, brain4cars_lane_features, \
                # # Euler_features_20frame, Brain4cars_Face_20frame, Brain4cars_Speed_20frame, Brain4cars_Lane_20frame = Brain4carFeatureExtract(action)
                # All_features                  = np.concatenate((Euler_features, brain4cars_face_features, brain4cars_lane_features, brain4cars_speed_features), axis=1)
                All_features_withmediapipe      = np.concatenate((Euler_features, brain4cars_face_features, mediapipe_features ,brain4cars_lane_features, brain4cars_speed_features), axis=1)
                # All_features_20frame          = np.concatenate((Euler_features_20frame, Brain4cars_Face_20frame, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame), axis=1)
                # All_features_20frame_AddNoise = np.concatenate((Euler_features_20frame_AddNoise, Brain4cars_Face_20frame_AddNoise, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_AddNoise), axis=1)
                # All_features_20frame_Convolve = np.concatenate((Euler_features_20frame_Convolve, Brain4cars_Face_20frame_Convolve, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_Convolve), axis=1)
                # All_features_20frame_Dropout  = np.concatenate((Euler_features_20frame_Dropout, Brain4cars_Face_20frame_Dropout, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_Dropout), axis=1)
                # All_features_20frame_Pool     = np.concatenate((Euler_features_20frame_Pool, Brain4cars_Face_20frame_Pool, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_Pool), axis=1)
                # All_features_20frame_Quantize = np.concatenate((Euler_features_20frame_Quantize, Brain4cars_Face_20frame_Quantize, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_Quantize), axis=1)
                # All_features_20frame_TimeWarp = np.concatenate((Euler_features_20frame_TimeWarp, Brain4cars_Face_20frame_TimeWarp, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_TimeWarp), axis=1)
                # All_features_20frame_Combine  = np.concatenate((Euler_features_20frame_Combine, Brain4cars_Face_20frame_Combine, Brain4cars_Lane_20frame, Brain4cars_Speed_20frame_Combine), axis=1)

                # All_features_header =['yaw', 'pitch', 'row'] +  [f'face_angle_hist_{i}' for i in range(4)] + [f'face_move_in_x_hist_{i}' for i in range(4)] \
                #                     + ["face_mean_move"] + ['left_action', 'right_action', 'intersection'] + ['mean_speed'] 
                
                All_features_withmediapipe_header = ['yaw', 'pitch', 'row'] +  [f'face_angle_hist_{i}' for i in range(4)] + [f'face_move_in_x_hist_{i}' for i in range(4)] + ["face_mean_move_x"] \
                                                  + [f'eye_angle_hist_{i}' for i in range(4)] + [f'eye_move_in_x_hist_{i}' for i in range(4)]  + ["eye_mean_move_x"] \
                                                  + ['left_action', 'right_action', 'intersection'] + ['mean_speed']

                # All_features_header_20frame = ['yaw', 'pitch', 'row'] + [f'face_angle_hist_{i}' for i in range(4)] + [f'face_move_in_x_hist_{i}' for i in range(4)] \
                #                     + ["face_mean_move_x"] + ['left_action', 'right_action', 'intersection'] + ['mean_speed'] 

                # All_features_csv_path =  action2csvpath(action, prefix='allfeatures')
                All_features_withmediapipe_csv_path = action2csvpath(action, prefix='allfeatures_withmediapipe')
                # Frame_csv_path = action2csvpath(action, prefix='20frame_features')
                # Frame_csv_path_AddNoise = action2csvpath(action, prefix='20frame_features_AddNoise')
                # Frame_csv_path_Convolve = action2csvpath(action, prefix='20frame_features_Convolve')
                # Frame_csv_path_Dropout = action2csvpath(action, prefix='20frame_features_Dropout')
                # Frame_csv_path_Pool = action2csvpath(action, prefix='20frame_features_Pool')
                # Frame_csv_path_Quantize = action2csvpath(action, prefix='20frame_features_Quantize')
                # Frame_csv_path_TimeWarp = action2csvpath(action, prefix='20frame_features_TimeWarp')
                # Frame_csv_path_Combine = action2csvpath(action, prefix='20frame_features_Combine')
                
                # ndarray2csvfile(All_features_csv_path, All_features, All_features_header)
                ndarray2csvfile(All_features_withmediapipe_csv_path, All_features_withmediapipe, All_features_withmediapipe_header)
                # ndarray2csvfile(Frame_csv_path, All_features_20frame, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_AddNoise, All_features_20frame_AddNoise, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_Convolve, All_features_20frame_Convolve, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_Dropout, All_features_20frame_Dropout, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_Pool, All_features_20frame_Pool, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_Quantize, All_features_20frame_Quantize, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_TimeWarp, All_features_20frame_TimeWarp, All_features_header_20frame)
                # ndarray2csvfile(Frame_csv_path_Combine, All_features_20frame_Combine, All_features_header_20frame)
                person_video_count += 1
                processed_video += 1
                print("The number of videos processed is {}.\n".format(processed_video))
        print("person{}的视频数量为：{}".format(p_idx, person_video_count))


def clear_csv(dir_path):
    for root, dir, files in sorted(os.walk(dir_path)):
        for file in files:
            if file.endswith('.csv'):
                if "eye_landmarks" in file:
                    os.remove(os.path.join(root, file))
                    print("已删除文件：", os.path.join(root, file))
if __name__ == "__main__":
    # clear_csv("/home/ubuntu/zsj/brain4cars_video/brain4cars_data/face_camera")
    main_loop(dataset_dict, person_list, action_list)
    # TrainTestSet_byperson(dataset_dict, person_list, action_list)
    # TrainTestSetRandom(dataset_dict, person_list, action_list, label_type='one')






