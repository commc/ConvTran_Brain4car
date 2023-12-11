import os
import sys
import cv2
import json
import csv
import logging
import random
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.io import loadmat

json_file = '/home/ubuntu/zsj/GTN-master/brain4cars_process/Brain4cars_datasetbyperson.json'
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


# 画图：
# def cart2pol(x, y):
#     """Convert from Cartesian to polar coordinates.
#     """
#     radius = np.hypot(x, y)
#     theta = np.arctan2(y, x)
#     return theta, radius

# def pol2cart(theta, radius):
#     """
#     Convert from polar to Cartesian coordinates.
#     """
#     u = radius * np.cos(theta)
#     v = radius * np.sin(theta)
#     return u, v

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
                # cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 0), -5)
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

def nparray2csvfile(csv_path, np_array, header):
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
    csv_path = csv_path.split('video_')[0] + csv_path.split('video_')[1]
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
        header.append(f'x_{index}'), header.append(f'y_{index}')
    pos_xy_array = np.column_stack((pos_x_array.flatten(), pos_y_array.flatten()))
    pos_xy_array = pos_xy_array.reshape((x_videolength, 478*2))
    nparray2csvfile(csv_path, pos_xy_array, ','.join(header))
    # np.savetxt(csv_path, pos_xy_array, delimiter=',', header=','.join(header), comments='', fmt='%.8f')
    # print(".csv save at {} successfully!".format(csv_path))

def EyeFeaturesExtract(action):
    """
    descriptions: 传入视频地址, 转为csv地址再进行关键点处理
    """
    hist_angle_values = [-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi]
    hist_distance_x = [-1e3, -2.0, 0, 2, 1e3]
    hist_distance_values = [-1e3, 2, 5, 8, 10, 1e3]

    csv_path = action.split('.')[0] + '.csv'
    csv_path = csv_path.split('video_')[0] + csv_path.split('video_')[1]
    df = pd.read_csv(csv_path)
    landmarks = df.values[:, 946:] / 2
    points_move_vec = np.diff(landmarks, axis=0)
    # 限制最大值
    points_move_vec[np.abs(points_move_vec) > 50] = 0
    frame = points_move_vec.shape[0]
    target_shape = (150, 956)
    if frame < 150:
        pad_with = [(0, target_shape[0]- frame), (0, 0)]
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
    lane_info = Data['laneInfo'].item().item().split(',')
    lane_info = np.array([int(i) for i in lane_info])
    frame_data = Data['frame_data'].item()
    landmarks = np.zeros((frame_data.shape[1], 68 * 2))
    Euler = np.zeros((frame_data.shape[1], 3))
    Speed = np.zeros((frame_data.shape[1], 1))
    for i in range(frame_data.shape[1]):
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

    # set to be 151 rows
    if frame_data.shape[1] < 151:
        landmarks_last_row = landmarks[-1, :]
        Speed_last_row = np.zeros_like(Speed[-1, :])
        Euler_last_row = np.zeros_like(Euler[-1: ])
        landmarks = np.concatenate((landmarks, np.tile(landmarks_last_row, (151 - frame_data.shape[1], 1))))
        Euler = np.concatenate((Euler, np.tile(Euler_last_row, (151 - frame_data.shape[1], 1))))
        Speed = np.concatenate((Speed, np.tile(Speed_last_row, (151 - frame_data.shape[1], 1))))
    elif frame_data.shape[1] > 151:
        landmarks = landmarks[:151, :]
        Euler = Euler[:151, :]
        Speed = Speed[:151, :]
    # extract face features
    hist_angle_values = [-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi]
    hist_distance_x = [-1e3, -2.0, 0, 2, 1e3]
    hist_distance_values = [-1e3, 2, 5, 8, 10, 1e3]
    # landmarks: 151 rows -> 150 rows
    points_move_vec = np.diff(landmarks, axis=0)
    points_move_vec[np.abs(points_move_vec) > 20] = 0
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
    """Euler angle"""
    Euler = Euler[:150, :]
    """Speed features 这里和论文中的不一样，这里的原则是每隔获取最大速度，而不是5s中共用一个最大速度"""
    num_intervals = int(Speed.shape[0] / 30)
    # Speed: 151 rows -> 150 rows
    Mean_speed = np.zeros_like(Speed[:150])
    Max_speed = np.zeros_like(Speed[:150])
    Min_speed = np.zeros_like(Speed[:150])
    for i in range(150):
        if Speed[i, :] == -1:
            Speed[i, :] = 30 / 160
        else:
            Speed[i, :] = Speed[i, :] / 160
    Speed_features = Speed[:150]
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
    # lane features 151 rows -> 150 rows
    Lane_features = np.array([left_action, right_action, intersection])
    Lane_features = np.tile(Lane_features, (150, 1))
    # 150 x 12
    Brain4cars_Face = np.concatenate((features_hist_angle_normalize, features_hist_move_in_x_normalize, features_mean_movement_x, Euler), axis=1)
    # 150 x3
    # Brain4cars_Speed = np.concatenate((Mean_speed, Max_speed, Min_speed), axis=1)
    # 150 x1
    Brain4cars_Speed = Speed_features
    # 150 x3
    Brain4cars_Lane = Lane_features
    return Brain4cars_Face, Brain4cars_Speed, Brain4cars_Lane


def TrainTestSetByperson(dataset_dict, person_list, action_list):
    train_dataset = {'annotations': []}
    test_dataset = {'annotations': []}
    action_list = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']
    test_person = ['person2', 'person4', 'person7']
    total_video_nums = 542
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


def TrainTestSetRandom(dataset_dict, person_list, action_list):
    test_num = 160
    train_dataset = {'annotations': []}
    test_dataset = {'annotations': []}
    dataset_list = []
    labels_list = []
    train_list = []
    train_labels = []
    test_list = []
    test_labels =[]
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
                dataset_list.append(All_features_csv_path)
                labels_list.append(action_list.index(action))
    test_list = random.sample(dataset_list, test_num)
    for i in range(len(test_list)):
        test_labels.append(labels_list[dataset_list.index(test_list[i])])
    
    train_list = list(set(dataset_list) - set(test_list))
    for i in range(len(train_list)):
        train_labels.append(labels_list[dataset_list.index(train_list[i])])
    
    for i in range(len(train_list)):
        train_dataset['annotations'].append({'filename': train_list[i], 'label': train_labels[i]})
    for i in range(len(test_list)):
        test_dataset['annotations'].append({'filename': test_list[i], 'label': test_labels[i]})
    
    trainset_json = json.dumps(train_dataset, indent=1)
    testset_json = json.dumps(test_dataset, indent=1)
    with open('./brain4cars_train_dataset_random.json', 'w') as f_train:
        f_train.write(trainset_json)
    f_train.close()
    with open('./brain4cars_test_dataset_random.json', 'w') as f_test:
        f_test.write(testset_json)
    f_test.close()

def main_loop(dataset_dict, person_list, action_list):
    total_video_nums = 542
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
                brain4cars_face_features, brain4cars_speed_features, brain4cars_lane_features = Brain4carFeatureExtract(action)
                All_features = np.concatenate((brain4cars_face_features, brain4cars_lane_features, brain4cars_speed_features), axis=1)
                All_features_header = ['yaw', 'pitch', 'row'] + [f'face_angle_hist_{i}' for i in range(4)] + [f'face_move_in_x_hist_{i}' for i in range(4)] \
                                    + ["face_mean_move"] \
                                    + ['left_action', 'right_action', 'intersection']

                csv_list = action.split(os.sep)
                csv_mark = csv_list[-1].replace('video_', 'allfeatures_').replace('.avi', '.csv')
                csv_list[-1] = csv_mark
                All_features_csv_path = os.sep.join(csv_list)
                nparray2csvfile(All_features_csv_path, All_features, All_features_header)
                person_video_count += 1
                processed_video += 1
                print("The number of videos processed is {}, and the remaining number of videos to be processed is {}.\n"
                    .format(processed_video, total_video_nums - processed_video))
        print("person{}的视频数量为：{}".format(p_idx, person_video_count))


if __name__ == "__main__":
    # main_loop(dataset_dict, person_list, action_list)
    # TrainTestSet(dataset_dict, person_list, action_list)
    TrainTestSetRandom(dataset_dict, person_list, action_list)






