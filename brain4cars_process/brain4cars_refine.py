import os
import sys
import json
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

json_file_path = "./person_all_data.json"
dataset_dict = {f"person{i+1}": {"road_camera": {}, "face_camera": {}} for i in range(8)}
with open(json_file_path) as f:
    face_dataset_original = json.load(f)
f.close()
face_filterd_files = []
road_filterd_files = []

person_list = ['person1', 'person2', 'person3', 'person4', 'person5', 'person6', 'person7', 'person8']
activity_list = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']

def grouping(face_dataset_original):
    thred = 60
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
                # face_video_time, face_save_flag =  filter_by_video(thred, face_video_path)
                face_video_time, face_save_flag = filter_by_mat(thred, face_video_path)
                # filter video_path
                # road_video_time, road_save_flag = filter_by_video(thred, road_video_path)
                
                if not face_save_flag:
                    face_filterd_files.append(face_video_path)
                    face_filterd_files_count += 1
                    print("{}视频长度为 {:.2f}frame, 少于{}frame，被过滤掉！".format(face_video_path, face_video_time, thred))
                
                if not face_save_flag:
                    road_filterd_files.append(road_video_path)
                    road_filterd_files_count += 1
                    print("{}视频长度为 {:.2f}frame, 少于{}frame，被过滤掉！".format(road_video_path, face_video_time, thred))                
                   
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
    frame_strat = Data['frame_start'].item().item()
    frame_end = Data['frame_end'].item().item()
    total_valid_frame = frame_end - frame_strat + 1

    return total_valid_frame, total_valid_frame > thred


# 过滤掉时长少于thred秒的视频
def filter_by_video(thred ,video_path):
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



if __name__ == "__main__":
    grouping(face_dataset_original)
    dataset_dict_json = json.dumps(dataset_dict, indent=1)
    with open('./Brain4cars_datasetbyperson_alldata.json', 'w') as f:
        f.write(dataset_dict_json)
    f.close()
    # barplot('./Brain4cars_datasetbyperson.json')

    
