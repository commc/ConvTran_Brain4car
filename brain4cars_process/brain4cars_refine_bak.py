import os
import json
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

json_file_path = "/home/ubuntu/zsj/GTN-master/brain4cars_process/person.json"
dataset_dict = {f"person{i+1}": {"road_camera": {}, "face_camera": {}} for i in range(8)}
with open(json_file_path) as f:
    face_dataset_original = json.load(f)
f.close()
face_filterd_files = []
road_filterd_files = []

person_list = ['person1', 'person2', 'person3', 'person4', 'person5', 'person6', 'person7', 'person8']
activity_list = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']

def grouping(face_dataset_original):
    thred = 3.0
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
                face_video_time, face_save_flag =  filter(thred, face_video_path)
                # filter video_path
                road_video_time, road_save_flag = filter(thred, road_video_path)
                
                if not face_save_flag:
                    face_filterd_files.append(face_video_path)
                    face_filterd_files_count += 1
                    print("{}视频长度为 {:.2f}s,少于{}s，被过滤掉！".format(face_video_path, face_video_time, thred))
                
                if not road_save_flag:
                    road_filterd_files.append(road_video_path)
                    road_filterd_files_count += 1
                    print("{}视频长度为 {:.2f}s,少于{}s，被过滤掉！".format(road_video_path, road_video_time, thred))                
                   
                if face_save_flag and road_save_flag:
                    remained_video += 1
                    face_activity_list.append(face_video_path)
                    road_activity_list.append(road_video_path)
            
            if activity not in dataset_dict[person][face_camera]:
                dataset_dict[person][face_camera][activity] = face_activity_list
                dataset_dict[person][road_camera][activity] = road_activity_list
            else:
                dataset_dict[person][face_camera][activity] += face_activity_list
                dataset_dict[person][road_camera][activity] += road_activity_list
    print("视频过滤的标准：视频长度少于{}s的被过滤掉！".format(thred))
    print("face视频总数是{}, 被过滤掉视频总数是{}, 剩余视频数量{}".format(face_total_files_count, face_filterd_files_count, remained_video))
    print("road视频总数是{}, 被过滤掉视频总数是{}, 剩余视频数量{}".format(road_total_files_count, road_filterd_files_count, remained_video))

# 过滤掉时长少于2秒的视频
def filter(thred ,video_path):
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
    with open('/home/ubuntu/zsj/GTN-master/brain4cars_process/Brain4cars_datasetbyperson.json', 'w') as f:
        f.write(dataset_dict_json)
    f.close()
    # barplot('./Brain4cars_datasetbyperson.json')

    
