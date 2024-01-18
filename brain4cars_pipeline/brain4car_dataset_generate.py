import os
import sys
sys.path.append('/home/dulab/ML/Code_project/GTN-master')
import json
from utils.brain4cars_datasetprocess import splitbyperson
from utils.brain4cars_datasetprocess import filter_dataset
from utils.brain4cars_datasetprocess import clear_csv
from utils.brain4cars_datasetprocess import generated_feature_csv
from utils.brain4cars_datasetprocess import TrainTestSetRandom
from utils.brain4cars_datasetprocess import TrainTestSetKFolder

"""
这个文py件产生从原始数据集到能够用于训练的json文件    
"""

# original params
excel_path = "/home/dulab/ML/Code_project/brain4cars_video/time.xlsx"
dataset_source_path = os.path.normpath("/home/dulab/ML/Code_project/brain4cars_video/brain4cars_data")
origin_save_path = "./brain4cars_process/"

# filter params
original_dataset_json_path = "./brain4cars_pipeline/original_data/person_all_data.json"
filter_save_path = "./brain4cars_pipeline/temp_data/"

# clear_csv
clear_csv_path = "/home/dulab/ML/Code_project/brain4cars_video/brain4cars_data/face_camera"
csv_name = "allfeatures_withmediapipeandeye"
# generate features csv
filterd_dataset_json_path = './brain4cars_pipeline/original_data/Brain4cars_datasetbyperson_alldata_thred_50.json'

# generate dataset
dataset_save_path = "./brain4cars_pipeline/temp_data/"

if __name__ == "__main__":
    # uncomment the step need to be executed, when the code has been exectued comment it again
    
    # step.1
    # splitbyperson(excel_path, dataset_source_path, origin_save_path)
    # step.2
    # filter_dataset(original_dataset_json_path, filter_save_path)
    # step.3 
    # clear_csv(clear_csv_path, csv_name)
    generated_feature_csv(filterd_dataset_json_path)
    # step.4
    # TrainTestSetRandom(filterd_dataset_json_path, dataset_save_path, label_type='multi')
    # TrainTestSetKFolder(filterd_dataset_json_path, dataset_save_path, k=5)

