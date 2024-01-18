import os
import json
import pickle
import pandas as pd
import numpy as np

"""
_summary_
这个文件针对时间序列形状是7*cahnnel的情况，把所有数据打包成pik文件进行加载
"""


def generate_pikfile(file_path, data_type):
    if file_path.endswith('.json'):
        with open(file_path) as f:
            dataset_dict = json.load(f)
        f.close()
        dataset = dataset_dict['annotations']
        filenames = []
        labels = []
        labels_tmp = []
        for csv_info in dataset_dict['annotations']:
            filenames.append(csv_info['filename'])
            labels_tmp.append(csv_info['label'])
        dataset = []
        if data_type == 'train':
            for idx, csv_files in enumerate(filenames):
                for csv_file in csv_files:
                    try:
                        table = pd.read_csv(csv_file)
                        # table_value: channelx150
                        table_value = table.values
                    except Exception as e:
                        raise Exception(f'Error loading csv from {csv_file}') from e
                    dataset.append(table_value)
                    labels.append(labels_tmp[idx])
        else:
            for idx, csv_files in enumerate(filenames):
                try:
                    if type(csv_files) == list:
                        table = pd.read_csv(csv_files[0])
                    else:
                        table = pd.read_csv(csv_files)
                    table_value = table.values
                except Exception as e:
                        raise Exception(f'Error loading csv from {csv_files[0]}') from e
                dataset.append(table_value)
            labels = labels_tmp             
        dataset = np.array(dataset)
        labels = np.array(labels)
        dataset_pik = {'features': dataset, 'labels': labels}
        pickle_path = os.path.normpath(file_path.split('.')[0] + '_aug.pik')
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset_pik, f)


if __name__ == '__main__':
    train_path = '/home/ubuntu/zsj/GTN-master/dataset/annotations/trainset/brain4cars_train_dataset_random_20frame_aug.json'
    valid_path = '/home/ubuntu/zsj/GTN-master/dataset/annotations/validset/brain4cars_valid_dataset_random_20frame_aug.json'
    test_path = '/home/ubuntu/zsj/GTN-master/dataset/annotations/testset/brain4cars_test_dataset_random_20frame_aug.json'
    generate_pikfile(train_path, data_type='train')
    generate_pikfile(valid_path, data_type='valid')
    # generate_pikfile(test_path, data_type='test')