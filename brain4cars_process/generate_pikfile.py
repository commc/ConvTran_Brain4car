import os
import json
import pickle
import pandas as pd
import numpy as np

"""
_summary_
这个文件针对时间序列形状是7*cahnnel的情况，把所有数据打包成pik文件进行加载
"""


def generate_pikfile(file_path):
    if file_path.endswith('.json'):
        with open(file_path) as f:
            dataset_dict = json.load(f)
        f.close()
        dataset = dataset_dict['annotations']
        filenames = []
        labels = []
        for csv_info in dataset_dict['annotations']:
            filenames.append(csv_info['filename'])
            labels.append(csv_info['label'])
        dataset = []
        for csv_file in filenames:
            try:
                table = pd.read_csv(csv_file)
                # table_value: channelx150
                table_value = table.values
            except Exception as e:
                raise Exception(f'Error loading csv from {csv_file}') from e
            dataset.append(table_value)
        dataset = np.array(dataset)
        labels = np.array(labels)
        dataset_pik = {'features': dataset, 'labels': labels}
        pickle_path = os.path.normpath(file_path.split('.')[0] + '.pik')
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset_pik, f)


if __name__ == '__main__':
    generate_pikfile('/home/ubuntu/zsj/GTN-master/brain4cars_process/brain4cars_valid_dataset_random_20frame.json')