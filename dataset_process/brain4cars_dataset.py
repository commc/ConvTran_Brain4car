import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Brain4carDataset(Dataset):
    def __init__(self, file_path):
        super(Brain4carDataset, self).__init__()
        self.dataset, \
        self.labels, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.dataset_len, \
        self.max_length_sample_inTest = self.get_dataset_labels_from_fils(file_path)

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]

    def __len__(self):
        return self.dataset_len
    
    def get_dataset_labels_from_fils(self, file_path):
        if file_path.endswith('.json'):
            try:
                with open(file_path, 'r') as f:
                    dataset_dict = json.load(f)
            except Exception as e:
                raise Exception(f'Error loading json from {file_path}') from e
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
            input_len = dataset.shape[1]
            input_channel = dataset.shape[-1]
            output_len = len(tuple(set(labels)))
            dataset_len = dataset.shape[0]
            max_length_sample_inTest = []
            labels = torch.Tensor(labels)    
            dataset = torch.Tensor(dataset).float()
            for i in dataset:
                max_length_sample_inTest.append(i)
        elif file_path.endswith('.pik'):
            try:
                with open(file_path, 'rb') as f:
                    dataset_dict = pickle.load(f, encoding='latin1')
            except Exception as e:
                raise Exception(f'Error loading pickle from {file_path}') from e
            dataset = np.array(dataset_dict['features'])
            dataset_shape = dataset.shape
            # eliminate redundancy axis
            if 1 in dataset_shape:
                dataset = np.squeeze(dataset)
            labels = np.array(dataset_dict['labels'])
            labels_shape = labels.shape
            # eliminate redundancy axis
            if 1 in labels_shape:
                labels = np.squeeze(labels)
            
            if dataset_shape[0] < dataset_shape[1]:
                dataset = dataset.transpose((1, 0, 2))
            if labels_shape[0] < labels_shape[1]:
                labels = labels.transpose((1, 0))
            
            label_mask = labels == 0
            labels = np.average(labels, axis=1, weights=~label_mask) - 1
            
            input_len = dataset.shape[1]
            input_channel = dataset.shape[-1]
            output_len = len(tuple(set(labels)))
            dataset_len = dataset.shape[0]
            max_length_sample_inTest = []
            labels = torch.Tensor(labels)    
            dataset = torch.Tensor(dataset).float()
            for i in dataset:
                max_length_sample_inTest.append(i)
            
        else:
            raise FileNotFoundError

        return dataset, labels, input_len, input_channel, output_len, dataset_len, max_length_sample_inTest
            


if __name__ == '__main__':
    dataset = Brain4carDataset('/home/ubuntu/zsj/GTN-master/dataset/annotations/testset/test_data_846483_fold2.pik')
    print(dataset)


        

