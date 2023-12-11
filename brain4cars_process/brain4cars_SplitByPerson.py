import os
import re
import json
import pandas as pd
import numpy as np

excel_path = "/home/ubuntu/zsj/brain4cars_video/time.xlsx"
source_path = os.path.normpath("/home/ubuntu/zsj/brain4cars_video/brain4cars_data")
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
with open('/home/ubuntu/zsj/GTN-master/brain4cars_process/person.json', 'w') as f_train:
    f_train.write(person_json_data)
f_train.close()