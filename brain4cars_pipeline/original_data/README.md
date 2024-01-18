## person_all_data.json
This json file contains the dir path of all the data in the dataset. There are a total of 594 raw data entries, and the data addresses correspond to the folder addresses of each video. With these addresses, it is easy to obtain both driver and driving scenarios videos.
I have divided the data based on drivers, which facilitates the application of federated learning algorithms. 

The data structure in the json file is as follows:
|-person1
|---->|end_action
|     |---->"/home/ubuntu/zsj/brain4cars_video/brain4cars_data/face_camera/end_action/20141019_091035_1106_1256"
|      ...
|
|---->|lchange
|     |---->"/home/ubuntu/zsj/brain4cars_video/brain4cars_data/face_camera/lchange/20141019_132535_1229_1379"
|      ...
|
|-person2
...

## Brain4cars_datasetbyperson_alldata.json
The JSON file is generated through the ["refine.py"] script. The purpose of "refine.py" is to filter out data that does not meet the criteria, resulting in a file containing various video addresses. 
The structure of the generated JSON file is as follows:
|-person1
|---->|road_camera
|     |---->|end_action
|     |     |---->"/home/ubuntu/zsj/brain4cars_video/brain4cars_data/road_camera/end_action/20141019_091035_1106_1256.avi"
|     |     ...
|     |
|     |---->|lchange
|     |     |---->"/home/ubuntu/zsj/brain4cars_video/brain4cars_data/road_camera/lchange/20141019_132535_1229_1379.avi"
|     |     ...
|
|---->|face_camera
|     |---->|end_action
|     |     |---->"/home/ubuntu/zsj/brain4cars_video/brain4cars_data/face_camera/end_action/20141019_091035_1106_1256/video_20141019_091035_1106_1256.avi"
|     |     ...
|     |
|     |---->|lchange
|     |     |---->"/home/ubuntu/zsj/brain4cars_video/brain4cars_data/road_camera/lchange/20141019_132535_1229_1379.avi"
|     |     ...
|
|-person2
...