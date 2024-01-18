import numpy as np

def predictManeuver(time_prediction):
    time_prediction = time_prediction.detach().cpu().numpy()
    action_name = ['end_action', 'lchange', 'lturn', 'rchange', 'rturn']
    end_action_index = action_name.index('end_action')
    delta_frame = 25
    anticipation_time = 0.0
    anticipation_time_list = []
    prediction_list = []

    for row in time_prediction:
        count = 1.0
        prediction = end_action_index
        for p in row:
            anticipation_time = (len(row) * 1.0 - count) * delta_frame / 25.0
            if not p == end_action_index:
                prediction = p
                break
            count += 1.0
        prediction_list.append(prediction)
        anticipation_time_list.append(anticipation_time)
    return prediction_list, anticipation_time_list

def confusionMat(P, Y, T):
    size = np.max(Y) + 1
    P = np.array(P)
    Y = np.array(Y)
    T = np.array(T)
    confMat = np.zeros((size, size))
    TimeMat = np.zeros((size, size))
    for p, y, t in zip(P, Y, T):
        confMat[p, y] += 1.0
        TimeMat[p, y] += t
    col_sum = np.reshape(np.sum(confMat, axis=1), (size, 1))
    row_sum = np.reshape(np.sum(confMat, axis=0), (1, size))
    # zero division
    col_sum[col_sum == 0] = 1.0
    row_sum[row_sum == 0] = 1.0
    confMat_mask = np.eye(confMat.shape[0]) * (np.diag(confMat == 0))
    confMat += confMat_mask
    
    precision_confMat = confMat / np.repeat(col_sum, size, axis=1)
    recall_confMat = confMat / np.repeat(row_sum, size, axis=0)
    return confMat, precision_confMat, recall_confMat, TimeMat

