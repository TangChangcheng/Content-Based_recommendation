import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix, csr_matrix


data_path = '/Users/ice/Data/cbvrp-acmmm-2019-release-train-val/cbvrp_data/track_1_series'
feature_path = '{}/train_val/features'.format(data_path)
record_path = '{}/train_val/records'.format(data_path)
ith_frame_ft = feature_path + '/frame_ft/{}.npy'
ith_audio_ft = feature_path + '/audio_ft/{}.npy'
ith_video_ft = feature_path + '/video_ft/{}.npy'

train_record = record_path + '/series_train.csv'
valid_record = record_path + '/series_valid.csv'


def load_feature(i):

    frame = np.load(ith_frame_ft.format(i))
    video = np.load(ith_video_ft.format(i))
    audio = np.load(ith_audio_ft.format(i))
    return frame, video, audio


def load_all_features():
    i = 2
    while True:
        try:
            yield load_feature(i)
            i += 1
        except expression as identifier:
            raise StopIteration


def load_records():
    train = pd.read_csv(train_record, header=-1)
    valid = pd.read_csv(valid_record, header=-1)
    return train, valid


def records_preprocess(records, to='matrix', title='uirt'):
    records = np.asarray(records).astype(int)
    y_true = records[:, 0]
    # [0, 1, 0, ...]
    liked = records[:, 1:-1]
    # [[ui1, ui2, ...], ...]
    recommended = records[:, -1]
    data_count = liked.shape[0] * (liked.shape[1])
    rec_count = recommended.shape[0] * 1
    row = np.empty(data_count, dtype=np.int)
    col = np.empty(data_count, dtype=np.int)
    data = np.empty(data_count, dtype=np.int)
    timestamp = np.empty(data_count, dtype=np.int)

    gt_row = np.empty(rec_count, dtype=np.int)
    gt_col = np.empty(rec_count, dtype=np.int)
    gt_data = np.empty(rec_count, dtype=np.int)
    gt_timestamp = np.empty(rec_count, dtype=np.int)

    i = 0
    for _row, (l, r, y) in enumerate(zip(liked, recommended, y_true)):
        gt_row[_row] = _row
        gt_col[_row] = (r)
        gt_data[_row] =  (y) if int(y) == 1 else -1
        for u in l:
            row[i] = _row
            col[i] = (u)
            data[i] = 1
            timestamp[i] = i % len(l)
            i += 1
        gt_timestamp[_row] = len(l)
    if to == 'matrix':
        return coo_matrix((data, (row, col))), coo_matrix((gt_data, (gt_row, gt_col)))
    elif to == 'csv':
        t_map = {'t': 'Timestamp', 'u': 'UserID', 'i': 'ItemID', 'r': 'Rating'}
        order = [t_map[t] for t in title]
        return pd.DataFrame({'UserID': row, 'ItemID': col, 'Rating': data, 'Timestamp': timestamp}).loc[:, order], \
        pd.DataFrame({'UserID': gt_row, 'ItemID': gt_col, 'Rating': gt_data, 'Timestamp': gt_timestamp}).loc[:, order]

def frames_to_dat(frames, path):
    frame = pd.concat(frames, axis=0)
    print('concat frame: ', frame.shape)
    frame.to_csv(path, index=False, header=None)
