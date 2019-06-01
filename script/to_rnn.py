from util.load_data import *


train, valid = load_records()
frames = records_preprocess(train, to='csv')

print(frames[0].shape, frames[1].shape)

frames_to_dat(frames, 'data/train.dat')
