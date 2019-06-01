from util.load_data import *

def test_records_preprocess():
    train, valid = load_records()
    print(records_preprocess(train, to='csv')[0])

if __name__ == "__main__":
    test_records_preprocess()
