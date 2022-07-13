from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def generate_pairs(line, window_size):
    line = np.array(line)
    line = line[:, 0]

    seqs = []
    for i in range(0, len(line), window_size):
        seq = line[i:i + window_size]
        seqs.append(seq)
    seqs += []
    seq_pairs = []
    for i in range(1, len(seqs)):
        seq_pairs.append([seqs[i - 1], seqs[i]])
    return seqs


def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0, is_label=False):
    seq_label = 0
    token_labels = []
    if is_label:
        records = line.split(",")
        seq_label = int(records[0])
        token_labels = [int(v) for v in records[1].split()]
        line =  [ln.split(",") for ln in records[2].split()]#split()以空格为分隔符，包括\n
    else:
        line = [ln.split(",") for ln in line.split()]


    # filter the line/session shorter than 10
    if len(line) <= min_len:
        return [], [], [],[]

    # max seq len
    if seq_len is not None:
        line = line[:seq_len]
        token_labels = token_labels[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.array(line)
    token_labels = np.array(token_labels)

    # if time duration exists in data
    if len(line.shape)>1 and line.shape[1] == 2:
        tim = line[:,1].astype(float)
        line = line[:, 0]

        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
    else:
        line = line.squeeze()
        # if time duration doesn't exist, then create a zero array for time
        tim = np.zeros(line.shape)

    logkey_seqs = []
    time_seq = []
    seq_labels = []
    token_label_seq = []
    for i in range(0, len(line), window_size):
        logkey_seqs.append(line[i:i + window_size])
        time_seq.append(tim[i:i + window_size])
        seq_labels.append(seq_label)
        token_label_seq.append(token_labels[i:i+window_size])

    return logkey_seqs, time_seq, seq_labels, token_label_seq

# WINDOW SIZE = 120
def generate_train_valid(data_path, window_size=20, adaptive_window=True,
                         sample_ratio=1, valid_size=0.1, output_path=None,
                         scale=None, scale_path=None, seq_len=None, min_len=0, is_label=False):
    with open(data_path, 'r') as f:
        data_iter = f.readlines()

    num_session = int(len(data_iter) * sample_ratio)
    # only even number of samples, or drop_last=True in DataLoader API
    # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # num_session += num_session % 2

    test_size = int(min(num_session, len(data_iter)) * valid_size)
    # only even number of samples
    # test_size += test_size % 2

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("="*40)

    logkey_seq_pairs = []
    time_seq_pairs = []
    label_seq_paris = []
    token_label_seq_paris = []
    session = 0
    for line in tqdm(data_iter):
        if session >= num_session:
            break
        session += 1

        #generate multiple window sequences out of one block_id sequence
        logkeys, times, seq_label, token_labels = fixed_window(line, window_size, adaptive_window, seq_len, min_len, is_label= is_label)
        logkey_seq_pairs += logkeys
        time_seq_pairs += times
        label_seq_paris +=seq_label
        token_label_seq_paris +=token_labels

    logkey_seq_pairs = np.array(logkey_seq_pairs,  dtype=object)
    time_seq_pairs = np.array(time_seq_pairs,  dtype=object)
    label_seq_paris = np.array(label_seq_paris,  dtype=object)
    token_label_seq_paris = np.array(token_label_seq_paris,  dtype=object)

    logkey_trainset, logkey_validset, time_trainset, time_validset,\
    label_trainset, label_validset,\
        token_label_trainset, token_label_validset = train_test_split(logkey_seq_pairs,
                                                      time_seq_pairs,
                                                      label_seq_paris,
                                                      token_label_seq_paris,
                                                      test_size=test_size,
                                                      random_state=1234)

    # sort seq_pairs by seq len in descending order
    train_len = list(map(len, logkey_trainset))
    valid_len = list(map(len, logkey_validset))
    # the indices of training data in seq len descending order
    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    logkey_trainset = logkey_trainset[train_sort_index]
    logkey_validset = logkey_validset[valid_sort_index]

    time_trainset = time_trainset[train_sort_index]
    time_validset = time_validset[valid_sort_index]

    label_trainset = label_trainset[train_sort_index]
    label_validset = label_validset[valid_sort_index]

    token_label_trainset = token_label_trainset[train_sort_index]
    token_label_validset = token_label_validset[valid_sort_index]

    print("="*40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("="*40)


    return logkey_trainset, logkey_validset, time_trainset, time_validset,label_trainset, label_validset,token_label_trainset, token_label_validset

