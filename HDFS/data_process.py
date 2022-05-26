import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from logparser import Spell, Drain
from datetime import datetime

# get [log key, delta time] as input for deeplog
input_dir  = os.path.expanduser('~/Documents/ADdatasets/HDFS_1/')
output_dir = '../output/hdfs/'  # The output directory of parsing results
log_file   = "HDFS.log"  # The input log file name

log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "hdfs_sequence.csv"

#map the event id generated by drain to a simpler incrementing number
def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
    print(log_temp_dict)
    with open (output_dir + "hdfs_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


def parser(input_dir, output_dir, log_file, log_format, type='drain'):
    if type == 'spell':
        tau        = 0.5  # Message type threshold (default: 0.5)
        regex      = [
            "(/[-\w]+)+", #replace file path with *
            "(?<=blk_)[-\d]+" #replace block_id with *

        ]  # Regular expression list for optional preprocessing (default: [])

        parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=False)
        parser.parse(log_file)

    elif type == 'drain':
        regex = [
            r"(?<=blk_)[-\d]+", # block_id
            r'\d+\.\d+\.\d+\.\d+',  # IP
            r"(/[-\w]+)+",  # file path
            #r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.5  # Similarity threshold
        depth = 5  # Depth of all leaf nodes


        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=False)
        parser.parse(log_file)


def hdfs_sampling(log_file, window='session', if_time=False):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_file)
    df = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})
    #replace the event id by new mapping
    with open(output_dir + "hdfs_log_templates.json", "r") as f:
        event_num = json.load(f)
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    data_dict = defaultdict(list) #preserve insertion order of items

    time_dict = defaultdict(list)  # preserve insertion order of items
    dur_dict = defaultdict(list)  # preserve insertion order of items
    for idx, row in tqdm(df.iterrows()):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])   # find the block id
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set: #append the current event into relevant block ids
            data_dict[blk_Id].append(row["EventId"])  # find the event sequence of each block id

            if if_time:
                date_time_obj = datetime.strptime(row["Date"]+" "+row["Time"], '%y%m%d %H%M%S')
                time_dict[blk_Id].append(date_time_obj)
                length = len(time_dict[blk_Id])
                delta_sec = 0
                if length>1:
                    delta_sec = int((date_time_obj-time_dict[blk_Id][length-2]).total_seconds())
                dur_dict[blk_Id].append(delta_sec)
                assert len(dur_dict[blk_Id])==len(data_dict[blk_Id])==len(time_dict[blk_Id])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    if if_time:
        dur_df = pd.DataFrame(list(dur_dict.items()), columns=['BlockId', 'TimeDur'])
        time_df = pd.DataFrame(list(time_dict.items()), columns=['BlockId', 'TimeStamp'])
        result = pd.merge(data_df, dur_df, on=["BlockId"])
        result1 = pd.merge(result, time_df, on = ["BlockId"])
        result1.to_csv(log_sequence_file, index=False)
    else:
        data_df.to_csv(log_sequence_file, index=False)
    print("hdfs sampling done")


def generate_train_test(hdfs_sequence_file, n=None, ratio=0.3, dirty_ratio = 0.0, if_time=False):
    blk_label_dict = {}
    blk_label_file = os.path.join(input_dir, "anomaly_label.csv")
    blk_df = pd.read_csv(blk_label_file)
    for _ , row in tqdm(blk_df.iterrows()):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    seq = pd.read_csv(hdfs_sequence_file)
    seq["Label"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x)) #add label to the sequence of each blockid

    #obtain normal and abnormal sequences

    normal_seq = seq[seq["Label"] == 0]["EventSequence"]

    normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data
    abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    if if_time:
        normal_seq["TimeDur"] = seq[seq["Label"] == 0]["TimeDur"]
        abnormal_seq["TimeDur"]=seq[seq["Label"] == 1]["TimeDur"]

    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)

    # each sequence is [eventid, timeduration eventid, timeduration...]

    train_len = n if n else int(normal_len * ratio)
    print("normal size {0}, abnormal size {1}, training size {2}".format(normal_len, abnormal_len, train_len))

    normal_train_num = int(train_len*(1-dirty_ratio))
    # add normal sequences into the training data set
    train = normal_seq.iloc[:normal_train_num]
    # add abnormal sequences into the training data set
    train = train.append(abnormal_seq[:train_len - normal_train_num])
    # shuffle
    train = train.sample(frac=1, random_state=20)

    test_normal = normal_seq.iloc[int(train_len*(1-dirty_ratio)):]
    test_abnormal = abnormal_seq[train_len - int(train_len*(1-dirty_ratio)):]

    dirty_output_dir = ""
    if dirty_ratio>0.0:
        dirty_output_dir="dirty_"+str(dirty_ratio)+"/"
        folder = os.path.exists(output_dir+dirty_output_dir)
        if not folder:
            os.makedirs(dirty_output_dir)
        else:
            print(dirty_output_dir +" exists!")

    df_to_file(train, output_dir +dirty_output_dir+ "train", if_time)
    df_to_file(test_normal, output_dir +dirty_output_dir+ "test_normal", if_time)
    df_to_file(test_abnormal, output_dir +dirty_output_dir+ "test_abnormal", if_time)
    print("generate train test data done")



def df_to_file(df, file_name, if_time=False):
    with open(file_name, 'w') as f:
        df = df.reset_index()  # make sure indexes pair with number of rows
        for index, row in df.iterrows():
            line = ""
            seq_list = eval(row["EventSequence"])  #eval 将一个list的str转化为list
            seq_len = len(seq_list)
            if if_time:
                dur_list = eval(row["TimeDur"])
                assert seq_len==len(dur_list)

            for i in range(seq_len):
                line+=str(seq_list[i])
                if if_time:
                    line+=","+str(dur_list[i])
                if i!=seq_len-1:
                    line+=" "
            f.write(line)
            f.write('\n')


if __name__ == "__main__":
    # 1. parse HDFS log
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    # parser(input_dir, output_dir, log_file, log_format, 'drain')
    # mapping()
    # hdfs_sampling(log_structured_file, if_time=False)
    generate_train_test(log_sequence_file, dirty_ratio = 0.05, n=4855, if_time=False)
    # generate_train_test(log_sequence_file, n=4855, dirty_ratio=0.05)