import sys


sys.path.append("../")
# sys.path.append("../../")
#
# import os
# dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, '../deeplog')


import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor, Trainer
from logdeep.tools.utils import *

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

options["window_size"] = 128
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512 # for position embedding
options["min_len"] = 1

options["mask_ratio"] = 0.5
options["is_robust"] = False

options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

# features
options["is_logkey"] = True
options["is_time"] = False
options["is_label"] = True   #if train with labels

options["hypersphere_loss"] = False
options["hypersphere_loss_test"] = False

options["scale"] = None # MinMaxScaler()

# model
options["hidden"] = 256 # embedding size
options["layers"] = 4
options["attn_heads"] = 4
options["output_attentions"] = True
options["tracking"] = True

options["epochs"] = 120  # 200
options["n_epochs_stop"] = 10
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 4
options["lr"] = 1e-4
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
options["num_candidates"] = 15
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')
    train_parser.add_argument("--robust_method", type=str, default="None")
    train_parser.add_argument("--weight_on", type=str, default="loss")
    train_parser.add_argument("--weight_method", type=str, default="None")
    train_parser.add_argument("--if_clean", type=str, default="None")

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std", type=float, default=1)
    predict_parser.add_argument("--robust_method", type=str, default="None")
    predict_parser.add_argument("--weight_on", type=str, default="loss")
    predict_parser.add_argument("--weight_method", type=str, default="None")
    predict_parser.add_argument("--if_clean", type=str, default="None")

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')
    vocab_parser.add_argument("-s", "--vocab_size", type=int, default=None)
    vocab_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    vocab_parser.add_argument("-m", "--min_freq", type=int, default=1)

    args = parser.parse_args()
    print("*****************************************************************************")
    print("arguments", args)
    # Trainer(options).train()
    # Predictor(options).predict()

    print("device", options["device"])
    print("features logkey:{} time: {}".format(options["is_logkey"], options["is_time"]))
    print("mask ratio", options["mask_ratio"])

    if args.mode == 'train':

        options["robust_method"] = args.robust_method
        options["weight_method"] = args.weight_method
        options["weight_on"] = args.weight_on  # loss or attention
        options["if_clean"] = args.if_clean

        options["output_dir"] = "../output/bgl/datasets/"+options["if_clean"]+"/"
        options["data_dir"] = options["output_dir"]

        options["model_dir"] = options["output_dir"] +options["weight_on"]+"/"+ options["robust_method"] +"/"

        if options["robust_method"]=="activeBias":
            options["model_dir"] =options["model_dir"] +options["weight_method"]+"/"

        options["model_path"] = options["model_dir"] + "best_bert.pth"
        options["scale_path"] = options["model_dir"] + "scale.pkl"
        options["train_vocab"] = options["data_dir"] + 'train'
        options["vocab_path"] = options["data_dir"] + "vocab.pkl"

        if not os.path.exists(options['model_dir']):
            os.makedirs(options['model_dir'], exist_ok=True)
        Trainer(options).train()

    elif args.mode == 'predict':
        options["robust_method"] = args.robust_method
        options["weight_method"] = args.weight_method
        options["weight_on"] = args.weight_on  # loss or attention
        options["if_clean"] = args.if_clean

        options["output_dir"] = "../output/bgl/datasets/" + options["if_clean"] + "/"
        options["data_dir"] = options["output_dir"]

        options["model_dir"] = options["output_dir"] + options["weight_on"] + "/" + options["robust_method"] + "/"

        if options["robust_method"] == "activeBias":
            options["model_dir"] = options["model_dir"] + options["weight_method"] + "/"

        options["model_path"] = options["model_dir"] + "best_bert.pth"
        options["scale_path"] = options["model_dir"] + "scale.pkl"
        options["vocab_path"] = options["data_dir"] + "vocab.pkl"

        if not os.path.exists(options['model_dir']):
            os.makedirs(options['model_dir'], exist_ok=True)
        else:
            Predictor(options).predict()

    elif args.mode == 'vocab':
        options["train_vocab"] = "../output/bgl/datasets/noisy_small/train"
        options["vocab_path"] = "../output/bgl/datasets/noisy_small/vocab.pkl"
        with open(options["train_vocab"], 'r') as f:
            logs = f.readlines()
        vocab = WordVocab(logs, if_seq_label=True, if_token_label=True)
        print("vocab_size", len(vocab))
        vocab.save_vocab(options["vocab_path"])





