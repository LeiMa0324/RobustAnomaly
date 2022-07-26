import os.path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .optim_schedule import ScheduledOptim
from ..model import BERTLog, BERT
from training_tracker import Tracker


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int, epochs: int, model_dir,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, is_logkey=True, is_time=False, is_robust=True,is_label=False,
                 hypersphere_loss=False, robust_method = None, weight_method = None, weight_on= "loss", warm_up_epochs = 5  ):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param valid_dataloader: valid dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.robust_method = robust_method
        self.warm_up_epochs = warm_up_epochs
        self.weight_on = weight_on
        self.model_dir = model_dir


        # This BERT model will be saved every epoch
        self.bert = bert
        self.tracker = None
        # if self.robust_method =="activeBias":
        self.tracker = Tracker(self.warm_up_epochs, self.bert.n_layers, self.bert.attn_heads, weight_method= weight_method, output_dir =self.model_dir )


        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLog(bert, vocab_size, output_attentions = False).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and valid data loader
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optim = None
        self.optim_schedule = None
        self.init_optimizer()


        # Using Negative Log Likelihood Loss function for predicting the masked_token
        # ignore the loss of label = 0
        self.criterion = nn.NLLLoss(ignore_index=0,reduce=False)  #return a scalar
        self.time_criterion = nn.MSELoss()
        self.hyper_criterion = nn.MSELoss()

        # deep SVDD hyperparameters
        self.hypersphere_loss = hypersphere_loss
        self.radius = 0
        self.hyper_center = None
        self.nu = 0.25
        # self.objective = "soft-boundary"
        self.objective = None

        self.log_freq = log_freq

        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss", 'hyper_loss']},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss", 'hyper_loss']}
        }

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.is_logkey = is_logkey
        self.is_time = is_time
        self.is_robust = is_robust
        self.epochs = epochs
        self.is_label = is_label

    def init_optimizer(self):
        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=self.warmup_steps)

    def train(self, epoch):
        if self.robust_method=="vanilla":
            return self.vanilla_iteration(epoch, self.train_data, start_train=True)
        elif self.robust_method == "activeBias":
            if self.weight_on=="loss":
                return self.active_iteration(epoch, self.train_data, start_train=True, weight_loss=True)
            elif self.weight_on=="attention":
                return self.active_iteration(epoch, self.train_data, start_train=True, weight_attention=True)
            elif self.weight_on=="attention-loss":
                return self.active_iteration(epoch, self.train_data, start_train=True, weight_loss=True, weight_attention=True)
            else:
                print("Please specify a weight method: loss or attention!")
        elif self.robust_method == "token_label":
            if self.weight_on == "loss":
                return self.iteration_weight_token_label(epoch, self.train_data, start_train=True, weight_loss=True)
            elif self.weight_on == "attention":
                return self.iteration_weight_token_label(epoch, self.train_data, start_train=True, weight_attention=True)
            elif self.weight_on == "attention-loss":
                return self.iteration_weight_token_label(epoch, self.train_data, start_train=True, weight_attention=True, weight_loss=True)


    def valid(self, epoch):
        if self.robust_method=="vanilla":
            return self.vanilla_iteration(epoch, self.valid_data, start_train=False)
        elif self.robust_method == "activeBias":
            if self.weight_on=="loss":
                return self.active_iteration(epoch, self.valid_data, start_train=False, weight_loss=True)
            elif self.weight_on=="attention":
                return self.active_iteration(epoch, self.valid_data, start_train=False, weight_attention=True)
            elif self.weight_on=="attention-loss":
                return self.active_iteration(epoch, self.valid_data, start_train=False, weight_loss=True, weight_attention=True)
            else:
                print("Please specify a weight method: loss or attention!")
        elif self.robust_method == "token_label":
            if self.weight_on == "loss":
                return self.iteration_weight_token_label(epoch, self.valid_data, start_train=False, weight_loss=True)
            elif self.weight_on == "attention":
                return self.iteration_weight_token_label(epoch, self.valid_data, start_train=False, weight_attention=True)
            elif self.weight_on == "attention-loss":
                return self.iteration_weight_token_label(epoch, self.valid_data, start_train=False, weight_attention=True, weight_loss=True)
            else:
                print("Please specify a weight method: loss or attention!")


    # vanilla iteration without robust method
    def vanilla_iteration(self, epoch, data_loader, start_train):
        str_code = "train" if start_train else "valid"
        lr = self.optim.state_dict()['param_groups'][0]['lr']
        start = time.strftime("%H:%M:%S")
        self.log[str_code]['lr'].append(lr)
        self.log[str_code]['time'].append(start)

        # Setting the tqdm progress bar
        totol_length = len(data_loader)
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)

        # tracking info
        train_log = pd.DataFrame()
        valid_log = pd.DataFrame()

        total_loss = 0.0  # total loss
        total_logkey_loss = 0.0  # the loss of next log key prediction
        total_hyper_loss = 0.0  # the loss of next log key prediction
        total_dist = []

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            result = self.model.forward(data["bert_input"], data["time_input"])

            # return the prediction of the masked log key and the time interval
            mask_lm_output = result["logkey_output"]

            # 2-2. NLLLoss of predicting masked token word ignore_index = 0 to ignore unmasked tokens
            # since the last layer is a logsoftmax, here use NLlloss, if its a soft max layer, use CrossEntropy loss instead
            mask_loss = torch.tensor(0) if not self.is_logkey else self.criterion(mask_lm_output.transpose(1, 2),
                                                                                  data["bert_label"])
            softmax = nn.Softmax(dim=2)
            probabilities = softmax(mask_lm_output)
            # compute the weighted loss
            total_logkey_loss += mask_loss.sum()  # logkey loss is the sum of mask loss

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = mask_loss.sum()
            total_loss += loss.item()

            if str_code == "train":
                train_log = pd.concat([train_log, self.process_batch_log(probabilities, data, mask_lm_output)])
            else:
                valid_log = pd.concat([valid_log, self.process_batch_log(probabilities, data, mask_lm_output)])

            # 3. backward and optimization only in train
            if start_train:
                self.optim_schedule.zero_grad()  # reset the gradient
                loss.backward()  # back propagation
                self.optim_schedule.step_and_update_lr()

        avg_loss = total_loss / totol_length  # after the epoch, calculate the avg loss of this epoch
        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)
        self.log[str_code]['hyper_loss'].append(0.0)
        print("Epoch: {} | phase: {}, learning rate ={}, loss={}".format(epoch, str_code, lr , avg_loss))
        # print(f"logkey weighted loss: {total_logkey_loss / totol_length}\n")
        if start_train:
            self.tracker.load_tracking(train_log, epoch)  # load the result of this batch
            if epoch >= self.warm_up_epochs:
                diff_weights = pd.DataFrame()
                diff_weights["token_label"]= self.tracker.token_labels[str_code]
                diff_weights["token_loss"] = self.tracker.token_loss[str_code]
                self.tracker.weight_by_difficulty(str_code)
                diff_weights["weight_by_difficulty"] = self.tracker.get_weights_for_Seq(seq_indices=None, seq_len=None, code_str=str_code)
                self.tracker.weight_by_history_entropy(str_code)
                diff_weights["weight_by_history_entropy"] = self.tracker.get_weights_for_Seq(seq_indices=None, seq_len=None,
                                                                                        code_str=str_code)
                diff_weights.to_csv(self.model_dir+f"diff_weights_{epoch}.csv")

        return avg_loss, total_dist

    # iteration with active bias method, weight the loss
    def active_iteration(self, epoch, data_loader, start_train, weight_loss=False, weight_attention=False):

        str_code = "train" if start_train else "valid"

        lr = self.optim.state_dict()['param_groups'][0]['lr']
        start = time.strftime("%H:%M:%S")
        self.log[str_code]['lr'].append(lr)
        self.log[str_code]['time'].append(start)

        # Setting the tqdm progress bar
        totol_length = len(data_loader)
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)

        # tracking info
        train_log = pd.DataFrame()
        valid_log = pd.DataFrame()

        total_loss = 0.0   # total loss
        total_dist = []

        for i, data in data_iter:
            total_loss, train_log, valid_log =\
                self.batch_run( epoch, data, str_code, train_log, valid_log, total_loss, weight_loss=weight_loss, weight_attention=weight_attention)


        avg_loss = total_loss / totol_length  # after the epoch, calculate the avg loss of this epoch
        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)
        print("Epoch: {} | phase: {}, learning rate ={}, loss={}".format(epoch, str_code, lr , avg_loss))
        if start_train:
            self.tracker.load_tracking(train_log, epoch)  # load the result of this batch
            if epoch >= self.warm_up_epochs:
                self.tracker.compute_weight()
        else:
            self.tracker.load_tracking(valid_log, epoch, "valid")  # load the result of this epoch
            if epoch >= self.warm_up_epochs:
                self.tracker.compute_weight("valid")

        return avg_loss, total_dist


    def iteration_weight_token_label(self, epoch, data_loader, start_train,
                                     weight_attention= False, weight_loss=False):

        str_code = "train" if start_train else "valid"

        lr = self.optim.state_dict()['param_groups'][0]['lr']
        start = time.strftime("%H:%M:%S")
        self.log[str_code]['lr'].append(lr)
        self.log[str_code]['time'].append(start)

        # Setting the tqdm progress bar
        totol_length = len(data_loader)
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)

        total_loss = 0.0   # total loss
        total_dist = []
        total_hyper_loss = 0.0 # the loss of hypersphere

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            # (batchsize, seq_len)->(batchsize, seq_len, 1)
            weights = (1-data["token_label"]).unsqueeze(2).float()
            if weight_attention:
                attention_weights = torch.matmul(weights, weights.transpose(-2, -1))
                result = self.model.forward(data["bert_input"], data["time_input"], attention_weights)
            else:
                result = self.model.forward(data["bert_input"], data["time_input"])

            # return the prediction of the masked log key and the time interval
            #(batch_size, seq_len, vocab_size)
            mask_lm_output = result["logkey_output"]

            if self.hypersphere_loss:
                # the distance beween cls and the input
                # cls: (batch_size, embedding_size)
                # hyper_center after expansion (batch_size, embedding_size)
                hyper_loss = self.hyper_criterion(result["cls_output"].squeeze(), self.hyper_center.expand(data["bert_input"].shape[0], -1))

                dist = torch.sum((result["cls_output"] - self.hyper_center) ** 2, dim=1)
                total_dist += dist.cpu().tolist()
                total_hyper_loss += hyper_loss.item()

            # 2-2. NLLLoss of predicting masked token word ignore_index = 0 to ignore unmasked tokens
            # since the last layer is a logsoftmax, here use NLlloss, if its a soft max layer, use CrossEntropy loss instead
            mask_loss = torch.tensor(0) if not self.is_logkey else self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
            if weight_loss:
                mask_loss = mask_loss*weights.squeeze()

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = mask_loss.sum()
            total_loss += loss.item()

            # 3. backward and optimization only in train
            if start_train:
                self.optim_schedule.zero_grad()  #reset the gradient
                loss.backward()  # back propagation
                self.optim_schedule.step_and_update_lr()

        avg_loss = total_loss / totol_length  # after the epoch, calculate the avg loss of this epoch
        avg_hyper_loss = total_hyper_loss / totol_length
        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)
        self.log[str_code]['hyper_loss'].append(avg_hyper_loss)
        print("Epoch: {} | phase: {}, learning rate ={}, loss={}".format(epoch, str_code, lr , avg_loss))
        print(f"hyper loss: {avg_hyper_loss}\n")

        return avg_loss, total_dist


    def batch_run(self, epoch, data, str_code, train_log, valid_log, total_loss, weight_loss=False, weight_attention=False):

        data = {key: value.to(self.device) for key, value in data.items()}

        weights = torch.ones_like(data["bert_input"]).to(self.device)  # initialize weight as 1


        if epoch > self.warm_up_epochs:
            indices = data["index"].tolist()  # seq indices
            # (batchsize, seq_len)->(batchsize, seq_len, 1)
            weights = self.tracker.get_weights_for_Seq(indices, data["bert_input"].size(1), str_code)
            if weight_attention:
                unsqueezed_weights = weights.unsqueeze(2)
                attention_weights = torch.matmul(unsqueezed_weights, unsqueezed_weights.transpose(-2, -1))
                result = self.model.forward(data["bert_input"], data["time_input"], attention_weights)
            else:
                result = self.model.forward(data["bert_input"], data["time_input"])
        else:
            result = self.model.forward(data["bert_input"], data["time_input"])

        # return the prediction of the masked log key and the time interval
        mask_lm_output = result["logkey_output"]
        softmax = nn.Softmax(dim=2)
        probabilities = softmax(mask_lm_output)

        # store the log of this batch
        if str_code=="train":
            train_log = pd.concat([train_log, self.process_batch_log(probabilities, data, mask_lm_output)])
        else:
            valid_log = pd.concat([valid_log, self.process_batch_log(probabilities, data, mask_lm_output)])

        # 2-2. NLLLoss of predicting masked token word ignore_index = 0 to ignore unmasked tokens
        # since the last layer is a logsoftmax, here use NLlloss, if its a soft max layer, use CrossEntropy loss instead
        mask_loss = torch.tensor(0) if not self.is_logkey else self.criterion(mask_lm_output.transpose(1, 2),
                                                                              data["bert_label"])
        if weight_loss:
            mask_loss = mask_loss*weights
        # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
        loss = mask_loss.sum()
        total_loss += loss.item()

        # 3. backward and optimization only in train
        if str_code=="train":
            self.optim_schedule.zero_grad()  # reset the gradient
            loss.backward()  # back propagation
            self.optim_schedule.step_and_update_lr()

        return total_loss, train_log, valid_log



    def save_log(self, save_dir, surfix_log):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(save_dir + key + f"_{surfix_log}.csv",
                                            index=False)
            print("Log saved")
        except Exception as e:
            print(str(e))
            print("Failed to save logs")

    def save(self, save_dir="output/bert_trained.pth"):
        """
        Saving the current BERT model on file_path

        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        torch.save(self.model, save_dir)
        # self.bert.to(self.device)
        print(" Model Saved on:", save_dir)
        return save_dir

    @staticmethod
    def get_radius(dist: list, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist), 1 - nu)


    def process_batch_log(self, probabilities, data, mask_lm_output):

        batch_tracking = pd.DataFrame()
        # monitor the prediction uncertainties
        # [batchsize, seq_len]
        predictions = probabilities.argmax(dim=2, keepdim=True)
        prediction_prob_of_true_token = torch.gather(probabilities, 2, data["bert_ori_input"].unsqueeze(2))

        labels = data["label"].reshape(-1, 1).repeat(1, data["bert_input"].size(1))

        # monitor all token loss
        all_token_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_ori_input"])

        batch_tracking["predictions"] = np.array(predictions.reshape(-1).tolist())
        batch_tracking["prob_of_correct_pred"] = np.array(prediction_prob_of_true_token.reshape(-1).tolist())
        batch_tracking["token_label"] = np.array(data["token_label"].reshape(-1).tolist())
        batch_tracking["seq_label"] = np.array(labels.reshape(-1).tolist())
        batch_tracking["token_loss"] = np.array(all_token_loss.reshape(-1).tolist())

        return batch_tracking