import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
import torch
from matplotlib import pyplot as plt

class Tracker():
    def __init__(self, q, layer_num, head_num, weight_method, output_dir):
        self.entropy_window = q # the window size of computing entropy

        self.epoch = {"train": 0, "valid": 0}
        self.predictions = {"train": pd.DataFrame(), "valid": pd.DataFrame()}

        self.entropy = {"train":pd.DataFrame(columns=["seq_label", "token_label", "entropy"]),
                        "valid": pd.DataFrame(columns=["seq_label", "token_label", "entropy"])
                        }
        self.variance = {"train":pd.DataFrame(columns=["seq_label", "token_label", "variance"]),
                         "valid": pd.DataFrame(columns=["seq_label", "token_label", "variance"]),
                         }
        self.weight ={"train": np.array([]),
                      "valid": np.array([])
                      }
        self.weight_method = weight_method
        self.smoothness = 0.01

        self.token_labels = {"train":[],
                             "valid": []}

        self.token_loss = {"train":[],
                             "valid": []}

        self.token_pseudo_labels = {"train":[],
                             "valid": []}

        self.seq_labels = {"train":[],
                             "valid": []}


        self.prob_of_correct_pred = {"train":pd.DataFrame(),
                                     "valid": pd.DataFrame()
        }


        self.attentions = pd.DataFrame(columns=["seq_label", "token_label","layer","head","attending_att", "attended_att"])
        self.layer_num = layer_num
        self.head_num = head_num
        self.output_dir = output_dir
        self.normalize_weight_to_mean_one = False


    def load_tracking(self, log_df, epoch, code_str="train"):


        self.token_labels[code_str] = log_df["token_label"]
        self.seq_labels[code_str]  = log_df["seq_label"]
        self.token_loss[code_str] = log_df["token_loss"]

        # self.save_token_loss(log_df, "../output/bgl/dirty/tracking", epoch)
        # self.save_attentions(log_df, "../output/bgl/dirty/tracking", epoch)

        self.epoch[code_str] = epoch
        #add new predictions

        # if self.predictions[code_str].shape[1]<self.entropy_window:
        #     self.predictions[code_str][f"{epoch}"] = log_df["predictions"]
        # else:
        #     self.predictions[code_str] = self.predictions[code_str].drop(self.predictions[code_str].columns[0], axis=1)
        #     self.predictions[code_str][f"{epoch}"] = log_df["predictions"]
        self.predictions[code_str][f"{epoch}"] = log_df["predictions"]

        self.prob_of_correct_pred[code_str] = pd.concat((self.prob_of_correct_pred[code_str],
                                                         log_df["prob_of_correct_pred"].rename(f'{epoch}')), axis=1)


    def compute_weight(self, code_str="train"):

        if self.weight_method =="SGD-WD":
            self.weight_by_difficulty(code_str)
        elif self.weight_method=="high-entropy":
            self.weight_by_high_entropy(code_str)
        elif self.weight_method=="history-entropy":
            self.weight_by_history_entropy(code_str)

        print(f"if normalize weight to mean one?{self.normalize_weight_to_mean_one}")

        #plot the weights
        if code_str=="train":
            self.plot_weight_distribution(code_str)

    # calculate the prediction entropy of q epochs
    def calculate_prediction_entropy(self, code_str, if_history=False):
        # todo: enropy of certain window is discrete, consider longer windows

        self.entropy[code_str]["token_label"] = self.token_labels[code_str]
        self.entropy[code_str]["seq_label"] = self.seq_labels[code_str]
        entropies = []

        if self.predictions[code_str].shape[1]<self.entropy_window:
            raise Exception("Not enough epochs for entropy computation!")

        for idx, row in self.predictions[code_str].iterrows():
            if if_history:
                prob = row.value_counts() / len(row)  # the probability of each exsting predicted label
            else:
                prob = row.value_counts()/self.entropy_window  # the probability of each exsting predicted label
            en = stats.entropy(prob)
            entropies.append(en)

        # normalize entropy to 0,1
        self.entropy[code_str]["entropy"]  = preprocessing.normalize([np.array(entropies)])[0]

        # self.save_entropy("../output/bgl/dirty/tracking", epoch)
        # self.predictions.drop(self.predictions.index,inplace=True)
        # self.entropy.drop(self.entropy.index, inplace=True)

    # calculate the prediction probability variance of the correct label

    def calculate_prediction_variance(self, code_str="train"):
        self.entropy[code_str]["token_label"] = self.token_labels[code_str]
        self.entropy[code_str]["seq_label"] = self.seq_labels[code_str]
        variance = []

        for idx, row in self.prob_of_correct_pred[code_str].iterrows():
            var = np.var(row)
            variance.append(var)

        variance = np.array(variance)
        self.variance[code_str]["variance"] = variance


    def weight_by_high_entropy(self, code_str="train"):
        #own method, reweight the sample with higher entropy
        self.calculate_prediction_entropy(code_str)
        self.weight[code_str] = self.entropy[code_str]["entropy"] + self.smoothness
        if self.normalize_weight_to_mean_one:
            total_sum = np.sum(self.weight[code_str])
            #normalize the weight to mean 1
            self.weight[code_str] = self.weight[code_str] * float(len(self.weight[code_str])) / total_sum

    def weight_by_history_entropy(self, code_str="train"):
        #own method, reweight the sample with higher entropy
        self.calculate_prediction_entropy(code_str, if_history=True)
        self.weight[code_str] = self.entropy[code_str]["entropy"] + self.smoothness
        if self.normalize_weight_to_mean_one:
            total_sum = np.sum(self.weight[code_str])
            #normalize the weight to mean 1
            self.weight[code_str] = self.weight[code_str] * float(len(self.weight[code_str])) / total_sum


    def weight_by_difficulty(self, code_str="train"):
        # active bias:SGD Weighted by Difficulty, SGD-WD
        # vi = 1 − p ̄Ht−1 (yi|xi) + εD

        self.weight[code_str] = 1 - self.prob_of_correct_pred[code_str].mean(axis=1) + self.smoothness
        if self.normalize_weight_to_mean_one:
            total_sum = np.sum(self.weight[code_str])
            #normalize the weight to mean 1
            self.weight[code_str] = self.weight[code_str] * float(len(self.weight[code_str])) / total_sum


    def get_weights_for_Seq(self, seq_indices, seq_len, code_str):
        if seq_indices==None:
            return self.weight[code_str]

        seq_weight= []
        for idx in seq_indices:
            w = self.weight[code_str][idx * seq_len: idx * seq_len + seq_len].tolist()
            assert len(w)==seq_len
            seq_weight.append(w)
        # asarray convert list of list to np arrays
        cuda_condition = torch.cuda.is_available()
        weights = torch.from_numpy(np.array(seq_weight, dtype=np.float32)).to("cuda:0" if cuda_condition else "cpu")
        #ValueError: setting an array element with a sequence.
        # The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (32,) + inhomogeneous part.#
        return weights

    def save_entropy(self, save_dir, epoch):
        self.entropy[str].to_csv(save_dir + f"pred_entropy_{epoch}.csv",
                            index=False)
        print(f"Prediction entropy of epoch {epoch} saved!")

    def save_token_loss(self, log_df, save_dir, epoch):
        loss_df = log_df[["seq_label", "token_label", "token_loss"]]
        loss_df.to_csv(save_dir +f"token_loss_{epoch}.csv", index=False)
        print(f"Token Loss of epoch {epoch} saved!")

    def save_attentions(self, log_df, save_dir, epoch):
        attending_cols =[c for c in log_df.columns if "attending" in str(c)]
        attending_cols.extend(["seq_label", "token_label"])
        attending_df = log_df[attending_cols]
        # attended_cols = [c for c in log_df.columns if "attended" in str(c)]
        # attended_cols.extend(["seq_label", "token_label"])
        # attended_df = log_df[attended_cols]
        attending_df.to_csv(save_dir +f"attending_atten_{epoch}.csv", index=False)
        # attended_df.to_csv(save_dir +f"attended_atten_{epoch}.csv", index=False)

        print(f"Attentions of {epoch} saved!")

    def plot_weight_distribution(self, code_str):


        assert len(self.token_labels[code_str] )==len(self.weight[code_str])

        good_indices = np.array(self.token_labels[code_str])==0
        bad_indices = np.array(self.token_labels[code_str])==1

        plt.hist(self.weight[code_str][good_indices],bins=50, color='steelblue', alpha=0.4, label="good token")
        plt.hist(self.weight[code_str][bad_indices], bins=50, color='purple', alpha=0.4, label="bad token")

        plt.xlabel("Weight")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(self.output_dir+f"weight_hist_epoch_{self.epoch[code_str]}.png")
        plt.cla()