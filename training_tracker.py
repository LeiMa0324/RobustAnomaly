import pandas as pd
import numpy as np
from scipy import stats

class Tracker():
    def __init__(self, q):
        self.q = q

        self.counter = 0
        self.predictions = pd.DataFrame()
        self.uncertainties = pd.DataFrame(columns=["seq_label","token_label","uncertainty"])
        self.token_labels = []
        self.seq_labels = []


    def loadTracking(self, log_df, epoch):

        self.token_labels = log_df["token_label"]
        self.seq_labels = log_df["seq_label"]
        self.save_token_loss(log_df, "../output/bgl/dirty/", epoch)

        self.counter = (self.counter+1)%self.q  #reset counter
        #add new predictions
        self.predictions = pd.concat([self.predictions, pd.DataFrame(log_df["pred_uncertainty"])], axis=1)

        if self.counter==0:
            self.calculate_uncertainty(epoch)

    # but how to detect if there is a bad token in it?
    def calculate_uncertainty(self, epoch):
        self.uncertainties["token_label"] = self.token_labels
        self.uncertainties["seq_label"] = self.seq_labels
        uncertainties = []

        for idx, row in self.predictions.iterrows():
            prob = row.value_counts()/self.q  # the probability of each exsting predicted label
            en = stats.entropy(prob)
            uncertainties.append(en)

        self.uncertainties["uncertainty"] = uncertainties

        self.save_uncertainty("../output/bgl/dirty/", epoch)
        self.predictions.drop(self.predictions.index,inplace=True)
        self.uncertainties.drop(self.uncertainties.index, inplace=True)

    def save_uncertainty(self, save_dir, epoch):
        self.uncertainties.to_csv(save_dir + f"pred_uncertainties_{epoch}.csv",
                                            index=False)
        print(f"Prediction uncertainty of epoch {epoch} saved!")

    def save_token_loss(self, log_df, save_dir, epoch):
        loss_df = log_df[["seq_label", "token_label", "token_loss"]]
        loss_df.to_csv(save_dir +f"token_loss_{epoch}.csv", index=False)
        print(f"Token Loss of epoch {epoch} saved!")