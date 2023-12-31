import json
import os
import ast
import csv
import io
from io import StringIO, BytesIO, TextIOWrapper
import gzip
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import ast
from datetime import timedelta
from tqdm import tqdm
import warnings
import sys
import time
import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
from pathlib import Path
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LogisticRegresser(torch.nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(LogisticRegresser, self).__init__()
        self.linear1 = torch.nn.Linear(feature_size, hidden_size)
        self.linear2 = torch.nn.Linear(feature_size, hidden_size)
    def forward(self, x):
        outputs = torch.sigmoid(self.linear2(self.linear1(x)))
        return outputs

class LSTMClassifier(nn.Module):
    def __init__(self, feature_size, n_state, hidden_size, rnn="GRU", regres=True, bidirectional=False, return_all=False,
                 seed=random.seed('2021'), is_classifier=False):
        
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        self.is_classifier = is_classifier
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(feature_size, self.hidden_size, bidirectional=bidirectional, batch_first=True).to(self.device)
        else:
            self.rnn = nn.LSTM(feature_size, self.hidden_size, bidirectional=bidirectional, batch_first=True).to(self.device)



        self.regressor = nn.Sequential(nn.BatchNorm1d(num_features=self.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(self.hidden_size, self.n_state))

    def forward(self, input, past_state=None, CLS=False, **kwargs):
        input = input.to(self.device)
        self.rnn.to(self.device)
        self.regressor.to(self.device)
        if not past_state:
            #  hidden states: (num_layers * num_directions, batch, hidden_size)
            past_state = torch.zeros([1, input.shape[0], self.hidden_size]).to(self.device)
        if self.rnn_type == 'GRU':
            all_encodings, encoding = self.rnn(input, past_state)
        else:
            all_encodings, (encoding, state) = self.rnn(input, (past_state, past_state))

        if CLS:
            out = self.regressor(encoding.view(encoding.shape[1], -1))
            return {'cls_emb': encoding.view(encoding.shape[1], -1), 'output': out, 'classification_output': out}

        if self.regres:
            if not self.return_all:
                out = self.regressor(encoding.view(encoding.shape[1], -1))
                if self.is_classifier:
                    return {'output': out, 'classification_output': out.unsqueeze(1)}
                else:
                    return out
            else:
                reshaped_encodings = all_encodings.view(all_encodings.shape[1]*all_encodings.shape[0],-1)
                return torch.t(self.regressor(reshaped_encodings).view(all_encodings.shape[0],-1))
        else:
            return encoding.view(encoding.shape[1], -1)


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        # self._decoder = nn.Sequential(
        #     nn.Linear(d_model, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(),
        #     nn.LayerNorm(2048),
        #     nn.Linear(2048, n_cls)
        # )
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
        # return x