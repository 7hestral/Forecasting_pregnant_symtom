import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
from typing import Dict, Iterable, Callable
from torch import nn

class Config:
    def __init__(self, seq_len, pred_len, feature_size, label_len=12,enc_in=7,dec_in=7, output_attention=True, individual=False, is_classifier=False):
        self.feature_size = feature_size
        self.is_classifier = is_classifier
        self.label_len = label_len
        self.output_attention = output_attention
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.embed_type = 0
        self.d_model = 8
        self.embed = 'timeF'
        self.freq = 'd'
        self.dropout = 0.05
        self.e_layers = 2
        self.d_layers = 1


        self.c_out = 7
        self.d_ff = 1024
        self.moving_avg = 1
        self.dropout = 0.05
        self.activation = 'gelu'
        self.factor = 3
        self.n_heads = 8


class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.if_test = True
        if 1 - train - valid < 0.05:
            print(1-train-valid)
            print("no test")
            self.if_test = False
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        if self.if_test:
            tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
        else:
            tmp = self.valid[1] * self.scale.expand(self.valid[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        if self.if_test:
            self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            # print(X.shape)
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class MinMaxScaler():
    def __init__(self,feature_size, sub_symptom_idx=-1):
        self.feature_size = feature_size
        self.sub_symptom_idx = sub_symptom_idx
        self.min_value_lst = [99999] * feature_size
        self.max_value_lst = [-99999] * feature_size
        self._reset_symptom_range()
    def _reset_symptom_range(self):
        self.min_value_lst[self.sub_symptom_idx] = 1
        self.max_value_lst[self.sub_symptom_idx] = 6
    def fit(self, original_df_lst):
        for target_feature in range(self.feature_size):
            for curr_user_mat in original_df_lst:
                curr_max = np.max(curr_user_mat[:, target_feature])
                curr_min = np.min(curr_user_mat[:, target_feature])
                if self.min_value_lst[target_feature] > curr_min:
                    self.min_value_lst[target_feature] = curr_min
                if self.max_value_lst[target_feature] < curr_max:
                    self.max_value_lst[target_feature] = curr_max
            if self.min_value_lst[target_feature] == self.max_value_lst[target_feature]:
                    self.max_value_lst[target_feature] += 1
        self._reset_symptom_range()
    def transform(self, original_df_lst):
        normalized_data_lst = []
        for curr_user_mat in original_df_lst:
            for target_feature in range(self.feature_size):
                curr_user_mat[:, target_feature] = (curr_user_mat[:, target_feature] - self.min_value_lst[target_feature]) / (self.max_value_lst[target_feature] - self.min_value_lst[target_feature])
            normalized_data_lst.append(curr_user_mat)
    
        if len(original_df_lst) == 1:
            normalized_data_lst = normalized_data_lst[0]
        return normalized_data_lst
    def fit_transform(self, original_df_lst):
        self.fit(original_df_lst)
        return self.transform(original_df_lst)
    def inverse_transform(self, normalized_df_lst):
        original_data_lst = []
        for curr_user_mat in normalized_df_lst:
            for target_feature in range(self.feature_size):
                curr_user_mat[:, target_feature] = curr_user_mat[:, target_feature] * (self.max_value_lst[target_feature] - self.min_value_lst[target_feature]) + self.min_value_lst[target_feature]
            original_data_lst.append(curr_user_mat)
        if len(normalized_df_lst) == 1:
            original_data_lst = original_data_lst[0]
        return original_data_lst

def min_max_normalization(original_df_lst, min_value_lst=None, max_value_lst=None, sub_symptom_idx=-1):
    # find min and max across population
    _, feature_size = original_df_lst[0].shape


    if min_value_lst is None:
        min_value_lst = [99999] * feature_size
        max_value_lst = [-99999] * feature_size
        for target_feature in range(feature_size):
            for curr_user_mat in original_df_lst:
                curr_max = np.max(curr_user_mat[:, target_feature])
                curr_min = np.min(curr_user_mat[:, target_feature])
                if min_value_lst[target_feature] > curr_min:
                    min_value_lst[target_feature] = curr_min
                if max_value_lst[target_feature] < curr_max:
                    max_value_lst[target_feature] = curr_max

    normalized_data_lst = []
    min_value_lst[sub_symptom_idx] = 1
    max_value_lst[sub_symptom_idx] = 6
    for curr_user_mat in original_df_lst:
        for target_feature in range(feature_size):
                
            if min_value_lst[target_feature] == max_value_lst[target_feature]:
                max_value_lst[target_feature] += 1
            curr_user_mat[:, target_feature] = (curr_user_mat[:, target_feature] - min_value_lst[target_feature]) / (max_value_lst[target_feature] - min_value_lst[target_feature])
        normalized_data_lst.append(curr_user_mat)

    if len(original_df_lst) == 1:
        normalized_data_lst = normalized_data_lst[0]
    return normalized_data_lst, min_value_lst, max_value_lst

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        # for param in self.params:
        #     grad_norm += math.pow(param.grad.data.norm(), 2)
        #
        # grad_norm = math.sqrt(grad_norm)
        # if grad_norm > 0:
        #     shrinkage = self.max_grad_norm / grad_norm
        # else:
        #     shrinkage = 1.
        #
        # for param in self.params:
        #     if shrinkage < 1:
        #         param.grad.data.mul_(shrinkage)
        self.optimizer.step()
        return  grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()
            