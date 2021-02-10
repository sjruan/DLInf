import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import os
import pickle
import numpy as np


class PadCollate:
    def __init__(self):
        pass

    def pad_collate(self, data):
        # data: list of (X_addr, X_addr_type, X_loc_dense_seq, X_time_dist_seq, Y)
        data.sort(key=lambda x: len(x[2]), reverse=True)
        data_length = [len(x[2]) for x in data]
        X_addr, X_addr_type, X_loc_dense_seq, X_time_dist_seq, Y = zip(*data)
        X_addr = torch.stack(list(X_addr), dim=0)
        X_addr_type = torch.stack(list(X_addr_type), dim=0)
        X_loc_dense_seq = pad_sequence(list(X_loc_dense_seq), batch_first=True, padding_value=0)
        X_time_dist_seq = pad_sequence(list(X_time_dist_seq), batch_first=True, padding_value=0)
        Y = torch.stack(list(Y), dim=0)
        return X_addr, X_addr_type, X_loc_dense_seq, X_time_dist_seq, data_length, Y

    def __call__(self, batch):
        return self.pad_collate(batch)


class LocMatcherDataset(Dataset):
    def __init__(self, X_addr, X_addr_type, X_loc_dense_seq, X_time_dist_seq, Y):
        super(LocMatcherDataset, self).__init__()
        self.X_addr = X_addr
        self.X_addr_type = X_addr_type
        self.X_loc_dense_seq = X_loc_dense_seq
        self.X_time_dist_seq = X_time_dist_seq
        self.Y = Y

    def __getitem__(self, index):
        return self.X_addr[index], self.X_addr_type[index], self.X_loc_dense_seq[index], self.X_time_dist_seq[index], \
               self.Y[index]

    def __len__(self):
        return len(self.X_addr)


class ClassDataset(Dataset):
    def __init__(self, X_addr_type, X_dense, X_time_dist, Y):
        super(ClassDataset, self).__init__()
        self.X_addr_type = X_addr_type
        self.X_dense = X_dense
        self.X_time_dist = X_time_dist
        self.Y = Y

    def __getitem__(self, index):
        return self.X_addr_type[index], self.X_dense[index], self.X_time_dist[index], self.Y[index]

    def __len__(self):
        return len(self.X_addr_type)


def get_class_scaler(data_path):
    scaler = preprocessing.StandardScaler()
    quant_X = np.load(os.path.join(data_path, 'X.npy'))[:, 1:]
    scaler.fit(quant_X)
    return scaler


def get_scaler(data_path):
    addr_scaler = preprocessing.StandardScaler()
    loc_scaler = preprocessing.StandardScaler()
    with open(os.path.join(data_path, 'X.pkl'), 'rb') as f:
        X = pickle.load(f)
    addr_data = np.vstack([np.concatenate([d['addr']]) for d in X])
    addr_scaler.fit(addr_data)
    loc_data = np.vstack([loc for d in X for loc in d['locs']])
    loc_scaler.fit(loc_data)
    return addr_scaler, loc_scaler


def get_class_data_loader(data_path, batch_size, mode='train'):
    feature_infos = [
        ('trip_cov', (0, 1)),
        ('loc_comm', (1, 2)),
        ('distance', (2, 3)),
        ('avg_duration', (3, 4)),
        ('nb_couriers', (4, 5)),
        ('time_dist', (5, 20))
    ]
    feature_info = dict(feature_infos)
    selected_features = ['trip_cov', 'loc_comm', 'distance', 'avg_duration', 'nb_couriers', 'time_dist']
    scaler = get_class_scaler(os.path.join(data_path, 'train'))
    data_path = os.path.join(data_path, mode)

    device = torch.device("cpu")
    X = np.load(os.path.join(data_path, 'X.npy'))
    X_addr_type = torch.tensor(X[:, 0:1]).type(torch.long).to(device)
    X_quant = torch.tensor(scaler.transform(X[:, 1:])).type(torch.float).to(device)
    # also with addr feature
    X_dense = X_quant[:, :6]
    X_time_dist = X_quant[:, 6:]
    Y = np.load(os.path.join(data_path, 'Y.npy'))
    Y = torch.tensor(Y).type(torch.long).to(device)
    dataset = ClassDataset(X_addr_type, X_dense, X_time_dist, Y)
    weights = [80 if y == 1 else 20 for y in Y]
    sampler = WeightedRandomSampler(weights, len(dataset))
    if mode == 'train':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dense_dim = sum([feature_info[f][1] - feature_info[f][0] for f in selected_features if f not in ['time_dist']]) + 1
    time_dist_dim = (feature_info['time_dist'][1] - feature_info['time_dist'][0])
    return data_loader, (dense_dim, time_dist_dim)


def get_data_loader(data_path, batch_size, model_name, mode='train'):
    feature_infos = [
        ('trip_cov', (0, 1)),
        ('loc_comm', (1, 2)),
        ('distance', (2, 3)),
        ('avg_duration', (3, 4)),
        ('nb_couriers', (4, 5)),
        ('time_dist', (5, 20))
    ]
    feature_info = dict(feature_infos)
    if model_name.endswith('-nP'):
        selected_features = ['trip_cov', 'loc_comm', 'distance']
    elif model_name.endswith('-nT'):
        selected_features = ['trip_cov', 'loc_comm', 'distance', 'avg_duration', 'nb_couriers']
    elif model_name.endswith('-nD'):
        selected_features = ['trip_cov', 'loc_comm', 'avg_duration', 'nb_couriers', 'time_dist']
    elif model_name.endswith('-nLC'):
        selected_features = ['trip_cov', 'distance', 'avg_duration', 'nb_couriers', 'time_dist']
    elif model_name.endswith('-nTC'):
        selected_features = ['loc_comm', 'distance', 'avg_duration', 'nb_couriers', 'time_dist']
    else:
        selected_features = ['trip_cov', 'loc_comm', 'distance', 'avg_duration', 'nb_couriers', 'time_dist']
    addr_scaler, loc_scaler = get_scaler(os.path.join(data_path, 'train'))
    data_path = os.path.join(data_path, mode)

    device = torch.device("cpu")
    with open(os.path.join(data_path, 'X.pkl'), 'rb') as f:
        X = pickle.load(f)
    X_addr = torch.tensor(addr_scaler.transform(np.vstack([d['addr'] for d in X]))).type(torch.float).to(device)
    X_addr_type = torch.tensor(np.vstack([d['addr_type'] for d in X])).type(torch.long).to(device)
    X_loc_normed = [loc_scaler.transform(np.vstack([loc for loc in d['locs']])) for d in X]
    X_loc_dense_normed_selected = []
    for d in X_loc_normed:
        selected_d = []
        for f in selected_features:
            # we only use dense features to construct X_loc_dense_seq
            if f in ['time_dist']:
                continue
            start_idx, end_idx = feature_info[f]
            selected_d.append(d[:, start_idx:end_idx])
        selected_d = np.hstack(selected_d)
        X_loc_dense_normed_selected.append(selected_d)
    X_time_dist_normed = []
    for d in X_loc_normed:
        X_time_dist_normed.append(d[:, feature_info['time_dist'][0]:feature_info['time_dist'][1]])
    with open(os.path.join(data_path, 'Y.pkl'), 'rb') as f:
        Y = pickle.load(f)
    Y = torch.tensor(Y).type(torch.long).to(device)
    X_loc_dense_seq = [torch.tensor(d).type(torch.float).to(device) for d in X_loc_dense_normed_selected]
    X_time_dist_seq = [torch.tensor(d).type(torch.float).to(device) for d in X_time_dist_normed]
    dataset = LocMatcherDataset(X_addr, X_addr_type, X_loc_dense_seq, X_time_dist_seq, Y)
    if mode == 'train':
        is_shuffle = True
    else:
        is_shuffle = False
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, collate_fn=PadCollate())
    loc_dense_dim = sum([feature_info[f][1] - feature_info[f][0] for f in selected_features if f not in ['time_dist']])
    time_dist_dim = (
                feature_info['time_dist'][1] - feature_info['time_dist'][0]) if 'time_dist' in selected_features else 0
    return data_loader, (loc_dense_dim, time_dist_dim)
