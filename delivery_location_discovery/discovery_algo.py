import math
from data_preprocessing.location_candidate_generation import get_location_candidates
from tptk.common.spatial_func import distance, SPoint
from delivery_location_discovery.feature_extraction import match_to_cluster, get_location_commonality, \
    extract_matching_features
from delivery_location_discovery.dataset import get_scaler
from delivery_location_discovery.models import LocMatcher, LocMatcherPN, MLP
from data_preprocessing.ground_truth_construction import cluster_stay_points_DBSCAN, get_centroid
import pickle
import numpy as np
from sklearn import preprocessing
import torch
import os
import torch.nn.functional as F


class DiscoveryAlgo:
    def __init__(self, locs_path):
        self.indexed_locs, self.id2locs = get_location_candidates(locs_path)

    def discover(self, trip_sps, geocoding_loc, addr=None):
        pass


class DLInf(DiscoveryAlgo):
    def __init__(self, locs_path, hidden_dim, model_name, model_path, train_data_path, trip_inverted_data, geocoding2poi_type, lc_type='geocoding', use_addr=True):
        super(DLInf, self).__init__(locs_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_addr = use_addr
        self.feature_infos = [
            ('trip_cov', (0, 1)),
            ('loc_comm', (1, 2)),
            ('distance', (2, 3)),
            ('avg_duration', (3, 4)),
            ('nb_couriers', (4, 5)),
            ('time_dist', (5, 20))
        ]
        self.feature_info = dict(self.feature_infos)
        if model_name.endswith('-nP'):
            self.selected_features = ['trip_cov', 'loc_comm', 'distance']
        elif model_name.endswith('-nD'):
            self.selected_features = ['trip_cov', 'loc_comm', 'avg_duration', 'nb_couriers', 'time_dist']
        elif model_name.endswith('-nLC'):
            self.selected_features = ['trip_cov', 'distance', 'avg_duration', 'nb_couriers', 'time_dist']
        elif model_name.endswith('-nTC'):
            self.selected_features = ['loc_comm', 'distance', 'avg_duration', 'nb_couriers', 'time_dist']
        else:
            self.selected_features = ['trip_cov', 'loc_comm', 'distance', 'avg_duration', 'nb_couriers', 'time_dist']
        self.addr_scaler, self.loc_scaler = get_scaler(train_data_path)
        loc_dense_dim = sum([self.feature_info[f][1] - self.feature_info[f][0] for f in self.selected_features if
                             f not in ['time_dist']])
        time_dist_dim = (self.feature_info['time_dist'][1] - self.feature_info['time_dist'][
            0]) if 'time_dist' in self.selected_features else 0
        if model_name == 'LocMatcherPN':
            self.model = LocMatcherPN(hidden_dim, loc_inp_dim=(loc_dense_dim, time_dist_dim)).to(self.device)
        elif model_name.startswith('LocMatcher'):
            self.model = LocMatcher(hidden_dim[0], nb_heads=hidden_dim[1], nb_layers=hidden_dim[2],
                                    loc_inp_dim=(loc_dense_dim, time_dist_dim), use_addr=use_addr).to(self.device)
        else:
            raise Exception('unknown method')
        self.model.load_state_dict(torch.load(model_path + '/final_model.pt', map_location=self.device))
        self.model.eval()
        self.trip_inverted_data = trip_inverted_data
        self.geocoding2poi_type = geocoding2poi_type
        self.lc_type = lc_type

    def discover(self, trip_sps, geocoding_loc, courier=None, addr=None):
        trip_matched = match_to_cluster(trip_sps, self.indexed_locs)
        addr_features = torch.tensor(self.addr_scaler.transform(np.asarray([len(trip_matched)]).reshape(1, -1))).type(torch.float).to(self.device)
        addr_type = torch.tensor([self.geocoding2poi_type[(geocoding_loc.lat, geocoding_loc.lng)]]).unsqueeze(dim=0).type(torch.long).to(self.device)
        if self.lc_type == 'geocoding':
            instance_key = (geocoding_loc.lat, geocoding_loc.lng)
        elif self.lc_type == 'addr':
            uid, _, _ = addr.split('_')
            instance_key = (int(uid), geocoding_loc.lat, geocoding_loc.lng)
        else:
            raise Exception('unknown lc_type')
        loc2features = extract_matching_features(trip_matched, instance_key,
                                                 self.id2locs, self.trip_inverted_data)
        loc_all_seq = []
        loc_features = list(loc2features.items())
        for loc_id, matching_features in loc_features:
            loc, avg_duration, _, time_dist, nb_couriers = self.id2locs[loc_id]
            location_features = np.concatenate([np.asarray([avg_duration]), np.asarray([nb_couriers]), np.asarray(time_dist)], axis=0)
            loc_all_seq.append(np.concatenate([matching_features, location_features], axis=0))
        loc_all_seq = self.loc_scaler.transform(np.asarray(loc_all_seq))
        loc_dense_selected_X = []
        for f in self.selected_features:
            if f in ['time_dist']:
                continue
            start_idx, end_idx = self.feature_info[f]
            loc_dense_selected_X.append(loc_all_seq[:, start_idx:end_idx])
        loc_dense_selected_X = np.hstack(loc_dense_selected_X)
        time_dist_X = loc_all_seq[:, self.feature_info['time_dist'][0]:self.feature_info['time_dist'][1]]
        loc_dense_seq = torch.tensor(loc_dense_selected_X).type(torch.float).unsqueeze(dim=0).to(self.device)
        time_dist_seq = torch.tensor(time_dist_X).type(torch.float).unsqueeze(dim=0).to(self.device)
        pred = self.model(addr_features, addr_type, loc_dense_seq, time_dist_seq, [len(loc2features)])
        pred_idx = int(torch.argmax(pred[0]))
        selected_loc_id = loc_features[pred_idx][0]
        return self.id2locs[selected_loc_id][0]


class DLInfMLP(DiscoveryAlgo):
    def __init__(self, locs_path, hidden_dim, model_path, train_data_path, trip_inverted_data, geocoding2poi_type):
        super(DLInfMLP, self).__init__(locs_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(np.load(os.path.join(train_data_path, 'X.npy'))[:, 1:])
        self.model = MLP(hidden_dim, (6, 15)).to(self.device)
        self.model.load_state_dict(torch.load(model_path + '/final_model.pt', map_location=self.device))
        self.model.eval()
        self.trip_inverted_data = trip_inverted_data
        self.geocoding2poi_type = geocoding2poi_type

    def discover(self, trip_sps, geocoding_loc, addr=None):
        trip_matched = match_to_cluster(trip_sps, self.indexed_locs)
        loc2features = extract_matching_features(trip_matched, (geocoding_loc.lat, geocoding_loc.lng),
                                                 self.id2locs, self.trip_inverted_data)
        addr_type = int(self.geocoding2poi_type[(geocoding_loc.lat, geocoding_loc.lng)])
        addr_features = np.asarray([len(trip_matched)])
        cate_X = []
        quant_X = []
        Y = []
        for loc in loc2features:
            _, avg_duration, _, time_dist, nb_couriers = self.id2locs[loc]
            location_features = np.concatenate(
                [np.asarray([avg_duration]), np.asarray([nb_couriers]), np.asarray(time_dist)], axis=0)
            matching_features = loc2features[loc]
            cate_X.append(addr_type)
            quant_X.append(np.concatenate([addr_features, matching_features, location_features], axis=0))
            Y.append(loc)
        cate_X = torch.tensor(np.vstack(cate_X)).type(torch.long).to(self.device)
        quant_X = self.scaler.transform(np.vstack(quant_X))
        quant_X = torch.tensor(quant_X).type(torch.float).to(self.device)
        pred = F.softmax(self.model(cate_X, quant_X[:, :6], quant_X[:, 6:]), dim=1)[:, 1].cpu().detach().numpy()
        pred_idx = int(np.argmax(pred))
        selected_loc_id = Y[pred_idx]
        return self.id2locs[selected_loc_id][0]


class DLInfTradi(DiscoveryAlgo):
    def __init__(self, locs_path, model_name, model_path, train_data_path, trip_inverted_data, geocoding2poi_type):
        super(DLInfTradi, self).__init__(locs_path)
        with open('{}/final_model.pt'.format(model_path), 'rb') as f:
            self.clf = pickle.load(f)
        self.trip_inverted_data = trip_inverted_data
        self.model_name = model_name
        self.geocoding2poi_type = geocoding2poi_type
        if self.model_name not in ['RF', 'GBDT']:
            self.scaler = preprocessing.StandardScaler()
            train_quant_X = np.load(os.path.join(train_data_path, 'X.npy'))[:, 1:]
            self.scaler.fit(train_quant_X)

    def discover(self, trip_sps, geocoding_loc, courier=None, addr=None):
        trip_matched = match_to_cluster(trip_sps, self.indexed_locs)
        loc2features = extract_matching_features(trip_matched, (geocoding_loc.lat, geocoding_loc.lng),
                                                 self.id2locs, self.trip_inverted_data)
        addr_type = int(self.geocoding2poi_type[(geocoding_loc.lat, geocoding_loc.lng)])
        cate_X = np.zeros(21)
        cate_X[addr_type] = 1
        addr_features = np.asarray([len(trip_matched)])
        X = []
        Y = []
        for loc in loc2features:
            _, avg_duration, _, time_dist, nb_couriers = self.id2locs[loc]
            location_features = np.concatenate(
                [np.asarray([avg_duration]), np.asarray([nb_couriers]), np.asarray(time_dist)], axis=0)
            matching_features = loc2features[loc]
            quant_X = np.concatenate([addr_features, matching_features, location_features], axis=0)
            if self.model_name not in ['RF', 'GBDT']:
                quant_X = self.scaler.transform(quant_X.reshape(1, -1)).reshape(-1)
            X.append(np.concatenate([cate_X, quant_X], axis=0))
            Y.append(loc)
        features = np.vstack(X)
        if hasattr(self.clf, 'decision_function'):
            pred = self.clf.decision_function(features)
        else:
            pred = self.clf.predict_proba(features)[:, 1]
        pred_idx = int(np.argmax(pred))
        selected_loc_id = Y[pred_idx]
        return self.id2locs[selected_loc_id][0]


class GeocodingDiscovery(DiscoveryAlgo):
    def __init__(self, locs_path):
        super(GeocodingDiscovery, self).__init__(locs_path)

    def discover(self, trip_sps, geocoding_loc, courier=None):
        return geocoding_loc


class MinDistDiscovery(DiscoveryAlgo):
    def __init__(self, locs_path):
        super(MinDistDiscovery, self).__init__(locs_path)

    def discover(self, trip_sps, geocoding_loc, courier=None):
        matched_trip_sps = match_to_cluster(trip_sps, self.indexed_locs)
        cluster2trips = {}
        for trip_idx in range(len(matched_trip_sps)):
            for sp, cluster in matched_trip_sps[trip_idx]:
                if cluster not in cluster2trips:
                    cluster2trips[cluster] = set()
                cluster2trips[cluster].add(trip_idx)
        pt, dist = min([(self.id2locs[cluster][0], distance(self.id2locs[cluster][0], geocoding_loc)) for cluster in
                        cluster2trips], key=lambda x: x[1])
        return pt


class MaxTCDiscovery(DiscoveryAlgo):
    """
    find the location using the center of the max diversity cluster
    """

    def __init__(self, locs_path):
        super(MaxTCDiscovery, self).__init__(locs_path)

    def discover(self, trip_sps, geocoding_loc, addr=None):
        matched_trip_sps = match_to_cluster(trip_sps, self.indexed_locs)
        cluster2trips = {}
        for trip_idx in range(len(matched_trip_sps)):
            for sp, cluster in matched_trip_sps[trip_idx]:
                if cluster not in cluster2trips:
                    cluster2trips[cluster] = set()
                cluster2trips[cluster].add(trip_idx)
        max_cover_cluster = max([(cluster, len(cluster2trips[cluster])) for cluster in cluster2trips],
                                key=lambda x: x[1])[0]
        return self.id2locs[max_cover_cluster][0]


class MaxTCILCDiscovery(DiscoveryAlgo):
    def __init__(self, locs_path, inverted_data, instance_type, is_wrt_courier):
        super(MaxTCILCDiscovery, self).__init__(locs_path)
        self.trip_inverted_data = inverted_data
        self.instance_type = instance_type
        self.is_wrt_courier = is_wrt_courier

    def discover(self, trip_sps, geocoding_loc, addr=None):
        if self.instance_type == 'addr':
            uid, _, _ = addr.split('_')
        trip_inverted_data = self.trip_inverted_data
        matched_trip_sps = match_to_cluster(trip_sps, self.indexed_locs)
        cluster2trips = {}
        nb_trips = len(trip_sps)
        for trip_idx in range(len(matched_trip_sps)):
            for sp, cluster in matched_trip_sps[trip_idx]:
                if cluster not in cluster2trips:
                    cluster2trips[cluster] = set()
                cluster2trips[cluster].add(trip_idx)
        cluster_with_score = []
        for cluster in cluster2trips:
            cov_rate = len(cluster2trips[cluster]) / nb_trips
            if self.instance_type == 'geocoding':
                loc_com = get_location_commonality((geocoding_loc.lat, geocoding_loc.lng), cluster, trip_inverted_data)
            elif self.instance_type == 'addr':
                loc_com = get_location_commonality((int(uid), geocoding_loc.lat, geocoding_loc.lng), cluster,
                                                   trip_inverted_data)
            else:
                raise NotImplemented('{} type is not implemented'.format(self.instance_type))
            if loc_com == 0:
                loc_com = 0.01
            score = cov_rate * math.log2(1 / loc_com)
            cluster_with_score.append((cluster, score))
        max_score_cluster = max(cluster_with_score, key=lambda x: x[1])[0]
        return self.id2locs[max_score_cluster][0]


class AnnotationDiscovery(DiscoveryAlgo):
    def __init__(self, locs_path):
        super(AnnotationDiscovery, self).__init__(locs_path)

    def discover(self, trip_sps, geocoding_loc, courier=None, addr=None):
        sps = [trip.pt_list[-1] for trip in trip_sps]
        mean_lat = sum([sp.lat for sp in sps]) / len(sps)
        mean_lng = sum([sp.lng for sp in sps]) / len(sps)
        return SPoint(mean_lat, mean_lng)


class GeoCloudDiscovery(DiscoveryAlgo):
    def __init__(self, locs_path, eps_dist):
        super(GeoCloudDiscovery, self).__init__(locs_path)
        self.eps_dist = eps_dist

    def discover(self, trip_sps, geocoding_loc, courier=None, addr=None):
        sps = [trip.pt_list[-1] for trip in trip_sps]
        data = []
        for sp in sps:
            data.append((sp.lng, sp.lat))
        if len(data) == 1:
            return SPoint(data[0][1], data[0][0])
        else:
            cluster_labels = cluster_stay_points_DBSCAN(data, self.eps_dist)
            cluster2pts = {}
            for i in range(len(cluster_labels)):
                label = cluster_labels[i]
                if label not in cluster2pts:
                    cluster2pts[label] = []
                cluster2pts[label].append(data[i])
            largest_cluster_pts = max([cluster2pts[cluster] for cluster in cluster2pts], key=lambda x: len(x))
            centroid = get_centroid(largest_cluster_pts)
            return SPoint(centroid[1], centroid[0])