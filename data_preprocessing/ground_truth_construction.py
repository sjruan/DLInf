import pickle
import os
import json
from tqdm import tqdm
from datetime import datetime, timedelta
from tptk.common.trajectory import parse_traj_file, store_traj_file
from tptk.query_utils import query_stay_points_by_temporal_range
from tptk.common.spatial_func import distance, LAT_PER_METER, LNG_PER_METER, SPoint
from data_preprocessing.stay_point_extraction import construct_stay_points
from data_loader import query_waybills_delv_in_trip, load_clean_waybill
import numpy as np
import multiprocessing
from data_preprocessing.courier_behavior_evaluation import get_good_couriers
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


def cluster_stay_points(data, dist_thresh_in_meter):
    dist_thresh = ((LAT_PER_METER + LNG_PER_METER) / 2) * dist_thresh_in_meter
    clustering_algo = AgglomerativeClustering(n_clusters=None, linkage='average',
                                              distance_threshold=dist_thresh, affinity='euclidean')
    labels = clustering_algo.fit_predict(data)
    return labels


def cluster_stay_points_DBSCAN(data, eps_in_meter):
    eps = ((LAT_PER_METER + LNG_PER_METER) / 2) * eps_in_meter
    clustering_algo = DBSCAN(eps=eps, min_samples=1)
    labels = clustering_algo.fit_predict(data)
    return labels


def get_centroid(pts):
    mean_x = sum([pt[0] for pt in pts]) / len(pts)
    mean_y = sum([pt[1] for pt in pts]) / len(pts)
    return mean_x, mean_y


def find_delivery_caused_stay_points(station_dir, dist_thresh, min_stay_time, accept_time_delay,
                                     good_couriers, delv_caused_stay_points_path):
    couriers_dir = os.path.join(station_dir, 'couriers')
    with open(os.path.join(station_dir, 'user2id.pkl'), 'rb') as f:
        user2id = pickle.load(f)
    os.makedirs(delv_caused_stay_points_path, exist_ok=True)
    time_fmt = '%Y-%m-%d %H:%M:%S'
    # addr -> list of [candidate stay points in each trip]
    addr_historical_sps = {}
    nb_waybills = 0
    for courier_id in good_couriers:
        courier_dir = os.path.join(couriers_dir, courier_id)
        with open(os.path.join(courier_dir, 'clean', 'trip_time_intervals.json'), 'r') as f:
            delivery_trip_intervals = json.load(f)
        clean_dir = os.path.join(couriers_dir, courier_id, 'clean')
        waybill_dir = os.path.join(courier_dir, 'raw', 'waybills')
        clean_traj_with_status_dir = os.path.join(clean_dir, 'trajs_with_status_{}_{}'.format(
            dist_thresh, min_stay_time))
        traj_filename = os.listdir(clean_traj_with_status_dir)[0]
        waybill_filename = os.listdir(waybill_dir)[0]
        all_trips = [trip_interval for day_trip_intervals in delivery_trip_intervals
                     for trip_interval in day_trip_intervals]
        for first_recv_tm_str, start_time_lb_str, first_delv_tm_str, end_time_lb_str, end_time_ub_str in tqdm(
                all_trips):
            tmp = datetime.strptime(first_recv_tm_str, time_fmt)
            delv_day = datetime(tmp.year, tmp.month, tmp.day, 0, 0, 0)
            delv_day_str = delv_day.strftime('%Y%m%d')
            daily_traj = parse_traj_file(os.path.join(clean_traj_with_status_dir, delv_day_str + traj_filename[8:]),
                                         extra_fields=['stay'])[0]
            daily_sps = construct_stay_points(daily_traj)
            daily_delv_waybills = load_clean_waybill(os.path.join(waybill_dir, delv_day_str + waybill_filename[8:]))
            # to make sure that each address will only be added once in each trip
            trip_added_addrs = set()
            start_time_lb = datetime.strptime(start_time_lb_str, time_fmt)
            end_time_ub = datetime.strptime(end_time_ub_str, time_fmt)
            trip_waybills = query_waybills_delv_in_trip(daily_delv_waybills, start_time_lb, end_time_ub)
            trip_sps = query_stay_points_by_temporal_range(daily_sps, start_time_lb, end_time_ub)
            nb_waybills += len(trip_waybills)
            for waybill in trip_waybills:
                addr = '{}_{}_{}'.format(user2id[waybill.member_id], waybill.lat, waybill.lng)
                # since for each trip, a user might place multiple orders,
                # we should make sure that for each address, each trip corresponds to at most one stay point record
                if addr in trip_added_addrs:
                    continue
                accept_delv_tm_lb = waybill.manual_delv_tm + timedelta(seconds=-accept_time_delay)
                temporal_near_stay_points = query_stay_points_by_temporal_range(
                    trip_sps, accept_delv_tm_lb, waybill.manual_delv_tm, temporal_relation='intersect')
                if len(temporal_near_stay_points) > 0:
                    true_sp = max(temporal_near_stay_points, key=lambda sp: sp.get_duration())
                    if addr not in addr_historical_sps:
                        addr_historical_sps[addr] = []
                    addr_historical_sps[addr].append(true_sp)
                    trip_added_addrs.add(addr)
    print('unique locations:{}'.format(len(addr_historical_sps)))
    print('total waybills:{}'.format(nb_waybills))
    for addr in addr_historical_sps:
        store_traj_file(addr_historical_sps[addr], os.path.join(delv_caused_stay_points_path, '{}.csv'.format(addr)))


def infer_delivery_locations_wrt_couriers(delv_caused_stay_points_path, label_path, clus_dist_thresh):
    addr2locs = {}
    for filename in tqdm(os.listdir(delv_caused_stay_points_path)):
        hist_sps = parse_traj_file(os.path.join(delv_caused_stay_points_path, filename))
        oids = [sp.oid for sp in hist_sps]
        main_courier = max(oids, key=oids.count)
        data = []
        for sp in hist_sps:
            if sp.oid != main_courier:
                continue
            centroid = sp.get_centroid()
            data.append((centroid.lng, centroid.lat))
        if len(data) == 1:
            centroid = hist_sps[0].get_centroid()
            addr2locs[filename[:-4]] = (centroid.lat, centroid.lng, 1.0, 1, hist_sps[0].oid)
            continue
        cluster_labels = cluster_stay_points(data, clus_dist_thresh)
        cluster2pts = {}
        for i in range(len(cluster_labels)):
            label = cluster_labels[i]
            if label not in cluster2pts:
                cluster2pts[label] = []
            cluster2pts[label].append(data[i])
        largest_cluster_pts = max([cluster2pts[cluster] for cluster in cluster2pts], key=lambda x: len(x))
        centroid = get_centroid(largest_cluster_pts)
        # addr -> loc, covered_rate, nb_tot_trips
        addr2locs[filename[:-4]] = (
        centroid[1], centroid[0], len(largest_cluster_pts) / len(data), len(data), main_courier)
    with open(label_path, 'w') as f:
        f.write('addr,lat,lng,rate,nb_trips,main_courier\n')
        for addr in addr2locs:
            lat, lng, rate, nb_trips, main_courier = addr2locs[addr]
            f.write('{},{},{},{},{},{}\n'.format(addr, lat, lng, rate, nb_trips, main_courier))


def get_courier2labels(label_path):
    courier2labels = {}
    with open(label_path, 'r') as f:
        f.readline()
        for line in f.readlines():
            attrs = line.strip().split(',')
            courier = attrs[5]
            if courier not in courier2labels:
                courier2labels[courier] = {}
            courier2labels[courier][attrs[0]] = (
            SPoint(float(attrs[1]), float(attrs[2])), float(attrs[3]), int(attrs[4]))
    return courier2labels


def get_train_val_test_couriers(label_path, train_rate, val_rate, seed):
    courier2labels = get_courier2labels(label_path)
    couriers = list(courier2labels.keys())
    nb_couriers = len(couriers)
    idxes = np.random.RandomState(seed=seed).permutation(nb_couriers)
    train_size = int(nb_couriers * train_rate)
    val_size = int(nb_couriers * val_rate)
    train_couriers = [couriers[train_idx] for train_idx in idxes[:train_size]]
    val_couriers = [couriers[val_idx] for val_idx in idxes[train_size:(train_size + val_size)]]
    test_couriers = [couriers[test_idx] for test_idx in idxes[(train_size + val_size):]]
    return train_couriers, val_couriers, test_couriers


def is_label_good_quality(addr, delv_loc, rate, nb_delvs, min_delvs, min_conf):
    uid, lat, lng = addr.split('_')
    lat, lng = float(lat), float(lng)
    geocoding_pt = SPoint(lat, lng)
    dist = distance(delv_loc, geocoding_pt)
    if rate < min_conf:
        return False
    if nb_delvs < min_delvs:
        return False
    return True


def get_good_quality_labels(label_path, target_couriers, indexed_locs, min_delvs, min_conf):
    courier2labels = get_courier2labels(label_path)
    target_labels = []
    for courier in target_couriers:
        for addr in courier2labels[courier]:
            delv_loc, rate, nb_delvs = courier2labels[courier][addr]
            if is_label_good_quality(addr, delv_loc, rate, nb_delvs, min_delvs, min_conf):
                nearest_cluster_id = list(indexed_locs.nearest((delv_loc.lng, delv_loc.lat)))[0]
                target_labels.append((courier, addr, delv_loc, nearest_cluster_id, nb_delvs))
    return target_labels


def search(sta_id):
    station_dir = base_dir + '{}/'.format(sta_id)
    courier_confirmation_behavior_path = station_dir + 'courier_behaviors_T{}.pkl'.format(t_thresh)
    target_couriers = get_good_couriers(courier_confirmation_behavior_path, behavior_min_trips, behavior_min_rate)
    print('{},{}'.format(sta_id, len(target_couriers)))
    result_dir = station_dir + 'result_DLInf/'
    delv_caused_stay_points_path = os.path.join(result_dir, 'delv_caused_stay_points_{}_{}_S{}_R{}_user/'.format(
        dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate))
    find_delivery_caused_stay_points(station_dir, dist_thresh, min_stay_time, accept_time_delay, target_couriers, delv_caused_stay_points_path)


if __name__ == '__main__':
    station_ids = ['']
    base_dir = '../data/'
    t_thresh = 0.5
    behavior_min_trips = 10
    behavior_min_rate = 0.8
    gt_cluster_dist_thresh = 30
    with open(os.path.join(base_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    dist_thresh, min_stay_time = params['preprocessing']['dist_thresh'], params['preprocessing']['min_stay_time']
    accept_time_delay = params['preprocessing']['accept_time_delay']

    # Find Delivery Caused Stay Points for Each Inference Instance
    with multiprocessing.Pool() as pool:
        pool.map(search, station_ids)

    # Generate Ground Truths
    for sta_id in station_ids:
        print('Station:{}'.format(sta_id))
        station_dir = base_dir + '{}/'.format(sta_id)
        result_dir = station_dir + 'result_DLInf/'
        delv_caused_stay_points_path = os.path.join(result_dir, 'delv_caused_stay_points_{}_{}_S{}_R{}_user/'.format(
            dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate))
        label_path = os.path.join(result_dir,
                                  'delv_locs_{}_{}_S{}_R{}_Dgt{}_user.csv'.format(dist_thresh, min_stay_time,
                                                                                  behavior_min_trips, behavior_min_rate,
                                                                                  gt_cluster_dist_thresh))
        infer_delivery_locations_wrt_couriers(delv_caused_stay_points_path, label_path, gt_cluster_dist_thresh)
