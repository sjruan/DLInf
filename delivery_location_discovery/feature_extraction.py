from tptk.common.spatial_func import distance, SPoint, LAT_PER_METER, LNG_PER_METER
from tptk.common.mbr import MBR
import numpy as np
from tptk.common.trajectory import parse_traj_file, get_tid, STPoint, Trajectory
from tqdm import tqdm
import json
import os
import pickle
from data_preprocessing.location_candidate_generation import get_location_candidates
from data_preprocessing.ground_truth_construction import get_train_val_test_couriers, get_good_quality_labels
import multiprocessing
from datetime import datetime
from data_preprocessing.stay_point_extraction import construct_stay_points
from data_loader import load_clean_waybill, query_waybills_delv_in_trip
from tptk.query_utils import query_stay_points_by_temporal_range
from data_preprocessing.courier_behavior_evaluation import get_good_couriers


def generate_instance_loc_to_trips(station_dir, instance_type, target_couriers, indexed_locs, instance2trip_path, dist_thresh, min_stay_time):
    instance2tids = {}
    loc2tids = {}
    if instance_type == 'addr':
        with open(os.path.join(station_dir, 'user2id.pkl'), 'rb') as f:
            user2id = pickle.load(f)
    couriers_dir = os.path.join(station_dir, 'couriers')
    time_fmt = '%Y-%m-%d %H:%M:%S'
    for courier_id in target_couriers:
        courier_dir = os.path.join(couriers_dir, courier_id)
        with open(os.path.join(courier_dir, 'clean', 'trip_time_intervals.json'), 'r') as f:
            delivery_trip_intervals = json.load(f)
        clean_dir = os.path.join(couriers_dir, courier_id, 'clean')
        waybill_dir = os.path.join(courier_dir, 'raw', 'waybills')
        clean_traj_with_status_dir = os.path.join(clean_dir, 'trajs_with_status_{}_{}'.format(
            dist_thresh, min_stay_time))
        traj_filename = os.listdir(clean_traj_with_status_dir)[0]
        waybill_filename = os.listdir(waybill_dir)[0]
        for day_trip_intervals in tqdm(delivery_trip_intervals):
            tmp = datetime.strptime(day_trip_intervals[0][0], time_fmt)
            delv_day = datetime(tmp.year, tmp.month, tmp.day, 0, 0, 0)
            delv_day_str = delv_day.strftime('%Y%m%d')
            daily_traj = parse_traj_file(os.path.join(clean_traj_with_status_dir, delv_day_str + traj_filename[8:]),
                                         extra_fields=['stay'])[0]
            daily_sps = construct_stay_points(daily_traj)
            daily_delv_waybills = load_clean_waybill(os.path.join(waybill_dir, delv_day_str + waybill_filename[8:]))
            for first_recv_tm_str, start_time_lb_str, first_delv_tm_str, end_time_lb_str, end_time_ub_str in day_trip_intervals:
                # to make sure that each address will only be added once in each trip
                start_time_lb = datetime.strptime(start_time_lb_str, time_fmt)
                end_time_ub = datetime.strptime(end_time_ub_str, time_fmt)
                trip_waybills = query_waybills_delv_in_trip(daily_delv_waybills, start_time_lb, end_time_ub)
                trip_sp_list = []
                trip_sps = query_stay_points_by_temporal_range(daily_sps, start_time_lb, end_time_ub)
                for sp in trip_sps:
                    trip_sp_list.append(STPoint(sp.get_centroid().lat, sp.get_centroid().lng, sp.get_mid_time()))
                if len(trip_sp_list) == 0:
                    continue
                tid = get_tid(courier_id, trip_sp_list)
                # update instance to trips
                for waybill in trip_waybills:
                    if instance_type == 'addr':
                        instance = (user2id[waybill.member_id], waybill.lat, waybill.lng)
                    elif instance_type == 'geocoding':
                        instance = (waybill.lat, waybill.lng)
                    else:
                        raise NotImplemented('{} type is not implemented'.format(instance_type))
                    if instance not in instance2tids:
                        instance2tids[instance] = set()
                    instance2tids[instance].add(tid)
                # update location to trips
                trip_matched = match_to_cluster([Trajectory(courier_id, tid, trip_sp_list)], indexed_locs)
                for sp, cluster_id in trip_matched[0]:
                    if cluster_id not in loc2tids:
                        loc2tids[cluster_id] = set()
                    loc2tids[cluster_id].add(tid)
    with open(instance2trip_path, 'wb') as f:
        pickle.dump((instance2tids, loc2tids), f)


def get_instance_loc_to_trips(instance_loc_inverted_path):
    all_trips = set()
    with open(instance_loc_inverted_path, 'rb') as f:
        instance2trips, loc2trips = pickle.load(f)
    for instance in instance2trips:
        for trip in instance2trips[instance]:
            all_trips.add(trip)
    for loc in loc2trips:
        for trip in loc2trips[loc]:
            all_trips.add(trip)
    return all_trips, instance2trips, loc2trips


def match_to_cluster(trip_sps, indexed_locs):
    trip_matched = []
    for trip in trip_sps:
        cluster_id_seq = []
        for sp in trip.pt_list:
            cluster_id = list(indexed_locs.nearest((sp.lng, sp.lat)))[0]
            cluster_id_seq.append((sp, cluster_id))
        trip_matched.append(cluster_id_seq)
    return trip_matched


def get_location_commonality(instance, loc, inverted_data):
    all_trips, instance2trips, loc2trips = inverted_data
    no_instance_trips = all_trips - instance2trips[instance]
    loc_trips = loc2trips[loc]
    no_instance_loc_trips = no_instance_trips & loc_trips
    return len(no_instance_loc_trips) / len(no_instance_trips)


def extract_matching_features(trip_matched, instance, id2locs, trip_inverted_data):
    # instance uid,lat,lng / lat,lng
    if len(instance) == 2:
        lat_g, lng_g = instance[0], instance[1]
    elif len(instance) == 3:
        lat_g, lng_g = instance[1], instance[2]
    else:
        raise Exception('invalid instance')
    cluster2sps = {}
    nb_trips = len(trip_matched)
    for trip_idx in range(nb_trips):
        trip = trip_matched[trip_idx]
        for sp, cluster_id in trip:
            if cluster_id not in cluster2sps:
                cluster2sps[cluster_id] = {trip_idx: sp}
            elif trip_idx not in cluster2sps[cluster_id]:
                cluster2sps[cluster_id][trip_idx] = sp
            else:
                if float(cluster2sps[cluster_id][trip_idx].data['duration']) < float(sp.data['duration']):
                    cluster2sps[cluster_id][trip_idx] = sp
    cluster2features = {}
    # loc_freq,loc_common,dist
    for cluster_id in cluster2sps:
        trip2sp = cluster2sps[cluster_id]
        loc_freq = len(trip2sp) / nb_trips
        loc_common = get_location_commonality(instance, cluster_id, trip_inverted_data)
        dist = distance(SPoint(lat_g, lng_g), id2locs[cluster_id][0])
        cluster2features[cluster_id] = np.asarray([loc_freq, loc_common, dist])
    return cluster2features


def generate_learning_samples_classification(dataset_path, labels, indexed_locs, id2locs, trip_inverted_data, geocoding2poi_type):
    X = []
    Y = []
    for label in tqdm(labels):
        courier, addr, delv_loc, gt_cluster_id, nb_delvs = label
        filename = addr + '.csv'
        addr_arr = addr.split('_')
        geocoding = (float(addr_arr[1]), float(addr_arr[2]))
        try:
            trip_sps = parse_traj_file(os.path.join(dataset_path, courier, filename), extra_fields=['duration', 'area'])
        except:
            print('FileNotFound:{}'.format(os.path.join(dataset_path, courier, filename)))
            continue
        trip_matched = match_to_cluster(trip_sps, indexed_locs)
        loc2matching_features = extract_matching_features(trip_matched, geocoding, id2locs, trip_inverted_data)
        for loc in loc2matching_features:
            # trip_cov, loc_common, dist
            addr_features = np.asarray([len(trip_matched)])
            addr_type = np.asarray([geocoding2poi_type[geocoding]])
            matching_features = loc2matching_features[loc]
            _, avg_duration, _, time_dist, nb_couriers = id2locs[loc]
            location_features = np.concatenate([np.asarray([avg_duration]), np.asarray([nb_couriers]), np.asarray(time_dist)], axis=0)
            X.append(np.concatenate([addr_type, addr_features, matching_features, location_features], axis=0))
            if loc == gt_cluster_id:
                Y.append(1)
            else:
                Y.append(0)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


def generate_learning_samples_selection(dataset_path, labels, indexed_locs, id2locs, trip_inverted_data, geocoding2poi_type, instance='geocoding'):
    X = []
    Y = []
    for label in tqdm(labels):
        courier, addr, delv_loc, gt_cluster_id, nb_delvs = label
        filename = addr + '.csv'
        addr_arr = addr.split('_')
        if instance == 'geocoding':
            instance_key = (float(addr_arr[1]), float(addr_arr[2]))
        elif instance == 'addr':
            instance_key = (int(addr_arr[0]), float(addr_arr[1]), float(addr_arr[2]))
        else:
            raise Exception('unknown instance')
        geocoding = (float(addr_arr[1]), float(addr_arr[2]))
        addr_type = np.asarray([geocoding2poi_type[geocoding]])
        try:
            trip_sps = parse_traj_file(os.path.join(dataset_path, courier, filename), extra_fields=['duration', 'area'])
        except:
            print('FileNotFound:{}'.format(os.path.join(dataset_path, courier, filename)))
            continue
        trip_matched = match_to_cluster(trip_sps, indexed_locs)
        addr_features = np.asarray([len(trip_matched)])
        loc2matching_features = extract_matching_features(trip_matched, instance_key, id2locs, trip_inverted_data)
        candidate_locations = list(loc2matching_features.items())
        label_idx = None
        all_loc_features = []
        for i in range(len(candidate_locations)):
            loc_id, matching_features = candidate_locations[i]
            loc, avg_duration, _, time_dist, nb_couriers = id2locs[loc_id]
            location_features = np.concatenate([np.asarray([avg_duration]), np.asarray([nb_couriers]), np.asarray(time_dist)], axis=0)
            loc_features = [matching_features, location_features]
            all_loc_features.append(np.concatenate(loc_features, axis=0))
            if loc_id == gt_cluster_id:
                label_idx = i
        X.append({'addr': addr_features, 'addr_type': addr_type, 'locs': all_loc_features})
        Y.append(label_idx)
    return X, Y


def query_poi_type(geocoding_pt, poi_indexed, radius=30):
    query_mbr = MBR(geocoding_pt.lat - radius * LAT_PER_METER,
                    geocoding_pt.lng - radius * LNG_PER_METER,
                    geocoding_pt.lat + radius * LAT_PER_METER,
                    geocoding_pt.lng + radius * LNG_PER_METER)
    poi_results = list(poi_indexed.intersection((query_mbr.min_lng, query_mbr.min_lat, query_mbr.max_lng, query_mbr.max_lat), objects=True))
    refined_results = [(r.object[0], distance(geocoding_pt, SPoint(r.bounds[2], r.bounds[0]))) for r in poi_results if r.object[0] < 20]
    refined_results.sort(key=lambda x: x[1])
    if len(refined_results) == 0:
        return None
    else:
        return refined_results[0][0]


def generate_learning_samples_station(sta_id):
    station_dir = base_dir + '{}/'.format(sta_id)
    result_dir = station_dir + 'result_DLInf/'
    label_path = os.path.join(result_dir, 'delv_locs_{}_{}_S{}_R{}_Dgt{}_user.csv'.format(dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate, gt_cluster_dist_thresh))
    station_train_couriers, station_val_couriers, station_test_couriers = get_train_val_test_couriers(label_path, train_rate, val_rate, seed)
    candi_delv_locs_path = os.path.join(result_dir, 'candi_delivery_locations_D{}.csv'.format(clus_dist_thresh))
    indexed_locs, id2locs = get_location_candidates(candi_delv_locs_path)
    dataset_path = os.path.join(result_dir, 'candi_delv_sps_{}_{}_S{}_R{}_BD{}/'.format(dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate, batch_delivery_times))
    instance_loc_inverted_path = os.path.join(result_dir, '{}_loc_inverted_data_S0_R0_D{}.pkl'.format(inverted_instance_type, clus_dist_thresh))
    trip_inverted_data = get_instance_loc_to_trips(instance_loc_inverted_path)
    with open(os.path.join(result_dir, 'geocoding2poi_type_{}_{}_S{}_R{}.pkl'.format(dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate)), 'rb') as f:
        geocoding2poi_type = pickle.load(f)
    station_train_labels = get_good_quality_labels(label_path, station_train_couriers, indexed_locs, min_delvs, min_conf)
    sta_train_X, sta_train_Y = generate_learning_samples_classification(dataset_path, station_train_labels, indexed_locs, id2locs, trip_inverted_data, geocoding2poi_type)
    station_val_labels = get_good_quality_labels(label_path, station_val_couriers, indexed_locs, min_delvs, min_conf)
    sta_val_X, sta_val_Y = generate_learning_samples_classification(dataset_path, station_val_labels, indexed_locs, id2locs, trip_inverted_data, geocoding2poi_type)
    return sta_train_X, sta_train_Y, sta_val_X, sta_val_Y


def generate_learning_samples_selection_station(sta_id):
    station_dir = base_dir + '{}/'.format(sta_id)
    result_dir = station_dir + 'result_DLInf/'
    label_path = os.path.join(result_dir, 'delv_locs_{}_{}_S{}_R{}_Dgt{}_user.csv'.format(dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate, gt_cluster_dist_thresh))
    station_train_couriers, station_val_couriers, station_test_couriers = get_train_val_test_couriers(label_path, train_rate, val_rate, seed)
    candi_delv_locs_path = os.path.join(result_dir, 'candi_delivery_locations_D{}{}.csv'.format(clus_dist_thresh, clus_method_suffix))
    indexed_locs, id2locs = get_location_candidates(candi_delv_locs_path)
    dataset_path = os.path.join(result_dir, 'candi_delv_sps_{}_{}_S{}_R{}_BD{}/'.format(dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate, batch_delivery_times))
    instance_loc_inverted_path = os.path.join(result_dir, '{}_loc_inverted_data_S0_R0_D{}{}.pkl'.format(inverted_instance_type, clus_dist_thresh, clus_method_suffix))
    trip_inverted_data = get_instance_loc_to_trips(instance_loc_inverted_path)
    with open(os.path.join(result_dir, 'geocoding2poi_type_{}_{}_S{}_R{}.pkl'.format(dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate)), 'rb') as f:
        geocoding2poi_type = pickle.load(f)
    station_train_labels = get_good_quality_labels(label_path, station_train_couriers, indexed_locs, min_delvs, min_conf)
    sta_train_X, sta_train_Y = generate_learning_samples_selection(dataset_path, station_train_labels, indexed_locs, id2locs, trip_inverted_data, geocoding2poi_type, instance=inverted_instance_type)
    station_val_labels = get_good_quality_labels(label_path, station_val_couriers, indexed_locs, min_delvs, min_conf)
    sta_val_X, sta_val_Y = generate_learning_samples_selection(dataset_path, station_val_labels, indexed_locs, id2locs, trip_inverted_data, geocoding2poi_type, instance=inverted_instance_type)
    return sta_train_X, sta_train_Y, sta_val_X, sta_val_Y


def generate_inverted_data_station(sta_id):
    station_dir = base_dir + '{}/'.format(sta_id)
    result_dir = station_dir + 'result_DLInf/'
    courier_confirmation_behavior_path = station_dir + 'courier_behaviors_T{}.pkl'.format(t_thresh)
    target_couriers = get_good_couriers(courier_confirmation_behavior_path, 0, 0)
    location_candidates_path = result_dir + 'candi_delivery_locations_D{}{}.csv'.format(clus_dist_thresh, clus_method_suffix)
    indexed_locs, _ = get_location_candidates(location_candidates_path)
    instance2trip_path = result_dir + '{}_loc_inverted_data_S0_R0_D{}{}.pkl'.format(
                inverted_instance_type, clus_dist_thresh, clus_method_suffix)
    generate_instance_loc_to_trips(station_dir, inverted_instance_type, target_couriers, indexed_locs, instance2trip_path, dist_thresh, min_stay_time)


if __name__ == '__main__':
    station_ids = ['']
    base_dir = '../data/'
    with open(os.path.join(base_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    dist_thresh, min_stay_time = params['preprocessing']['dist_thresh'], params['preprocessing']['min_stay_time']
    behavior_min_trips = 10
    behavior_min_rate = 0.8
    train_rate = 0.8
    val_rate = 0.1
    seed = 2017
    batch_delivery_times = 5
    gt_cluster_dist_thresh = 30
    min_delvs = 2
    min_conf = 0.51
    t_thresh = 0.5
    # default: hc + geocoding
    # inverted_instance_type = 'addr'
    clus_method = 'hc'
    inverted_instance_type = 'geocoding'
    # clus_method = 'grid'
    clus_method_suffix = '_www10' if clus_method == 'grid' else ''
    lc_type_suffix = '_addr_inverted' if inverted_instance_type == 'addr' else ''

    # Generate Instance Loc to Trip
    with multiprocessing.Pool() as pool:
        res = pool.map(generate_inverted_data_station, station_ids)

    # Generate Learning Samples based on Selection
    for clus_dist_thresh in [30, 40, 50, 60, 70]:
        with multiprocessing.Pool() as pool:
            res = pool.map(generate_learning_samples_selection_station, station_ids)
        train_X = []
        train_Y = []
        val_X = []
        val_Y = []
        for sta_train_X, sta_train_Y, sta_val_X, sta_val_Y in res:
            if len(sta_train_Y) != 0:
                train_X.extend(sta_train_X)
                train_Y.extend(sta_train_Y)
            if len(sta_val_Y) != 0:
                val_X.extend(sta_val_X)
                val_Y.extend(sta_val_Y)
        print('# train X:{}'.format(len(train_X)))
        print('# train Y:{}'.format(len(train_Y)))
        print('# val X:{}'.format(len(val_X)))
        print('# val Y:{}'.format(len(val_Y)))
        selection_learning_sample_path = os.path.join(base_dir, 'result_DLInf',
                                                      'learning_samples_selection_S{}-R{}_BD{}_D{}_LQ{}-{}_seed{}{}{}/'.format(
                                                          behavior_min_trips, behavior_min_rate, batch_delivery_times,
                                                          clus_dist_thresh, min_delvs, min_conf, seed, clus_method_suffix,
                                                          lc_type_suffix))
        train_path = os.path.join(selection_learning_sample_path, 'train')
        os.makedirs(train_path, exist_ok=True)
        with open(os.path.join(train_path, 'X.pkl'), 'wb') as f:
            pickle.dump(train_X, f)
        with open(os.path.join(train_path, 'Y.pkl'), 'wb') as f:
            pickle.dump(train_Y, f)
        val_path = os.path.join(selection_learning_sample_path, 'val')
        os.makedirs(val_path, exist_ok=True)
        with open(os.path.join(val_path, 'X.pkl'), 'wb') as f:
            pickle.dump(val_X, f)
        with open(os.path.join(val_path, 'Y.pkl'), 'wb') as f:
            pickle.dump(val_Y, f)

    # Generate Learning Samples based on Classification
    clus_dist_thresh = 50
    with multiprocessing.Pool() as pool:
        res = pool.map(generate_learning_samples_station, station_ids)
    train_X = []
    train_Y = []
    val_X = []
    val_Y = []
    for sta_train_X, sta_train_Y, sta_val_X, sta_val_Y in res:
        if len(sta_train_Y) != 0:
            train_X.append(sta_train_X)
            train_Y.append(sta_train_Y)
        if len(sta_val_Y) != 0:
            val_X.append(sta_val_X)
            val_Y.append(sta_val_Y)
    train_X, train_Y, val_X, val_Y = np.vstack(train_X), np.concatenate(train_Y), np.vstack(val_X), np.concatenate(
        val_Y)
    print('train X shape:{}'.format(train_X.shape))
    print('train Y shape:{}'.format(train_Y.shape))
    print('val X shape:{}'.format(val_X.shape))
    print('val Y shape:{}'.format(val_Y.shape))
    classification_learning_sample_path = os.path.join(base_dir, 'result_DLInf',
                                                       'learning_samples_S{}-R{}_BD{}_D{}_LQ{}-{}_seed{}/'.format(
                                                           behavior_min_trips, behavior_min_rate, batch_delivery_times,
                                                           clus_dist_thresh, min_delvs, min_conf, seed))
    train_path = os.path.join(classification_learning_sample_path, 'train')
    os.makedirs(train_path, exist_ok=True)
    np.save(os.path.join(train_path, 'X.npy'), train_X)
    np.save(os.path.join(train_path, 'Y.npy'), train_Y)
    val_path = os.path.join(classification_learning_sample_path, 'val')
    os.makedirs(val_path, exist_ok=True)
    np.save(os.path.join(val_path, 'X.npy'), val_X)
    np.save(os.path.join(val_path, 'Y.npy'), val_Y)
