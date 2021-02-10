import os
import math
import multiprocessing
import pickle
import json
from datetime import datetime
from tptk.common.spatial_func import distance, SPoint
from tptk.common.trajectory import parse_traj_file
from delivery_location_discovery.discovery_algo import GeocodingDiscovery, MinDistDiscovery, MaxTCDiscovery, AnnotationDiscovery, GeoCloudDiscovery, MaxTCILCDiscovery, DLInfTradi, DLInfMLP, DLInf
from data_preprocessing.location_candidate_generation import get_location_candidates
from data_preprocessing.ground_truth_construction import get_train_val_test_couriers, get_good_quality_labels
from delivery_location_discovery.feature_extraction import get_instance_loc_to_trips


def evaluate_label(station_test_label):
    courier, addr, delv_loc, nearest_cluster_id, _ = station_test_label
    delv_loc_matched = id2locs[nearest_cluster_id][0]
    hist_trip_sps = parse_traj_file(os.path.join(dataset_path, courier, addr + '.csv'),
                                    extra_fields=['duration', 'area'])
    nb_delvs = len(hist_trip_sps)
    if algo_name == 'DLInfSpeedUp':
        discovered_loc = algo.discover(None, None, None, addr)
    else:
        arr = addr.split('_')
        geocoding_loc = SPoint(float(arr[1]), float(arr[2]))
        discovered_loc = algo.discover(hist_trip_sps, geocoding_loc)
    error = distance(delv_loc, discovered_loc)
    error_matched = distance(delv_loc_matched, discovered_loc)
    is_matched = False
    if error_matched < 1.0:
        is_matched = True
    return error, is_matched, nb_delvs


def get_algo(algo_name, candi_delv_locs_path, instance_type, trip_inverted_data=None, geocoding2poi_type=None):
    if algo_name == 'Geocoding':
        return GeocodingDiscovery(candi_delv_locs_path)
    elif algo_name == 'Annotation':
        return AnnotationDiscovery(candi_delv_locs_path)
    elif algo_name == 'GeoCloud':
        eps = 100
        return GeoCloudDiscovery(candi_delv_locs_path, eps)
    elif algo_name == 'MinDist':
        return MinDistDiscovery(candi_delv_locs_path)
    elif algo_name == 'MaxTC':
        return MaxTCDiscovery(candi_delv_locs_path)
    elif algo_name == 'MaxTCILC':
        return MaxTCILCDiscovery(candi_delv_locs_path, trip_inverted_data, instance_type, False)
    elif algo_name == 'DLInfMLP':
        model_name = 'MLP'
        model_id = ''
        hidden_dim = 16
        model_path = os.path.join(global_result_path, 'saved_model/loc_selector_S{}_R{}_BD{}_D{}_LQ{}-{}_seed{}/{}/H{}_{}/'.format(
                behavior_min_trips, behavior_min_rate, batch_delivery_times, clus_dist_thresh, min_delvs, min_conf, seed, model_name, hidden_dim, model_id))
        train_data_path = os.path.join(global_result_path, 'learning_samples_S{}-R{}_BD{}_D{}_LQ{}-{}_seed{}/train/'.format(
            behavior_min_trips, behavior_min_rate, batch_delivery_times, clus_dist_thresh, min_delvs, min_conf, seed))
        return DLInfMLP(candi_delv_locs_path, hidden_dim, model_path, train_data_path, trip_inverted_data, geocoding2poi_type)
    elif algo_name == 'DLInfPN':
        hidden_dim = 16
        model_name = 'LocMatcherPN'
        model_id = ''
        model_path = os.path.join(global_result_path,
                                  'saved_model/loc_selector_S{}_R{}_BD{}_D{}_LQ{}-{}_seed{}{}{}/{}/H{}_{}/'.format(
                                      behavior_min_trips, behavior_min_rate, batch_delivery_times, clus_dist_thresh,
                                      min_delvs, min_conf, seed, clus_method_suffix, lc_type_suffix, model_name,
                                      hidden_dim, model_id))
        train_data_path = os.path.join(global_result_path,
                                       'learning_samples_selection_S{}-R{}_BD{}_D{}_LQ{}-{}_seed{}/train/'.format(
                                           behavior_min_trips, behavior_min_rate, batch_delivery_times,
                                           clus_dist_thresh, min_delvs, min_conf, seed))
        return DLInf(candi_delv_locs_path, hidden_dim, model_name, model_path, train_data_path, trip_inverted_data,
                     geocoding2poi_type, use_addr=True)
    elif algo_name == 'DLInf':
        hidden_dim = (32, 2, 3)
        model_name = 'LocMatcher'
        if clus_method == 'hc' and inverted_instance_type == 'geocoding':
            if clus_dist_thresh == 50:
                model_id = ''
            elif clus_dist_thresh == 30:
                model_id = ''
            elif clus_dist_thresh == 40:
                model_id = ''
            elif clus_dist_thresh == 60:
                model_id = ''
            elif clus_dist_thresh == 70:
                model_id = ''
        elif clus_method == 'hc' and inverted_instance_type == 'addr':
            model_id = ''
        elif clus_method == 'grid' and inverted_instance_type == 'geocoding':
            model_id = ''
        else:
            raise Exception('invalid combination')
#         model_name = 'LocMatcher-nP'
#         model_id = ''
#         model_name = 'LocMatcher-nD'
#         model_id = ''
#         model_name = 'LocMatcher-nLC'
#         model_id = ''
#         model_name = 'LocMatcher-nTC'
#         model_id = ''
#         model_name = 'LocMatcher-nA'
#         model_id = ''
        model_path = os.path.join(global_result_path, 'saved_model/loc_selector_S{}_R{}_BD{}_D{}_LQ{}-{}_seed{}{}{}/{}/H{}_{}/'.format(
                behavior_min_trips, behavior_min_rate, batch_delivery_times, clus_dist_thresh, min_delvs, min_conf, seed, clus_method_suffix, lc_type_suffix, model_name, hidden_dim, model_id))
        train_data_path = os.path.join(global_result_path,
                                       'learning_samples_selection_S{}-R{}_BD{}_D{}_LQ{}-{}_seed{}/train/'.format(
                                           behavior_min_trips, behavior_min_rate, batch_delivery_times,
                                           clus_dist_thresh, min_delvs, min_conf, seed))
        if model_name == 'Trans-nA':
            return DLInf(candi_delv_locs_path, hidden_dim, model_name, model_path, train_data_path, trip_inverted_data, geocoding2poi_type, use_addr=False)
        else:
            return DLInf(candi_delv_locs_path, hidden_dim, model_name, model_path, train_data_path, trip_inverted_data, geocoding2poi_type, use_addr=True)
    elif algo_name == 'DLInfTradi':
        model_name = 'SVM'
        if model_name == 'RF':
            params = (10, 400)
            model_id = ''
        elif model_name == 'GBDT':
            params = '150'
            model_id = ''
        elif model_name == 'SVM':
            params = ''
            model_id = ''
        else:
            raise Exception('not imple')
        model_path = os.path.join(global_result_path, 'saved_model/loc_selector_S{}_R{}_BD{}_D{}_LQ{}-{}_seed{}/{}/H{}_{}/'.format(
                behavior_min_trips, behavior_min_rate, batch_delivery_times, clus_dist_thresh, min_delvs, min_conf, seed, model_name, params, model_id))
        train_data_path = os.path.join(global_result_path, 'learning_samples_S{}-R{}_BD{}_D{}_LQ{}-{}_seed{}/train/'.format(
            behavior_min_trips, behavior_min_rate, batch_delivery_times, clus_dist_thresh, min_delvs, min_conf, seed))
        return DLInfTradi(candi_delv_locs_path, model_name, model_path, train_data_path, trip_inverted_data, geocoding2poi_type)
    else:
        raise Exception('unknown algo')


if __name__ == '__main__':
    station_ids = ['']
    base_dir = '../data/'
    global_result_path = base_dir + 'result_DLInf/'
    with open(os.path.join(base_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    dist_thresh, min_stay_time = params['preprocessing']['dist_thresh'], params['preprocessing']['min_stay_time']
    clus_dist_thresh = 50
    batch_delivery_times = 5
    behavior_min_trips = 10
    behavior_min_rate = 0.8
    train_rate = 0.8
    val_rate = 0.1
    seed = 2017
    gt_cluster_dist_thresh = 30
    min_delvs = 2
    min_conf = 0.51
    # inverted_instance_type = 'addr'
    clus_method = 'hc'
    inverted_instance_type = 'geocoding'
    # clus_method = 'grid'
    clus_method_suffix = '_www10' if clus_method == 'grid' else ''
    lc_type_suffix = '_addr_inverted' if inverted_instance_type == 'addr' else ''
    # algo_name = 'Annotation'
    # algo_name = 'Geocoding'
    # algo_name = 'MinDist'
    # algo_name = 'MaxTC'
    # algo_name = 'MaxTCILC'
    # algo_name = 'DLInfTradi'
    # algo_name = 'DLInfMLP'
    # algo_name = 'GeoCloud'
    algo_name = 'DLInf'

    start = datetime.now()
    print(start.strftime("%Y-%m-%d %H:%M:%S"))
    res = []
    for sta_id in station_ids:
        station_dir = base_dir + '{}/'.format(sta_id)
        result_dir = station_dir + 'result_DLInf/'
        dataset_path = os.path.join(result_dir, 'candi_delv_sps_{}_{}_S{}_R{}_BD{}/'.format(
            dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate, batch_delivery_times))
        candi_delv_locs_path = os.path.join(result_dir, 'candi_delivery_locations_D{}{}.csv'.format(clus_dist_thresh,
                                                                                                    clus_method_suffix))
        indexed_locs, id2locs = get_location_candidates(candi_delv_locs_path)
        label_path = os.path.join(result_dir,
                                  'delv_locs_{}_{}_S{}_R{}_Dgt{}_user.csv'.format(dist_thresh, min_stay_time,
                                                                                  behavior_min_trips, behavior_min_rate,
                                                                                  gt_cluster_dist_thresh))
        _, _, station_test_couriers = get_train_val_test_couriers(label_path, train_rate, val_rate, seed)
        station_test_labels = get_good_quality_labels(label_path, station_test_couriers, indexed_locs, min_delvs,
                                                      min_conf)
        print('{}:{}'.format(sta_id, len(station_test_labels)))
        instance_loc_inverted_path = os.path.join(result_dir, '{}_loc_inverted_data_S0_R0_D{}.pkl'.format(
            inverted_instance_type, clus_dist_thresh))
        trip_inverted_data = get_instance_loc_to_trips(instance_loc_inverted_path)
        with open(os.path.join(result_dir, 'geocoding2poi_type_{}_{}_S{}_R{}.pkl'.format(dist_thresh, min_stay_time,
                                                                                         behavior_min_trips,
                                                                                         behavior_min_rate)),
                  'rb') as f:
            geocoding2poi_type = pickle.load(f)
        algo = get_algo(algo_name, candi_delv_locs_path, inverted_instance_type, trip_inverted_data, geocoding2poi_type)
        with multiprocessing.Pool() as pool:
            r = pool.map(evaluate_label, station_test_labels)
        res.append(r)
    end = datetime.now()
    print(end.strftime("%Y-%m-%d %H:%M:%S"))
    print('time cost:{}min'.format((end - start).total_seconds() / 60.0))
    all_res = [err for sta_err in res for err, _, _ in sta_err]
    acc = sum([is_matched for sta_err in res for _, is_matched, _ in sta_err]) / len(all_res)
    rmse = math.sqrt(sum([e * e for e in all_res]) / len(all_res))
    mae = sum(all_res) / len(all_res)
    print('Algo Name:{}'.format(algo_name))
    print('ACC:{}'.format(acc))
    print('RMSE:{}'.format(rmse))
    print('MAE:{}'.format(mae))
