import json
import pickle
import os
from tqdm import tqdm
from datetime import datetime
from tptk.common.trajectory import parse_traj_file, store_traj_file, Trajectory, get_tid, STPoint
from tptk.query_utils import query_stay_points_by_temporal_range
from data_preprocessing.stay_point_extraction import construct_stay_points
from data_loader import query_waybills_delv_in_trip, load_clean_waybill
from data_preprocessing.courier_behavior_evaluation import get_good_couriers
from data_preprocessing.ground_truth_construction import get_courier2labels
import multiprocessing


def simulate_batch_confirmation(station_dir, label_path, batch_delivery_times, dataset_path, dist_thresh, min_stay_time):
    courier2labels = get_courier2labels(label_path)
    couriers_dir = os.path.join(station_dir, 'couriers')
    with open(os.path.join(station_dir, 'user2id.pkl'), 'rb') as f:
        user2id = pickle.load(f)
    for courier_id in courier2labels:
        print(courier_id)
        os.makedirs(dataset_path + '{}/'.format(courier_id), exist_ok=True)
        labels = courier2labels[courier_id]
        time_fmt = '%Y-%m-%d %H:%M:%S'
        addrs = set()
        # addr -> list of [candidate stay points in each trip]
        addr_historical_sps = {}
        nb_waybills = 0
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
                trip_added_addrs = set()
                start_time_lb = datetime.strptime(start_time_lb_str, time_fmt)
                first_delv_tm = datetime.strptime(first_delv_tm_str, time_fmt)
                end_time_lb = datetime.strptime(end_time_lb_str, time_fmt)
                end_time_ub = datetime.strptime(end_time_ub_str, time_fmt)
                trip_waybills = query_waybills_delv_in_trip(daily_delv_waybills, start_time_lb, end_time_ub)
                # sort waybills according to the delivery time
                trip_waybills.sort(key=lambda waybill: waybill.manual_delv_tm)
                trip_sps = query_stay_points_by_temporal_range(daily_sps, start_time_lb, end_time_ub)
                sps_during_delv = query_stay_points_by_temporal_range(daily_sps, first_delv_tm, end_time_lb,
                                                                      temporal_relation='intersect')
                if len(sps_during_delv) == 0:
                    continue
                nb_sps_per_group = int(len(sps_during_delv) / batch_delivery_times)
                group_sps = []
                for i in range(batch_delivery_times):
                    if i != batch_delivery_times - 1:
                        group_sps.append(sps_during_delv[i*nb_sps_per_group:(i+1)*nb_sps_per_group])
                    else:
                        group_sps.append(sps_during_delv[i*nb_sps_per_group:len(sps_during_delv)])
                batch_delivery_tss = [group[-1].get_mid_time() for group in group_sps if len(group) != 0]
                batch_delivery_tss[-1] = end_time_ub
                nb_waybills += len(trip_waybills)
                loc2cnt = {}
                for waybill in trip_waybills:
                    loc = (waybill.lat, waybill.lng)
                    if loc not in loc2cnt:
                        loc2cnt[loc] = 0
                    loc2cnt[loc] += 1
                for waybill in trip_waybills:
                    addrs.add((waybill.lat, waybill.lng))
                    addr = '{}_{}_{}'.format(user2id[waybill.member_id], waybill.lat, waybill.lng)
                    # we only search for the main courier for each address
                    if addr not in labels:
                        continue
                    # since for each trip, a user might place multiple orders,
                    # we should make sure that each trip corresponds to one record
                    if addr in trip_added_addrs:
                        continue
                    manual_delayed_delv_tm = None
                    for i in range(len(batch_delivery_tss)):
                        if i == 0:
                            if waybill.manual_delv_tm <= batch_delivery_tss[0]:
                                manual_delayed_delv_tm = batch_delivery_tss[0]
                                break
                        elif batch_delivery_tss[i-1] < waybill.manual_delv_tm <= batch_delivery_tss[i]:
                            manual_delayed_delv_tm = batch_delivery_tss[i]
                            break
                    candi_sps = query_stay_points_by_temporal_range(
                        trip_sps, start_time_lb, manual_delayed_delv_tm, temporal_relation='intersect')
                    if len(candi_sps) > 0:
                        if addr not in addr_historical_sps:
                            addr_historical_sps[addr] = []
                        sp_list = []
                        for sp in candi_sps:
                            sp_mbr = sp.get_mbr()
                            # we should normalize the duration with the parcel delivered
                            sp_list.append(STPoint(sp.get_centroid().lat, sp.get_centroid().lng, sp.get_mid_time(),
                                                   {'duration': sp.get_duration() / loc2cnt[(waybill.lat, waybill.lng)],
                                                    'area': sp_mbr.get_h() * sp_mbr.get_w()}))
                        addr_historical_sps[addr].append(
                            Trajectory(courier_id, get_tid(courier_id, sp_list), sp_list))
                        trip_added_addrs.add(addr)
        print('unique addrs:{}'.format(len(addr_historical_sps)))
        print('total waybills:{}'.format(nb_waybills))
        for addr in addr_historical_sps:
            store_traj_file(addr_historical_sps[addr],
                            os.path.join(dataset_path, courier_id, '{}.csv'.format(addr)),
                            extra_fields=['duration', 'area'])


def simulate(sta_id):
    station_dir = base_dir + '{}/'.format(sta_id)
    courier_confirmation_behavior_path = station_dir + 'courier_behaviors_T{}.pkl'.format(t_thresh)
    result_dir = station_dir + 'result_DLInf/'
    label_path = os.path.join(result_dir, 'delv_locs_{}_{}_S{}_R{}_Dgt{}_{}.csv'.format(
        dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate, gt_cluster_dist_thresh, 'user'))
    for batch_delivery_times in [1, 2, 3, 4, 5]:
        dataset_path = os.path.join(result_dir, 'candi_delv_sps_{}_{}_S{}_R{}_BD{}/'.format(
            dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate, batch_delivery_times))
        simulate_batch_confirmation(station_dir, label_path, batch_delivery_times, dataset_path, dist_thresh,
                                    min_stay_time)


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

    with multiprocessing.Pool() as pool:
        pool.map(simulate, station_ids)
