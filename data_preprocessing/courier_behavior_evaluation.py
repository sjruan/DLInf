import pickle
from tptk.common.spatial_func import SPoint, distance
import os
import json
from datetime import datetime
import multiprocessing
from data_loader import query_waybills_delv_in_trip, load_clean_waybill


def is_normal_confirmed(trip_waybills, max_time_interval_in_minutes):
    delv_tms = [waybill.manual_delv_tm for waybill in trip_waybills]
    delv_tms.sort()
    sub_tms = detect_batch_confirmation(delv_tms, max_time_interval_in_minutes)
    for sub_tm in sub_tms:
        if not is_batch_delivery_normal(sub_tm[0], sub_tm[-1], trip_waybills):
            return False
    return True


def is_batch_delivery_normal(st, et, trip_waybills):
    """
    actually, the loc2aoi is not used, because so many trips are detected as invalid
    """
    batch_delv_waybills = [waybill for waybill in trip_waybills if st <= waybill.manual_delv_tm <= et]
    locs = set()
    for waybill in batch_delv_waybills:
        loc = (waybill.lat, waybill.lng)
        locs.add(loc)
    locs = list(locs)
    if len(locs) == 1:
        return True
    # just for noise filtering
    pairwise_dists = []
    for i in range(len(locs)-1):
        for j in range(1, len(locs)):
            p_i = SPoint(locs[i][0], locs[i][1])
            p_j = SPoint(locs[j][0], locs[j][1])
            pairwise_dists.append(distance(p_i, p_j))
    if max(pairwise_dists) < 500:
        return True
    return False


def detect_batch_confirmation(delv_tms, max_time_interval_in_minutes):
    sub_tms = []
    tmp = [delv_tms[0]]
    for i in range(1, len(delv_tms)):
        if (delv_tms[i] - delv_tms[i - 1]).total_seconds() <= max_time_interval_in_minutes * 60.0:
            tmp.append(delv_tms[i])
        else:
            if len(tmp) > 1:
                sub_tms.append(tmp)
            tmp = [delv_tms[i]]
    if len(tmp) > 1:
        sub_tms.append(tmp)
    return sub_tms


def get_good_couriers(courier_confirmation_behavior_path, behavior_min_trips, behavior_min_rate):
    with open(courier_confirmation_behavior_path, 'rb') as f:
        courier_behavior = pickle.load(f)
    good_couriers = dict([(courier, courier_behavior[courier]) for courier in courier_behavior
                          if courier_behavior[courier][0] >= behavior_min_rate and
                          courier_behavior[courier][2] >= behavior_min_trips])
    return good_couriers


def get_courier_behavior(courier):
    courier_dir = os.path.join(couriers_dir, courier)
    with open(os.path.join(courier_dir, 'clean', 'trip_time_intervals.json'), 'r') as f:
            delivery_trip_intervals = json.load(f)
    waybill_dir = os.path.join(courier_dir, 'raw', 'waybills')
    waybill_filename = os.listdir(waybill_dir)[0]
    nb_trips = 0
    nb_normal_confirmed_trips = 0
    for day_trip_intervals in delivery_trip_intervals:
        nb_trips += len(day_trip_intervals)
        tmp = datetime.strptime(day_trip_intervals[0][0], time_fmt)
        delv_day = datetime(tmp.year, tmp.month, tmp.day, 0, 0, 0)
        delv_day_str = delv_day.strftime('%Y%m%d')
        daily_delv_waybills = load_clean_waybill(os.path.join(waybill_dir, delv_day_str + waybill_filename[8:]))
        for first_recv_tm_str, last_recv_tm_str, first_delv_tm_str, last_delv_tm_str, first_recv_tm_next_str in day_trip_intervals:
            start_time_lb = datetime.strptime(last_recv_tm_str, time_fmt)
            end_time_ub = datetime.strptime(first_recv_tm_next_str, time_fmt)
            trip_waybills = query_waybills_delv_in_trip(daily_delv_waybills, start_time_lb, end_time_ub)
            # first separate the waybills into several batch confirmation processes
            # if one process has delayed confirmation event, we think the whole trip has the problem
            if is_normal_confirmed(trip_waybills, t_thresh):
                nb_normal_confirmed_trips += 1
    if nb_trips == 0:
        return courier, None
    print('{},{},{},{}'.format(courier, nb_normal_confirmed_trips/nb_trips, nb_normal_confirmed_trips, nb_trips))
    return courier, (nb_normal_confirmed_trips/nb_trips, nb_normal_confirmed_trips, nb_trips)


if __name__ == '__main__':
    station_ids = ['']
    base_dir = '../data/'
    t_thresh = 0.5
    time_fmt = '%Y-%m-%d %H:%M:%S'
    for sta_id in station_ids:
        print('Station:{}'.format(sta_id))
        station_dir = base_dir + '{}/'.format(sta_id)
        courier_confirmation_behavior_path = station_dir + 'courier_behaviors_T{}.pkl'.format(t_thresh)
        couriers_dir = os.path.join(station_dir, 'couriers')
        couriers = os.listdir(couriers_dir)
        with multiprocessing.Pool(20) as pool:
            results = pool.map(get_courier_behavior, couriers)
        nb_tot = 0
        nb_tot_normal = 0
        courier2stats = {}
        for courier, stat in results:
            if stat is None:
                continue
            courier2stats[courier] = stat
            nb_tot_normal += stat[1]
            nb_tot += stat[2]
        print('normal rate:{}/{}={}'.format(nb_tot_normal, nb_tot, nb_tot_normal / nb_tot))
        with open(courier_confirmation_behavior_path, 'wb') as f:
            pickle.dump(courier2stats, f)
