from tptk.common.trajectory import Trajectory, STPoint, store_traj_file, get_tid, parse_traj_file
from tptk.stay_point_detection import find_first_exceed_max_distance, exceed_max_time
from tptk.common.mbr import MBR
from tptk.noise_filtering import HeuristicFilter, STFilter
from datetime import datetime
import multiprocessing
import timeit
import os
import json
import pandas as pd


def construct_stay_points(traj):
    stay_points = []
    open_flag = False
    sp_pt_list = []
    for pt in traj.pt_list:
        if pt.data['stay'] is True:
            if not open_flag:
                open_flag = True
            sp_pt_list.append(pt)
        else:
            if open_flag:
                stay_points.append(Trajectory(traj.oid, get_tid(traj.oid, sp_pt_list), sp_pt_list))
                open_flag = False
                sp_pt_list = []
    if open_flag:
        stay_points.append(Trajectory(traj.oid, get_tid(traj.oid, sp_pt_list), sp_pt_list))
    return stay_points


def load_hive_trajs(traj_path):
    date_parser = lambda date_str: pd.datetime.strptime(date_str, '%Y-%m-%d') \
        if isinstance(date_str, str) else date_str
    traj_df = pd.read_csv(traj_path, usecols=['psy_id', 'date', 'traj'], parse_dates=['date'],
                          sep='\t', encoding='utf-8', date_parser=date_parser)
    return traj_df


def load_hive_waybills(waybill_path):
    date_parser = lambda date_str: pd.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') \
        if isinstance(date_str, str) else date_str
    waybill_df = pd.read_csv(waybill_path, parse_dates=['rev_confirm_tm', 'succ_delv_tm'],
                             date_parser=date_parser, dtype={'rev_confirm_operr_id': str, 'succ_delv_operr_id': str,
                                                             'shipping_bill_id': str, 'memberid': str},
                             usecols=['shipping_bill_id', 'id', 'memberid', 'rev_confirm_tm', 'succ_delv_operr_id', 'succ_delv_tm', 'wgs_lat', 'wgs_lng'],
                             sep='\t', encoding='utf-8')
    return waybill_df


def do_clean(raw_traj, filters):
    clean_traj = raw_traj
    for filter in filters:
        clean_traj = filter.filter(clean_traj)
        if clean_traj is None:
            return None
    return clean_traj


def clean_station_traj(row):
    oid = str(row[0])
    traj_str_attrs = row[2].split('|')
    pt_list = []
    for pt_arr in traj_str_attrs:
        pt_attrs = pt_arr.split(',')
        tm = datetime.strptime(pt_attrs[0], pt_tm_format)
        lat = float(pt_attrs[1])
        lng = float(pt_attrs[2])
        pt_list.append(STPoint(lat, lng, tm))
    if len(pt_list) < 2:
        return
    raw_traj = Trajectory(oid, get_tid(oid, pt_list), pt_list)
    clean_traj = do_clean(raw_traj, filters)
    if clean_traj is not None:
        dest_traj_path = os.path.join(base_dir, sta_id, 'couriers', oid, 'clean', 'trajs')
        if not os.path.exists(dest_traj_path):
            try:
                os.makedirs(dest_traj_path)
            except:
                pass
        store_traj_file([clean_traj], os.path.join(dest_traj_path, '{}_{}_traj_jd_wgs.csv'.format(row[1].strftime(file_tm_format), oid)))


def add_stay_point_status(traj, stay_dist_thresh, max_stay_time):
    pt_list = traj.pt_list
    cur_idx = 0
    traj_idx = 0
    while cur_idx < len(pt_list) - 1:
        next_idx = find_first_exceed_max_distance(pt_list, cur_idx, stay_dist_thresh)
        if exceed_max_time(pt_list, cur_idx, next_idx, max_stay_time):
            for i in range(cur_idx, next_idx):
                pt = pt_list[i]
                if pt.data is None:
                    pt.data = {'stay': True}
                else:
                    pt.data['stay'] = True
            if traj_idx < cur_idx:
                # if only one point is moving, we regard it as stay
                if traj_idx == cur_idx - 1:
                    pt = pt_list[traj_idx]
                    if pt.data is None:
                        pt.data = {'stay': True}
                    else:
                        pt.data['stay'] = True
                else:
                    for i in range(traj_idx, cur_idx):
                        pt = pt_list[i]
                        if pt.data is None:
                            pt.data = {'stay': False}
                        else:
                            pt.data['stay'] = False
            traj_idx = next_idx
            cur_idx = next_idx
        else:
            cur_idx += 1
    if traj_idx < len(pt_list):
        # if only one point is moving, we regard it as stay
        if traj_idx == len(pt_list) - 1:
            pt = pt_list[traj_idx]
            if pt.data is None:
                pt.data = {'stay': True}
            else:
                pt.data['stay'] = True
        else:
            for i in range(traj_idx, len(pt_list)):
                pt = pt_list[i]
                if pt.data is None:
                    pt.data = {'stay': False}
                else:
                    pt.data['stay'] = False


def do_detection(courier):
    clean_dir = os.path.join(couriers_dir, courier, 'clean')
    clean_traj_dir = os.path.join(clean_dir, 'trajs')
    clean_traj_with_status_dir = os.path.join(clean_dir, 'trajs_with_status_{}_{}_tmp'.format(
        dist_thresh, min_stay_time))
    if not os.path.exists(clean_traj_with_status_dir):
        os.makedirs(clean_traj_with_status_dir)
    for traj_filename in os.listdir(clean_traj_dir):
        if not traj_filename.endswith('.csv'):
            continue
        clean_traj = parse_traj_file(os.path.join(clean_traj_dir, traj_filename))[0]
        add_stay_point_status(clean_traj, dist_thresh, min_stay_time)
        store_traj_file([clean_traj], os.path.join(clean_traj_with_status_dir, traj_filename),
                        extra_fields=['stay'])


if __name__ == '__main__':
    station_ids = ['']
    base_dir = '../data/'
    hive_dir = os.path.join(base_dir, 'hive')
    with open(os.path.join(base_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    dist_thresh, min_stay_time = params['preprocessing']['dist_thresh'], params['preprocessing']['min_stay_time']
    file_tm_format = '%Y%m%d'
    for sta_id in station_ids:
        print(sta_id)
        # noise filtering
        with open(os.path.join(base_dir, 'station_info_{}.json'.format(sta_id)), 'r') as f:
            station_info = json.load(f)
        time_fmt = '%Y-%m-%d'
        start_time = datetime.strptime(station_info['start_time'], time_fmt)
        end_time = datetime.strptime(station_info['end_time'], time_fmt)
        delv_mbr = MBR(station_info['delv_min_lat'], station_info['delv_min_lng'],
                       station_info['delv_max_lat'], station_info['delv_max_lng'])
        st_filter = STFilter(delv_mbr, start_time, end_time)
        heuristic_filter = HeuristicFilter(max_speed=params['preprocessing']['max_speed'])
        filters = [st_filter, heuristic_filter]
        traj_filename = 'raw_trajs_{}_20180101_20190901.csv'.format(sta_id)
        traj_path = os.path.join(hive_dir, 'trajs', traj_filename)
        traj_df = load_hive_trajs(traj_path)
        print('traj file loaded')
        file_tm_format = '%Y%m%d'
        pt_tm_format = '%Y-%m-%d %H:%M:%S'
        p = multiprocessing.Pool()
        start = timeit.default_timer()
        p.map(clean_station_traj, traj_df.values.tolist())
        p.close()
        p.join()
        end = timeit.default_timer()
        print('multi processing time:{}s'.format(str(end - start)))

        # stay point detection
        station_dir = base_dir + '{}/'.format(sta_id)
        couriers_dir = os.path.join(station_dir, 'couriers')
        couriers = os.listdir(couriers_dir)
        start = timeit.default_timer()
        with multiprocessing.Pool(20) as pool:
            pool.map(do_detection, couriers)
        end = timeit.default_timer()
        print('multi processing time:{}s'.format(str(end - start)))
