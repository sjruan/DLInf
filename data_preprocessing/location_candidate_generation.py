import json
import os
from tqdm import tqdm
from datetime import datetime
from tptk.common.trajectory import parse_traj_file
from data_preprocessing.stay_point_extraction import construct_stay_points
import numpy as np
from data_preprocessing.ground_truth_construction import cluster_stay_points, get_centroid
from rtree import Rtree
from tptk.common.temporal_idx import DailyTemporalIdx
from tptk.common.grid import Grid
from tptk.common.mbr import MBR
from tptk.common.spatial_func import LAT_PER_METER, LNG_PER_METER, SPoint
import multiprocessing


def get_location_candidates(candi_delv_locs_path):
    indexed_locs = Rtree()
    id2loc = {}
    with open(candi_delv_locs_path, 'r') as f:
        f.readline()
        for line in f.readlines():
            arr = line.strip().split(',')
            id = int(arr[0])
            lat, lng, avg_duration, avg_size, time_dist_str, nb_couriers = float(arr[1]), float(arr[2]), float(arr[3]), \
                                                              float(arr[4]), arr[5], int(arr[6])
            time_dist = []
            for attr in time_dist_str.split('_'):
                time_dist.append(float(attr))
            id2loc[id] = (SPoint(lat, lng), avg_duration, avg_size, time_dist, nb_couriers)
            indexed_locs.insert(id, (lng, lat, lng, lat))
    return indexed_locs, id2loc


def get_all_stay_points(couriers_dir, reulst_dir, dist_thresh, min_stay_time):
    # (courier_id,time,lat,lng)
    data = []
    for courier_id in os.listdir(couriers_dir):
        clean_dir = os.path.join(couriers_dir, courier_id, 'clean')
        clean_traj_with_status_dir = os.path.join(clean_dir, 'trajs_with_status_{}_{}'.format(
            dist_thresh, min_stay_time))
        for traj_filename in tqdm(os.listdir(clean_traj_with_status_dir)):
            if not traj_filename.endswith('.csv'):
                continue
            daily_traj = parse_traj_file(os.path.join(clean_traj_with_status_dir, traj_filename),
                                         extra_fields=['stay'])[0]
            daily_sps = construct_stay_points(daily_traj)
            for sp in daily_sps:
                data.append((courier_id, sp.get_mid_time(), sp.get_centroid().lat, sp.get_centroid().lng,
                             sp.get_duration(), sp.get_mbr().get_w() * sp.get_mbr().get_h()))
    with open(os.path.join(reulst_dir, 'all_stay_points.csv'), 'w') as f:
        f.write('courier_id,time,lat,lng,duration,size\n')
        for i in range(len(data)):
            d = data[i]
            f.write('{},{},{},{},{},{}\n'.format(d[0], datetime.strftime(d[1], '%Y-%m-%d %H:%M:%S'), d[2], d[3], d[4], d[5]))


def cluster_all_stay_points(result_dir, clus_dist_thresh, take=10000, seed=2017):
    data = []
    with open(os.path.join(result_dir, 'all_stay_points.csv'), 'r') as f:
        f.readline()
        for line in f.readlines():
            arr = line.strip().split(',')
            data.append((float(arr[2]), float(arr[3]), datetime.strptime(arr[1], '%Y-%m-%d %H:%M:%S'), float(arr[4]), float(arr[5]), arr[0]))
    nb_samples = len(data)
    idxes = np.random.RandomState(seed=seed).permutation(nb_samples)
    sampled = [data[idx] for idx in idxes[:take]]
    cluster_inp = np.asarray([(sample[0], sample[1]) for sample in sampled])
    print(cluster_inp.shape)
    cluster_labels = cluster_stay_points(cluster_inp, clus_dist_thresh)
    with open(os.path.join(result_dir, 'stay_points_clustered_D{}.csv'.format(clus_dist_thresh)), 'w') as f:
        f.write('lat,lng,time,duration,size,courier,cluster\n')
        for i in range(len(sampled)):
            d = sampled[i]
            f.write('{},{},{},{},{},{},{}\n'.format(d[0], d[1], d[2].strftime('%Y-%m-%d %H:%M:%S'), d[3], d[4], d[5], cluster_labels[i]))


def generate_candi_delivery_locations(result_dir, clus_dist_thresh):
    cluster2pts = {}
    with open(os.path.join(result_dir, 'stay_points_clustered_D{}.csv'.format(clus_dist_thresh)), 'r') as f:
        f.readline()
        for line in f.readlines():
            arr = line.strip().split(',')
            lat, lng, time, duration, size, courier, cluster_id = float(arr[0]), float(arr[1]), \
                                                         datetime.strptime(arr[2], '%Y-%m-%d %H:%M:%S'), float(arr[3]), float(arr[4]), arr[5], int(arr[6])
            if cluster_id not in cluster2pts:
                cluster2pts[cluster_id] = []
            cluster2pts[cluster_id].append((lng, lat, time, duration, size, courier))
    print('#clusters:{}'.format(len(cluster2pts)))
    daily_temporal_idx = DailyTemporalIdx(start_minutes=8*60, end_minutes=23*60,
                                          time_interval_in_minutes=60)
    with open(os.path.join(result_dir, 'candi_delivery_locations_D{}.csv'.format(clus_dist_thresh)), 'w') as f:
        f.write('cluster,lat,lng,avg_duration,avg_size,time_dist,nb_couriers\n')
        for cluster_id in tqdm(cluster2pts):
            pts = cluster2pts[cluster_id]
            lng, lat = get_centroid([(pt[0], pt[1]) for pt in pts])
            avg_duration = sum([pt[3] for pt in pts]) / len(pts)
            avg_size = sum([pt[4] for pt in pts]) / len(pts)
            time_dist = np.zeros(daily_temporal_idx.ts_num)
            for pt in pts:
                try:
                    t_idx = daily_temporal_idx.time_to_ts(pt[2])
                    time_dist[t_idx] += 1.0
                except:
                    continue
            if np.sum(time_dist) != 0:
                time_dist = time_dist / np.sum(time_dist)
            time_dist_str = '_'.join([str(p) for p in time_dist])
            nb_couriers = len(set([pt[5] for pt in pts]))
            f.write('{},{},{},{},{},{},{}\n'.format(cluster_id, lat, lng, avg_duration, avg_size, time_dist_str, nb_couriers))


def generate_candi_delivery_locations_www10(result_dir, clus_dist_thresh):
    grid_size = clus_dist_thresh / 3
    data = []
    min_lat = float('inf')
    min_lng = float('inf')
    max_lat = 0
    max_lng = 0
    with open(os.path.join(result_dir, 'all_stay_points.csv'), 'r') as f:
        f.readline()
        for line in f.readlines():
            arr = line.strip().split(',')
            lat, lng = float(arr[2]), float(arr[3])
            if lat < min_lat:
                min_lat = lat
            elif lat > max_lat:
                max_lat = lat
            if lng < min_lng:
                min_lng = lng
            elif lng > max_lng:
                max_lng = lng
            data.append((lat, lng, datetime.strptime(arr[1], '%Y-%m-%d %H:%M:%S'), float(arr[4]), float(arr[5]), arr[0]))
    nb_rows = int(((max_lat - min_lat) / LAT_PER_METER) / grid_size) + 1
    nb_cols = int(((max_lng - min_lng) / LNG_PER_METER) / grid_size) + 1
    mbr_min_lat = min_lat - 0.0001
    mbr_min_lng = min_lng - 0.0001
    mbr_max_lat = max_lat + 0.0001
    mbr_max_lng = max_lng + 0.0001
    mbr = MBR(mbr_min_lat, mbr_min_lng, mbr_max_lat, mbr_max_lng)
    grid = Grid(mbr, nb_rows, nb_cols)
    grid2sps = {}
    for record in data:
        lat, lng, time, duration, size, courier = record
        row_idx, col_idx = grid.get_matrix_idx(lat, lng)
        if (row_idx, col_idx) not in grid2sps:
            grid2sps[(row_idx, col_idx)] = [(SPoint(lat, lng), time, duration, size, courier)]
        else:
            grid2sps[(row_idx, col_idx)].append((SPoint(lat, lng), time, duration, size, courier))
    grid2cnt = {}
    for i in range(nb_rows):
        for j in range(nb_cols):
            if (i, j) in grid2sps:
                grid2cnt[(i, j)] = len(grid2sps[(i, j)])
    cluster2grids = {}
    avail_cluster_id = 0
    while len(grid2cnt) > 0:
        hottest_grid_i, hottest_grid_j = max(grid2cnt, key=grid2cnt.get)
        cluster_grids = []
        for i in range(max(0, hottest_grid_i - 1), min(hottest_grid_i + 2, nb_rows)):
            for j in range(max(0, hottest_grid_j - 1), min(hottest_grid_j + 2, nb_cols)):
                if (i, j) in grid2cnt:
                    cluster_grids.append((i, j))
                    del grid2cnt[(i, j)]
        cluster2grids[avail_cluster_id] = cluster_grids
        avail_cluster_id += 1
    daily_temporal_idx = DailyTemporalIdx(start_minutes=8 * 60, end_minutes=23 * 60,
                                          time_interval_in_minutes=60)
    with open(os.path.join(result_dir, 'candi_delivery_locations_D{}_www10.csv'.format(clus_dist_thresh)), 'w') as f:
        f.write('cluster,lat,lng,avg_duration,avg_size,time_dist,nb_couriers\n')
        for cluster in cluster2grids:
            cluster_grids = cluster2grids[cluster]
            all_sps = []
            for grid in cluster_grids:
                all_sps.extend(grid2sps[grid])
            mean_lat = sum([sp[0].lat for sp in all_sps]) / len(all_sps)
            mean_lng = sum([sp[0].lng for sp in all_sps]) / len(all_sps)
            avg_duration = sum([pt[2] for pt in all_sps]) / len(all_sps)
            avg_size = sum([pt[3] for pt in all_sps]) / len(all_sps)
            time_dist = np.zeros(daily_temporal_idx.ts_num)
            for pt in all_sps:
                try:
                    t_idx = daily_temporal_idx.time_to_ts(pt[2])
                    time_dist[t_idx] += 1.0
                except:
                    continue
            if np.sum(time_dist) != 0:
                time_dist = time_dist / np.sum(time_dist)
            time_dist_str = '_'.join([str(p) for p in time_dist])
            nb_couriers = len(set([sp[4] for sp in all_sps]))
            f.write('{},{},{},{},{},{},{}\n'.format(cluster, mean_lat, mean_lng, avg_duration, avg_size, time_dist_str, nb_couriers))


def get(sta_id):
    print(sta_id)
    station_dir = base_dir + '{}/'.format(sta_id)
    result_dir = station_dir + 'result_DLInf/'
    if os.path.exists(os.path.join(result_dir, 'all_stay_points.csv')):
        return
    couriers_dir = os.path.join(station_dir, 'couriers')
    get_all_stay_points(couriers_dir, result_dir, dist_thresh, min_stay_time)


def generate(sta_id):
    print(sta_id)
    station_dir = base_dir + '{}/'.format(sta_id)
    result_dir = station_dir + 'result_DLInf/'
    for clus_dist_thresh in [30, 40, 50, 60, 70]:
        cluster_all_stay_points(result_dir, clus_dist_thresh, take=25000)
        generate_candi_delivery_locations(result_dir, clus_dist_thresh)
    # generate_candi_delivery_locations_www10(result_dir, clus_dist_thresh=50)


if __name__ == '__main__':
    station_ids = ['']
    base_dir = '../data/'
    with open(os.path.join(base_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    dist_thresh, min_stay_time = params['preprocessing']['dist_thresh'], params['preprocessing']['min_stay_time']

    # Get All Stay Points
    with multiprocessing.Pool() as pool:
        pool.map(get, station_ids)

    # Generate Candidate Delivery Locations
    with multiprocessing.Pool() as pool:
        pool.map(generate, station_ids)
