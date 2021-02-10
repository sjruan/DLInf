import math
import pandas as pd
from tptk.common.spatial_func import SPoint


def load_waybill(waybill_path):
    date_parser = lambda date_str: pd.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') \
        if isinstance(date_str, str) else date_str
    waybill_df = pd.read_csv(waybill_path, parse_dates=['rev_confirm_tm', 'succ_delv_tm'],
                             date_parser=date_parser, dtype={'succ_delv_operr_id': str},
                             usecols=['shipping_bill_id', 'id', 'memberid',
                                      'goods_volume', 'goods_weight',
                                      'rev_confirm_tm', 'succ_delv_operr_id', 'succ_delv_tm',
                                      'wgs_lat', 'wgs_lng'],
                             sep=',', encoding='utf-8')
    return waybill_df


class Waybill(SPoint):
    def __init__(self, waybill_id, member_id, loc, recv_tm, manual_delv_tm, weight, volume):
        super(Waybill, self).__init__(loc.lat, loc.lng)
        self.waybill_id = waybill_id
        self.member_id = member_id
        self.recv_tm = recv_tm
        self.manual_delv_tm = manual_delv_tm
        self.weight = weight
        self.volume = volume

    def __hash__(self):
        return hash(str(self.waybill_id))

    def __repr__(self):
        return self.waybill_id


def load_clean_waybill(waybill_path):
    raw_waybill_df = load_waybill(waybill_path)
    clean_waybills = []
    for idx, row in raw_waybill_df.iterrows():
        confirmed_delivered_tm = row['succ_delv_tm'].to_pydatetime()
        recv_confirm_tm = row['rev_confirm_tm'].to_pydatetime()
        if pd.isnull(confirmed_delivered_tm) or pd.isnull(recv_confirm_tm):
            continue
        if recv_confirm_tm >= confirmed_delivered_tm:
            continue
        lat, lng = float(row['wgs_lat']), float(row['wgs_lng'])
        loc = SPoint(lat, lng)
        if math.isnan(lat) or math.isnan(lng):
            continue
        clean_waybills.append(Waybill(row['shipping_bill_id'], row['memberid'], loc,
                                      row['rev_confirm_tm'], row['succ_delv_tm'],
                                      row['goods_weight'], row['goods_volume']))
    return clean_waybills


def query_waybills_delv_in_trip(daily_waybills, start_time_lb, end_time_ub):
    trip_waybills = []
    for waybill in daily_waybills:
        if waybill.recv_tm <= start_time_lb < waybill.manual_delv_tm < end_time_ub:
            trip_waybills.append(waybill)
    return trip_waybills
