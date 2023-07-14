import copy

import numpy as np
import pandas as pd
import argparse
from model.SEIRQD import SEIRQD

parser = argparse.ArgumentParser()
parser.add_argument('--city_name', type=str, default='world',
                    help='To predict the urban epidemic information by name')
# month2：两个月
# year：一年
parser.add_argument('--folder', type=str, default='month2',
                    help='Data folder')
args = parser.parse_args()

# 新十条2022年12月7日数据
seir_data_china = {
    "n": 1400000000,  # 该城市人口总数
    "city_name": "world",  # 城市名称
    "susceptible": [1400000000.0],  # 易感者
    "exposed": [0.0],  # 暴露者
    "infectious_s": [4079.0],  # 感染者 中轻度患者
    "infectious_a": [17360.0],  # 感染者 无症状患者

    # "infectious_u": [0.0],  # 感染者 重症状患者
    # "quarantine_s": [0.0],  # 感染者 中轻度隔离患者
    # "quarantine_a": [0.0],  # 感染者 无症状隔离患者
    "quarantine": [0.0],  # 自我隔离患者
    "recovered": [3767.0],  # 康复者
    "predict_total": [0.0],  # 预测的患者 重症状患者+中轻度隔离患者+无症状隔离患者
    # "predict_all": [15023.0]  # 所有的患病情况
}

# 获取数据，开始运行
ans = SEIRQD(copy.deepcopy(seir_data_china), time=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             r_is=60.0, r_ia=40.0, beta_is=0.5, beta_ia=0.5, alpha=3.0, c=0.7, q=0.03, al=0.0083,
             gamma_s1=10.0, gamma_a1=7.0)
ans.train()
ans.data["predict_total"] = [int(i) for i in ans.data["predict_total"]]
# ans.drawGraph()
ans.saveResultToExcel()
