import copy

import numpy as np
import pandas as pd
import argparse
from model.SEIRQD import SEIRQD

parser = argparse.ArgumentParser()
parser.add_argument('--city_name', type=str, default='beijing',
                    help='To predict the urban epidemic information by name')
# month2：两个月
# year：一年
parser.add_argument('--folder', type=str, default='month2',
                    help='Data folder')
args = parser.parse_args()

# 新十条2022年12月7日数据
seir_data_beijing = {
    "n": 21890000,  # 该城市人口总数
    "city_name": "北京",  # 城市名称
    "susceptible": [21874977.0],  # 易感者
    "exposed": [67780.0],  # 暴露者
    "infectious_s": [1170.0],  # 感染者 中轻度患者
    "infectious_a": [2804.0],  # 感染者 无症状患者
    "infectious_u": [0.0],  # 感染者 重症状患者
    "quarantine_s": [0.0],  # 感染者 中轻度隔离患者
    "quarantine_a": [0.0],  # 感染者 无症状隔离患者
    "recovered": [11036.0],  # 康复者
    "dead": [13.0],  # 死亡者
    "predict_total": [0.0],  # 预测的患者 重症状患者+中轻度隔离患者+无症状隔离患者
    "predict_all": [15023.0]  # 所有的患病情况
}

# 获取数据，开始运行
city_name = args.city_name
real = pd.read_csv("./data/" + city_name + ".csv", encoding='utf-8')
names = ["date", "population shift"]
population_shift = np.array(real['population shift'])
population_shift = [float(i) for i in population_shift]
time = np.array(real['date'])
folder = args.folder

t = 0.9999
for num in range(0, 8):
    ans = SEIRQD(copy.deepcopy(seir_data_beijing), population_shift, time, None,
                 r_is=10.0, r_ia=20.0, beta_is=0.126, beta_ia=0.063,
                 t=t, alpha=3.0, i=float(num), c=0.15,
                 theta_s=0.8, theta_a=0.6, gamma_s1=7.0, gamma_a1=7.0, gamma_u=15.0, p=0.00009, m=0.0247)
    ans.train(beta_is=0.126, beta_ia=0.063)
    ans.data["predict_total"] = [int(i) for i in ans.data["predict_total"]]
    ans.drawGraph(path='./data/' + folder + '/result_{}_t=' + str(t) + '_i=' + str(num) + '.png')
    ans.saveResultToExcel(path='./data/' + folder + '/result_{}_t=' + str(t) + '_i=' + str(num) + '.xls')

t = 1.0
for num in range(0, 8):
    population_shift = [0 for i in range(len(time))]
    ans = SEIRQD(copy.deepcopy(seir_data_beijing), population_shift, time, None,
                 r_is=10.0, r_ia=20.0, beta_is=0.126, beta_ia=0.063,
                 t=t, alpha=3.0, i=float(num), c=0.15,
                 theta_s=0.8, theta_a=0.6, gamma_s1=7.0, gamma_a1=7.0, gamma_u=15.0, p=0.00009, m=0.0247)
    ans.train(beta_is=0.126, beta_ia=0.063)
    ans.data["predict_total"] = [int(i) for i in ans.data["predict_total"]]
    ans.drawGraph(path='./data/' + folder + '/result_{}_t=' + str(t) + '_i=' + str(num) + '.png')
    ans.saveResultToExcel(path='./data/' + folder + '/result_{}_t=' + str(t) + '_i=' + str(num) + '.xls')
