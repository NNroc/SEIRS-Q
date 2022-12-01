import numpy as np
import pandas as pd
import argparse
from model.SEIRQD import SEIRQD

parser = argparse.ArgumentParser()
parser.add_argument('--city_name', type=str, default='beijing', help='To predict the urban epidemic information by name')
args = parser.parse_args()

seir_data_beijing = {
    "n": 0,  # 该城市人口总数
    "city_name": "北京",  # 城市名称
    "susceptible": [21890000.0],  # 易感者
    "exposed": [0.0],  # 暴露者
    "infectious_s": [0.0],  # 感染者 中轻度患者
    "infectious_a": [3.0],  # 感染者 无症状患者
    "infectious_u": [0.0],  # 感染者 重症状患者
    "quarantine_s": [0.0],  # 感染者 中轻度隔离患者
    "quarantine_a": [0.0],  # 感染者 无症状隔离患者
    "recovered": [0.0],  # 康复者
    "dead": [0.0],  # 死亡者
    "predict_total": [3.0]  # 合计
}

# 获取数据，开始运行
city_name = args.city_name
real = pd.read_csv("data/" + city_name + ".csv", encoding='utf-8')
names = ["date", "population shift"]
population_shift = np.array(real['population shift'])
population_shift = [float(i) for i in population_shift]
time = np.array(real['date'])

ans = SEIRQD(seir_data_beijing, population_shift, time, None,
             r_is=20.0, r_ia=40.0, beta_is=0.126, beta_ia=0.063,
             t=0.0001, alpha=3.0, i=0, c=0.15,
             theta_s=0.8, theta_a=0.6, gamma_s1=10.0, gamma_a1=10.0, gamma_u=30.0, p=0.065, m=0.6)

ans.train(beta_is=0.126, beta_ia=0.063)

# 看看结果
ans.data["predict_total"] = [int(i) for i in ans.data["predict_total"]]
ans.drawGraph()
ans.saveResultToExcel()
print(ans.data)
