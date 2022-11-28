import numpy as np
import pandas as pd
import argparse
from model.SEIRQD import SEIRQD

parser = argparse.ArgumentParser()
parser.add_argument('--city_name', type=str, default='xian', help='To predict the urban epidemic information by name')
args = parser.parse_args()

seir_data_xian = {
    "n": 13163000.0,  # 该城市人口总数
    "city_name": "西安",  # 城市名称
    "susceptible": [13163000.0],  # 易感者
    "exposed": [0.0],  # 暴露者
    "infectious_s": [0.0],  # 感染者 中轻度患者
    "infectious_a": [3.0],  # 感染者 无症状患者
    "infectious_u": [0.0],  # 感染者 重症状患者
    "quarantine_s": [0.0],  # 感染者 中轻度隔离患者
    "quarantine_a": [0.0],  # 感染者 无症状隔离患者
    "recovered": [0.0],  # 康复者
    "dead": [0.0],  # 死亡者
    "real_patients": [],  # 真实病例
    "predict_total": [3.0]  # 合计
}

seir_data_shanghai = {
    "n": 24894300.0,  # 该城市人口总数
    "city_name": "上海",  # 城市名称
    "susceptible": [24894300.0],  # 易感者
    "exposed": [0.0],  # 暴露者
    "infectious_s": [0.0],  # 感染者 中轻度患者
    "infectious_a": [2.0],  # 感染者 无症状患者
    "infectious_u": [0.0],  # 感染者 重症状患者
    "quarantine_s": [0.0],  # 感染者 中轻度隔离患者
    "quarantine_a": [0.0],  # 感染者 无症状隔离患者
    "recovered": [0.0],  # 康复者
    "dead": [0.0],  # 死亡者
    "real_patients": [],  # 真实病例
    "predict_total": [2.0]  # 合计
}

# 获取数据，开始运行
city_name = args.city_name
real = pd.read_csv("data/" + city_name + ".csv", encoding='utf-8')
names = ["date", "population shift", "patients"]
population_shift = np.array(real['population shift'])
population_shift = [float(i) for i in population_shift]
time = np.array(real['date'])
real_patients = np.array(real['patients'])

ans = None
if city_name == 'xian':
    ans = SEIRQD(seir_data_xian, population_shift, time, real_patients,
                 r_is=20.0, r_ia=40.0, beta_is=0.046, beta_ia=0.023,
                 t=1.0, alpha=4.4, i=3.0, c=0.35,
                 theta_s=0.8, theta_a=0.6, gamma_s1=10.0, gamma_a1=10.0, gamma_u=10.0, p=0.15, m=0.064)
elif city_name == 'shanghai':
    ans = SEIRQD(seir_data_shanghai, population_shift, time, real_patients,

                 r_is=20.0, r_ia=40.0, beta_is=0.046, beta_ia=0.023,
                 t=1.0, alpha=4, i=2.0, c=0.25,
                 theta_s=0.8, theta_a=0.6, gamma_s1=10.0, gamma_a1=10.0, gamma_u=10.0, p=0.15, m=0.064)
else:
    FileNotFoundError()

# beta_is : beta_ia = 1.3~2 : 1
ans.train()

# 看看结果
ans.data["real_patients"] = [int(i) for i in real_patients]
ans.data["predict_total"] = [int(i) for i in ans.data["predict_total"]]
ans.drawGraph()
# ans.saveResultAsExcel()

# 绝对值和平方之前
