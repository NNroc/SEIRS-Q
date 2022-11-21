import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from model.SEIRQD import SEIRQD

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

# 开始运行
real = pd.read_csv("data/xian.csv", encoding='utf-8')
names = ["date", "population shift", "patients"]
population_shift = np.array(real['population shift'])
population_shift = [float(i) for i in population_shift]
time = np.array(real['date'])
real_patients = np.array(real['patients'])

ans = SEIRQD(seir_data_xian, population_shift, time, real_patients,
             r_is=20.0, r_ia=40.0, beta_is=0.046, beta_ia=0.023,
             t=1.0, alpha=4.4, i=3.0, c=0.4,
             theta_s=0.8, theta_a=0.6, gamma_s1=10.0, gamma_a1=10.0, gamma_u=10.0, p=0.15, m=0.064)
ans.train()

# 看看结果
ans.data["real_patients"] = [int(i) for i in real_patients]
ans.data["predict_total"] = [int(i) for i in ans.data["predict_total"]]
print(ans.data)
ans.drawGraph()
