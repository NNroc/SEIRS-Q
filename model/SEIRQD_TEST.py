# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from scipy.integrate import solve_ivp
# from scipy.optimize import minimize
#
#
# class SEIRQD:
#     def __init__(self, data: dict, a: dict, time: dict, real_patients: dict,
#                  r_is=20, r_ia=40, beta_is=0.05, beta_ia=0.025,
#                  t=1.0, alpha=4.4, i=3.0, c=0.4,
#                  theta_s=0.8, theta_a=0.6, gamma_s1=10.0, gamma_a1=10.0, gamma_u=10.0, p=0.15, m=0.064):
#         """
#         :param data: SEIR
#         :param a: 人口净流动量 原始数据集
#         :param time: 时间
#         :param real_patients: 真实病例数
#         :param r_is: 无症状感染者接易感人群的人数
#         :param r_ia: 感染者接触易感人群的人数
#         :param beta_is: 有症状感染系数 0.05 基于SEIR模型的高校新冠肺炎疫情传播风险管控研究
#         :param beta_ia: 无症状感染系数 0.025 基于SEIR模型的高校新冠肺炎疫情传播风险管控研究
#         :param t: 流动人口中的易感者比例	初设0（有隔离政策）、0.00001（无隔离政策）
#         :param alpha: 潜伏期	根据毒株而定
#         :param i: 核酸检测频率(几天一次)，初设1-7天、0天（即不检测，自然发生对照组）
#         :param c: 有症状比例 0.4
#         :param theta_s: 有症状核酸检出率 0.8
#         :param theta_a: 无症状核酸检出率 0.6
#         :param gamma_s1: 中轻度患者痊愈时间 10
#         :param gamma_a1: 中轻度无症状患者痊愈时间 10
#         :param gamma_u: 重症患者接受治疗后的痊愈时间 10
#         :param p: 中轻度患者重症率 15%
#         :param m: 重症患者死亡率 6.4%
#         """
#         self.data = data
#         self.a = a
#         self.time = time
#         self.real_patients = real_patients
#         self.loss = loss
#         self.r_is = r_is
#         self.r_ia = r_ia
#         self.beta_is = beta_is
#         self.beta_ia = beta_ia
#         self.r_beta_is = self.r_is * self.beta_is
#         self.r_beta_ia = self.r_ia * self.beta_ia
#         self.t = t
#         self.alpha = alpha
#         self.i = i
#         self.c = c
#         self.theta_s = theta_s
#         self.theta_a = theta_a
#         self.gamma_s1 = gamma_s1
#         self.gamma_a1 = gamma_a1
#         self.gamma_u = gamma_u
#         self.p = p
#         self.m = m
#
#     def run(self):
#         for indx in range(len(self.time) - 1):
#             # 易感者
#             susceptible = - (self.r_beta_is * self.data["susceptible"][indx] * self.data["infectious_s"][indx]
#                              + self.r_beta_ia * self.data["susceptible"][indx] * self.data["infectious_a"][indx]) / \
#                           self.data["n"] + self.t * self.a[indx] \
#                           + self.data["susceptible"][indx]
#             # 暴露者
#             exposed = (self.r_beta_is * self.data["susceptible"][indx] * self.data["infectious_s"][indx]
#                        + self.r_beta_ia * self.data["susceptible"][indx] * self.data["infectious_a"][indx]) / \
#                       self.data["n"] - self.data["exposed"][indx] / self.alpha + (1.0 - self.t) * self.a[indx] \
#                       + self.data["exposed"][indx]
#             # 感染者 中轻度患者
#             infectious_s = self.c * self.data["exposed"][indx] / self.alpha \
#                            - self.theta_s * self.data["infectious_s"][indx] / self.i \
#                            + self.data["infectious_s"][indx]
#             # 感染者 无症状患者
#             infectious_a = (1.0 - self.c) * self.data["exposed"][indx] / self.alpha \
#                            - self.theta_a * self.data["infectious_a"][indx] / self.i \
#                            + self.data["infectious_s"][indx] \
#                            + self.data["infectious_a"][indx]
#             # 感染者 重症状患者
#             infectious_u = self.p * self.data["quarantine_s"][indx] \
#                            - self.data["infectious_u"][indx] / self.gamma_u - self.m * self.data["infectious_u"][indx] \
#                            + self.data["infectious_u"][indx]
#             # 感染者 中轻度隔离患者
#             quarantine_s = self.theta_s * self.data["infectious_s"][indx] / self.i \
#                            - self.p * self.data["quarantine_s"][indx] - self.data["quarantine_s"][indx] / self.gamma_s1 \
#                            + self.data["quarantine_s"][indx]
#             # 感染者 无症状隔离患者
#             quarantine_a = self.theta_a * self.data["infectious_a"][indx] / self.i \
#                            - self.data["quarantine_a"][indx] / self.gamma_a1 \
#                            + self.data["quarantine_a"][indx]
#             # 康复者
#             recovered = self.data["infectious_u"][indx] / self.gamma_u \
#                         + self.data["quarantine_s"][indx] / self.gamma_s1 \
#                         + self.data["quarantine_a"][indx] / self.gamma_a1 \
#                         + self.data["recovered"][indx]
#             # 死亡者
#             dead = self.m * self.data["infectious_u"][indx] + self.data["dead"][indx]
#
#             self.data["susceptible"].append(float(susceptible))
#             self.data["exposed"].append(float(exposed))
#             self.data["infectious_s"].append(float(infectious_s))
#             self.data["infectious_a"].append(float(infectious_a))
#             self.data["infectious_u"].append(float(infectious_u))
#             self.data["quarantine_s"].append(float(quarantine_s))
#             self.data["quarantine_a"].append(float(quarantine_a))
#             self.data["recovered"].append(float(recovered))
#             self.data["dead"].append(float(dead))
#
#             self.data["predict_total"].append(self.data["infectious_s"][indx + 1]
#                                               + self.data["infectious_a"][indx + 1]
#                                               + self.data["infectious_u"][indx + 1]
#                                               + self.data["quarantine_s"][indx + 1]
#                                               + self.data["quarantine_a"][indx + 1])
#
#             print("----------")
#             print("seir:", infectious_s + infectious_a + infectious_u + quarantine_s + quarantine_a)
#             print("----------")
#
#     def train(self):
#         beta_is = self.beta_is
#         beta_ia = self.beta_ia
#         optimal = minimize(loss, [beta_is, beta_ia],
#                            # beta_is 和 beta_ia 就可以不要了
#                            args=(self.real_patients, self.data, self.a, self.r_is, self.r_ia,
#                                  # self.beta_is, self.beta_ia,
#                                  self.t, self.alpha, self.i, self.c,
#                                  self.theta_s, self.theta_a, self.gamma_s1, self.gamma_a1, self.gamma_u,
#                                  self.p, self.m),
#                            method='L-BFGS-B', bounds=[(0.001, 0.4), (0.001, 0.4)])
#         print("optimal:\n", optimal)
#         self.beta_is, self.beta_ia = optimal.x
#         beta_is = self.beta_is
#         beta_ia = self.beta_ia
#         self.r_beta_is = self.r_is * self.beta_is
#         self.r_beta_ia = self.r_ia * self.beta_ia
#         self.run()
#
#     def getItem(self, item: str):
#         return self.data[item]
#
#     def drawOne(self, real=None, size=(8, 5), dpi=100):
#         sns.set()
#         plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']  # 显示中文-微软雅黑字体
#         plt.figure(figsize=size, dpi=dpi)
#
#         if real is not None:
#             data = pd.DataFrame({
#                 "I": self.data["I"],
#                 "real":
#                     np.array((real["asymptomatic_cumulative"] + real["cumulative_diagnosis"])).reshape(1, -1).tolist()[
#                         0][::-1][:-1]
#             })
#         else:
#             data = pd.DataFrame({
#                 "I": self.data["I"],
#             })
#         sns.lineplot(data=data)
#
#     def drawGraph(self, size=(8, 5), dpi=200):
#         sns.set()
#         plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']  # 显示中文-微软雅黑字体
#         # figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
#         # num: 图像编号或名称，数字为编号 ，字符串为名称
#         # figsize: 指定figure的宽和高，单位为英寸；
#         # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
#         # 1英寸等于2.5cm, A4纸是21 * 30cm的纸张
#         # facecolor: 背景颜色
#         # edgecolor: 边框颜色
#         # frameon: 是否显示边框
#         plt.figure(figsize=size, dpi=dpi)
#         data = pd.DataFrame({
#             "susceptible": self.data["susceptible"], "exposed": self.data["exposed"],
#             "infectious_s": self.data["infectious_s"], "infectious_a": self.data["infectious_a"],
#             "infectious_u": self.data["infectious_u"], "quarantine_s": self.data["quarantine_s"],
#             "quarantine_a": self.data["quarantine_a"], "recovered": self.data["recovered"],
#             "dead": self.data["dead"]
#         })
#         sns.lineplot(data=data)
#
#
# # 参数拟合
# def loss(point, real_patients, data, a, r_is, r_ia, t, alpha, i, c,
#          theta_s, theta_a, gamma_s1, gamma_a1, gamma_u, p, m):
#     size = len(real_patients)
#     beta_is, beta_ia = point
#     # 平均人口流动
#     aa = 0
#     for j in a:
#         aa = aa + j
#     aa = aa / size
#
#     def SEIR(use, y):
#         print(y[2] + y[3] + y[4] + y[5] + y[6])
#         # 易感者
#         susceptible = - (r_is * beta_is * y[0] * y[2] + r_ia * beta_ia * y[0] * y[3]) / data["n"] \
#                       + t * aa
#         # 暴露者
#         exposed = (r_is * beta_is * y[0] * y[2]
#                    + r_ia * beta_ia * y[0] * y[3]) / data["n"] - y[1] / alpha + (1.0 - t) * aa
#         # 感染者 中轻度患者
#         infectious_s = c * y[1] / alpha - theta_s * y[2] / i
#         # 感染者 无症状患者
#         infectious_a = (1.0 - c) * y[1] / alpha - theta_a * y[3] / i + y[2]
#         # 感染者 重症状患者
#         infectious_u = p * y[5] - y[4] / gamma_u - m * y[4]
#         # 感染者 中轻度隔离患者
#         quarantine_s = theta_s * y[2] / i - p * y[5] - y[5] / gamma_s1
#         # 感染者 无症状隔离患者
#         quarantine_a = theta_a * y[3] / i - y[6] / gamma_a1
#         # 康复者
#         recovered = y[4] / gamma_u + y[5] / gamma_s1 + y[6] / gamma_a1
#         # 死亡者
#         dead = m * y[4]
#         return [susceptible, exposed, infectious_s, infectious_a, infectious_u,
#                 quarantine_s, quarantine_a, recovered, dead]
#
#     solution = solve_ivp(SEIR, [0, size],
#                          [data["susceptible"][0], data["exposed"][0],
#                           data["infectious_s"][0], data["infectious_a"][0], data["infectious_u"][0],
#                           data["quarantine_s"][0], data["quarantine_a"][0], data["recovered"][0], data["dead"][0]],
#                          t_eval=np.arange(0, size, 1), vectorized=True)
#     print('------')
#     # # 这里的 real_patients 值是总值
#     # real_patients = [real_patients[i] - real_patients[i - 1]
#     #                  if i != 0 else real_patients[0]
#     #                  for i in range(len(real_patients))]
#     # print(real_patients)
#     loss_val = np.sqrt(
#         np.mean((solution.y[2] + solution.y[3] + solution.y[4] + solution.y[5] + solution.y[6] - real_patients) ** 2))
#     return loss_val
