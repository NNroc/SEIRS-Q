import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns


class SEIRQD:
    def __init__(self, data: dict, a: dict, time: dict, real_patients: dict,
                 r_is=20.0, r_ia=40.0, beta_is=0.05, beta_ia=0.025,
                 t=1.0, alpha=4.4, i=3.0, c=0.4,
                 theta_s=0.8, theta_a=0.6, gamma_s1=10.0, gamma_a1=10.0, gamma_u=10.0, p=0.15, m=0.064):
        """
        :param data: SEIR
        :param a: 人口净流动量 原始数据集
        :param time: 时间
        :param real_patients: 真实病例数
        :param r_is: 无症状感染者接易感人群的人数
        :param r_ia: 感染者接触易感人群的人数
        :param beta_is: 有症状感染系数 0.05 基于SEIR模型的高校新冠肺炎疫情传播风险管控研究
        :param beta_ia: 无症状感染系数 0.025 基于SEIR模型的高校新冠肺炎疫情传播风险管控研究
        :param t: 流动人口中的易感者比例	初设0（有隔离政策）、0.00001（无隔离政策）
        :param alpha: 潜伏期	根据毒株而定
        :param i: 核酸检测频率(几天一次)，初设1-7天、0天（即不检测，自然发生对照组）
        :param c: 有症状比例 0.4
        :param theta_s: 有症状核酸检出率 0.8
        :param theta_a: 无症状核酸检出率 0.6
        :param gamma_s1: 中轻度患者痊愈时间 10
        :param gamma_a1: 中轻度无症状患者痊愈时间 10
        :param gamma_u: 重症患者接受治疗后的痊愈时间 10
        :param p: 中轻度患者重症率 15%
        :param m: 重症患者死亡率 6.4%
        """
        self.data = data
        self.a = a
        self.time = time
        self.real_patients = real_patients
        self.r_is = r_is
        self.r_ia = r_ia
        self.beta_is = beta_is
        self.beta_ia = beta_ia
        self.r_beta_is = self.r_is * self.beta_is
        self.r_beta_ia = self.r_ia * self.beta_ia
        self.t = t
        self.alpha = alpha
        self.i = i
        self.c = c
        self.theta_s = theta_s
        self.theta_a = theta_a
        self.gamma_s1 = gamma_s1
        self.gamma_a1 = gamma_a1
        self.gamma_u = gamma_u
        self.p = p
        self.m = m

    def run(self):
        for indx in range(len(self.time) - 1):
            # 易感者
            susceptible = - (self.r_beta_is * self.data["susceptible"][indx] * self.data["infectious_s"][indx]
                             + self.r_beta_ia * self.data["susceptible"][indx] * self.data["infectious_a"][indx]) / \
                          self.data["n"] + self.t * self.a[indx] \
                          + self.data["susceptible"][indx]
            # 暴露者
            exposed = (self.r_beta_is * self.data["susceptible"][indx] * self.data["infectious_s"][indx]
                       + self.r_beta_ia * self.data["susceptible"][indx] * self.data["infectious_a"][indx]) / \
                      self.data["n"] - self.data["exposed"][indx] / self.alpha + (1.0 - self.t) * self.a[indx] \
                      + self.data["exposed"][indx]
            # 感染者 中轻度患者
            infectious_s = self.c * self.data["exposed"][indx] / self.alpha \
                           - self.theta_s * self.data["infectious_s"][indx] / self.i \
                           + self.data["infectious_s"][indx]
            # 感染者 无症状患者
            infectious_a = (1.0 - self.c) * self.data["exposed"][indx] / self.alpha \
                           - self.theta_a * self.data["infectious_a"][indx] / self.i \
                           + self.data["infectious_s"][indx] \
                           + self.data["infectious_a"][indx]
            # 感染者 重症状患者
            infectious_u = self.p * self.data["quarantine_s"][indx] \
                           - self.data["infectious_u"][indx] / self.gamma_u - self.m * self.data["infectious_u"][indx] \
                           + self.data["infectious_u"][indx]
            # 感染者 中轻度隔离患者
            quarantine_s = self.theta_s * self.data["infectious_s"][indx] / self.i \
                           - self.p * self.data["quarantine_s"][indx] - self.data["quarantine_s"][indx] / self.gamma_s1 \
                           + self.data["quarantine_s"][indx]
            # 感染者 无症状隔离患者
            quarantine_a = self.theta_a * self.data["infectious_a"][indx] / self.i \
                           - self.data["quarantine_a"][indx] / self.gamma_a1 \
                           + self.data["quarantine_a"][indx]
            # 康复者
            recovered = self.data["infectious_u"][indx] / self.gamma_u \
                        + self.data["quarantine_s"][indx] / self.gamma_s1 \
                        + self.data["quarantine_a"][indx] / self.gamma_a1 \
                        + self.data["recovered"][indx]
            # 死亡者
            dead = self.m * self.data["infectious_u"][indx] + self.data["dead"][indx]

            self.data["susceptible"].append(float(susceptible))
            self.data["exposed"].append(float(exposed))
            self.data["infectious_s"].append(float(infectious_s))
            self.data["infectious_a"].append(float(infectious_a))
            self.data["infectious_u"].append(float(infectious_u))
            self.data["quarantine_s"].append(float(quarantine_s))
            self.data["quarantine_a"].append(float(quarantine_a))
            self.data["recovered"].append(float(recovered))
            self.data["dead"].append(float(dead))

            self.data["predict_total"].append(self.data["infectious_u"][indx + 1]
                                              + self.data["quarantine_s"][indx + 1]
                                              + self.data["quarantine_a"][indx + 1])

    def train(self):
        self.beta_is = 0.001
        self.beta_ia = 0.028

        # loss_val_min = float('inf')
        # for beta_is in np.arange(0.001, 0.400, 0.001):
        #     for beta_ia in np.arange(0.001, 0.400, 0.001):
        #         loss_val = getLoss(copy.deepcopy(self.data), self.a, self.time, self.real_patients,
        #                            r_is=self.r_is, r_ia=self.r_ia, beta_is=beta_is, beta_ia=beta_ia,
        #                            t=self.t, alpha=self.alpha, i=self.i, c=self.c,
        #                            theta_s=self.theta_s, theta_a=self.theta_a,
        #                            gamma_s1=self.gamma_s1, gamma_a1=self.gamma_a1, gamma_u=self.gamma_u, p=self.p,
        #                            m=self.m)
        #         if loss_val_min > loss_val:
        #             loss_val_min = loss_val
        #             self.beta_is = beta_is
        #             self.beta_ia = beta_ia
        #             print(self.beta_is, self.beta_ia, loss_val)

        self.r_beta_is = self.r_is * self.beta_is
        self.r_beta_ia = self.r_ia * self.beta_ia
        self.run()

    def drawGraph(self, size=(8, 5), dpi=200):
        sns.set()
        plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']  # 显示中文-微软雅黑字体
        # figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
        # num: 图像编号或名称，数字为编号 ，字符串为名称
        # figsize: 指定figure的宽和高，单位为英寸；
        # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
        # 1英寸等于2.5cm, A4纸是21 * 30cm的纸张
        # facecolor: 背景颜色
        # edgecolor: 边框颜色
        # frameon: 是否显示边框
        plt.figure(figsize=size, dpi=dpi)
        index = pd.date_range(start=self.time[0], periods=len(self.time), name="时间")
        data = pd.DataFrame(data={
            "真实患病人数": self.data["real_patients"], "预测患病人数": self.data["predict_total"]
        }, index=index)
        plt.title('2021年 西安疫情情况预测对比图')
        plt.xlabel('时间')
        plt.ylabel('确诊人数')
        sns.lineplot(data=data)
        # 获取图的坐标信息
        ax = plt.gca()
        # 设置日期的显示格式
        date_format = mpl.dates.DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_format)
        plt.show()

    def loss(self):
        loss_val = 0
        # for num in range(len(self.real_patients)):
        #     print(loss_val, np.sqrt(np.mean((self.data["predict_total"][num] - self.real_patients[num]) ** 2)))
        #     loss_val = loss_val + np.sqrt(np.mean((self.data["predict_total"][num] - self.real_patients[num]) ** 2))

        loss_val = np.sqrt(np.mean((self.data["predict_total"] - self.real_patients) ** 2))
        return loss_val


def getLoss(data: dict, a: dict, time: dict, real_patients: dict,
            r_is=20.0, r_ia=40.0, beta_is=0, beta_ia=0,
            t=1.0, alpha=4.4, i=3.0, c=0.4,
            theta_s=0.8, theta_a=0.6, gamma_s1=10.0, gamma_a1=10.0, gamma_u=10.0, p=0.15, m=0.064):
    use = SEIRQD(data, a, time, real_patients,
                 r_is=r_is, r_ia=r_ia, beta_is=beta_is, beta_ia=beta_ia,
                 t=t, alpha=alpha, i=i, c=c,
                 theta_s=theta_s, theta_a=theta_a,
                 gamma_s1=gamma_s1, gamma_a1=gamma_a1, gamma_u=gamma_u, p=p, m=m)
    use.run()
    return use.loss()
