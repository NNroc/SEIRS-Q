import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import xlwt


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
        :param r_is: 有症状感染者接易感人群的人数
        :param r_ia: 无症状感染者接触易感人群的人数
        :param beta_is: 有症状感染系数 例：0.05 基于SEIR模型的高校新冠肺炎疫情传播风险管控研究
        :param beta_ia: 无症状感染系数 例：0.025 基于SEIR模型的高校新冠肺炎疫情传播风险管控研究
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

    def train(self, beta_is=None, beta_ia=None):
        if beta_is is None or beta_ia is None:
            loss_val_min = float('inf')
            for beta_is in np.arange(0.001, 0.900, 0.001):
                for beta_ia in np.arange(0.001, 0.900, 0.001):
                    if not 1.3 * beta_ia < beta_is < 2 * beta_ia:
                        continue
                    loss_val = getLoss(copy.deepcopy(self.data), self.a, self.time, self.real_patients,
                                       r_is=self.r_is, r_ia=self.r_ia, beta_is=beta_is, beta_ia=beta_ia,
                                       t=self.t, alpha=self.alpha, i=self.i, c=self.c,
                                       theta_s=self.theta_s, theta_a=self.theta_a,
                                       gamma_s1=self.gamma_s1, gamma_a1=self.gamma_a1, gamma_u=self.gamma_u,
                                       p=self.p,
                                       m=self.m)
                    if loss_val_min > loss_val:
                        loss_val_min = loss_val
                        self.beta_is = beta_is
                        self.beta_ia = beta_ia
                        # print(self.beta_is, self.beta_ia, loss_val)
        else:
            self.beta_is = beta_is
            self.beta_ia = beta_ia

        print(self.beta_is, self.beta_ia)
        self.r_beta_is = self.r_is * self.beta_is
        self.r_beta_ia = self.r_ia * self.beta_ia
        self.run()

    def drawGraph(self, size=(8, 5), dpi=200, path='./data/result_{}.png'):
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
        if self.real_patients is not None:
            data = pd.DataFrame(data={
                "真实患病人数": self.data["real_patients"], "预测患病人数": self.data["predict_total"]
            }, index=index)
        else:
            data = pd.DataFrame(data={
                "预测患病人数": self.data["predict_total"]
            }, index=index)
        plt.title(self.data["city_name"] + '疫情情况预测对比图')
        plt.xlabel('时间')
        plt.ylabel('确诊人数')
        sns.lineplot(data=data)
        # 获取图的坐标信息
        ax = plt.gca()
        # 设置日期的显示格式
        date_format = mpl.dates.DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_format)
        plt.savefig(path.format(self.data["city_name"]))

    def saveResultToExcel(self, path='./data/result_{}.xls'):
        xls = xlwt.Workbook()
        sht1 = xls.add_sheet(self.data["city_name"])

        # 设置字体格式
        font0 = xlwt.Font()
        font0.name = "Times New Roman"
        font0.colour_index = 2
        font0.bold = True  # 加粗
        style0 = xlwt.XFStyle()

        # 输入到 excel
        sht1.write(0, 0, 'n', style0)
        sht1.write(1, 0, self.data["n"])

        sht1.write(0, 1, 'susceptible', style0)
        for i in range(1, len(self.data["susceptible"]) + 1):
            sht1.write(i, 1, self.data["susceptible"][i - 1])

        sht1.write(0, 2, 'exposed', style0)
        for i in range(1, len(self.data["exposed"]) + 1):
            sht1.write(i, 2, self.data["exposed"][i - 1])

        sht1.write(0, 3, 'infectious_s', style0)
        for i in range(1, len(self.data["infectious_s"]) + 1):
            sht1.write(i, 3, self.data["infectious_s"][i - 1])

        sht1.write(0, 4, 'infectious_a', style0)
        for i in range(1, len(self.data["infectious_a"]) + 1):
            sht1.write(i, 4, self.data["infectious_a"][i - 1])

        sht1.write(0, 5, 'infectious_u', style0)
        for i in range(1, len(self.data["infectious_u"]) + 1):
            sht1.write(i, 5, self.data["infectious_u"][i - 1])

        sht1.write(0, 6, 'quarantine_s', style0)
        for i in range(1, len(self.data["quarantine_s"]) + 1):
            sht1.write(i, 6, self.data["quarantine_s"][i - 1])

        sht1.write(0, 7, 'quarantine_a', style0)
        for i in range(1, len(self.data["quarantine_a"]) + 1):
            sht1.write(i, 7, self.data["quarantine_a"][i - 1])

        sht1.write(0, 8, 'recovered', style0)
        for i in range(1, len(self.data["recovered"]) + 1):
            sht1.write(i, 8, self.data["recovered"][i - 1])

        sht1.write(0, 9, 'dead', style0)
        for i in range(1, len(self.data["dead"]) + 1):
            sht1.write(i, 9, self.data["dead"][i - 1])

        sht1.write(0, 10, 'real_patients', style0)
        if self.real_patients is not None:
            for i in range(1, len(self.data["real_patients"]) + 1):
                sht1.write(i, 10, self.data["real_patients"][i - 1])

        sht1.write(0, 11, 'predict_total', style0)
        for i in range(1, len(self.data["predict_total"]) + 1):
            sht1.write(i, 11, self.data["predict_total"][i - 1])

        sht1.write(0, 12, 'beta_is 有症状感染系数', style0)
        sht1.write(1, 12, self.beta_is, style0)
        sht1.write(0, 13, 'beta_ia 无症状感染系数', style0)
        sht1.write(1, 13, self.beta_ia, style0)
        xls.save(path.format(self.data["city_name"]))

    def loss_huber(self):
        # 平方差损失函数
        # loss_val = np.sqrt(np.mean((self.data["predict_total"] - self.real_patients) ** 2))
        # huber损失函数
        true = self.data["predict_total"]
        pred = self.real_patients
        delta = sum(self.real_patients) / len(self.real_patients) * 0.1
        # print("delta:", delta)
        loss = np.where(np.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                        delta * np.abs(true - pred) - 0.5 * (delta ** 2))
        loss_val = np.sum(loss)
        return loss_val


def getLoss(data: dict, a: dict, time: dict, real_patients: dict,
            r_is, r_ia, beta_is, beta_ia, t, alpha, i, c, theta_s, theta_a, gamma_s1, gamma_a1, gamma_u, p, m):
    use = SEIRQD(data, a, time, real_patients,
                 r_is=r_is, r_ia=r_ia, beta_is=beta_is, beta_ia=beta_ia,
                 t=t, alpha=alpha, i=i, c=c,
                 theta_s=theta_s, theta_a=theta_a,
                 gamma_s1=gamma_s1, gamma_a1=gamma_a1, gamma_u=gamma_u, p=p, m=m)
    use.run()
    return use.loss_huber()
