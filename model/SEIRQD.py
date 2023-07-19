import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import xlwt


class SEIRQD:
    def __init__(self, data: dict, a=None, time=0, real_patients=None,
                 r_is=20.0, r_ia=40.0, beta_is=0.05, beta_ia=0.025,
                 t=1.0, alpha=4.4, c=0.4,
                 theta_s=0.8, theta_a=0.6, gamma_s1=10.0, gamma_a1=7.0, gamma_u=10.0, p=0.15, m=0.064,
                 al=0, q=0.2):
        """
        :param data: SEIR
        :param a: 人口净流动量 原始数据集 用 dict
        :param time: 时间
        :param real_patients: 真实病例数
        :param r_is: 有症状感染者接易感人群的人数
        :param r_ia: 无症状感染者接触易感人群的人数
        :param beta_is: 有症状感染系数 例：0.05 基于SEIR模型的高校新冠肺炎疫情传播风险管控研究
        :param beta_ia: 无症状感染系数 例：0.025 基于SEIR模型的高校新冠肺炎疫情传播风险管控研究
        :param t: 流动人口中的易感者比例	初设0（有隔离政策）、0.00001（无隔离政策）
        :param alpha: 潜伏期	根据毒株而定
        :param c: 有症状比例 0.4
        :param theta_s: 有症状核酸检出率 0.8
        :param theta_a: 无症状核酸检出率 0.6
        :param gamma_s1: 中轻度患者痊愈时间 10
        :param gamma_a1: 无症状患者痊愈时间 7
        :param gamma_u: 重症患者接受治疗后的痊愈时间 10
        :param al: 抗体水平随时间变化的表达式
        :param q: 自我隔离比例
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
        self.alpha = alpha
        self.c = c
        self.t = t
        self.theta_s = theta_s
        self.theta_a = theta_a
        self.gamma_s1 = gamma_s1
        self.gamma_a1 = gamma_a1
        self.gamma_u = gamma_u
        self.p = p
        self.m = m
        self.al = al
        self.q = q

    def get_val(self, time):
        l = [0, 160, 310, 450, 580, 700, 810, 910, 1000, 1080]
        gamma_s1 = self.gamma_s1
        gamma_a1 = self.gamma_a1
        beta_is = self.beta_is
        beta_ia = self.beta_ia
        num = 0
        for i in l:
            if time > i:
                num += 1
        for _ in range(num - 1):
            down = 0.8
            gamma_s1 *= down
            gamma_a1 *= down
            down = 0.5
            beta_is *= down
            beta_ia *= down
        if l[num] - time > 40:
            return 0, gamma_s1, gamma_a1, beta_is, beta_ia
        else:
            return self.al - num * 0.0005, gamma_s1, gamma_a1, beta_is, beta_ia

    def run(self):
        for indx in range(self.time - 1):
            al, gamma_s1, gamma_a1, beta_is, beta_ia = self.get_val(indx)
            self.r_beta_is = beta_is * self.r_is
            self.r_beta_ia = beta_ia * self.r_ia
            # 易感者
            susceptible = max(- (self.r_beta_is * self.data["susceptible"][indx] * self.data["infectious_s"][indx]
                                 + self.r_beta_ia * self.data["susceptible"][indx] * self.data["infectious_a"][indx]) / \
                              self.data["n"] + self.data["susceptible"][indx], 0.0) \
                          + al * self.data["recovered"][indx]
            # 暴露者
            exposed = (self.r_beta_is * self.data["susceptible"][indx] * self.data["infectious_s"][indx]
                       + self.r_beta_ia * self.data["susceptible"][indx] * self.data["infectious_a"][indx]) / \
                      self.data["n"] - self.data["exposed"][indx] / self.alpha + self.data["exposed"][indx]
            exposed = max(exposed, 0.0)
            # 感染者 中轻度患者
            infectious_s = self.c * self.data["exposed"][indx] / self.alpha \
                           - self.data["infectious_s"][indx] / gamma_s1 \
                           - self.data["infectious_s"][indx] * self.q \
                           + self.data["infectious_s"][indx]
            infectious_s = max(infectious_s, 0.0)
            # 感染者 无症状患者
            infectious_a = (1.0 - self.c) * self.data["exposed"][indx] / self.alpha \
                           - self.data["infectious_a"][indx] / gamma_a1 \
                           + self.data["infectious_a"][indx]
            infectious_a = max(infectious_a, 0.0)
            # 自我隔离者
            quarantine = self.q * self.data["infectious_s"][indx] - self.data["quarantine"][indx] / gamma_s1 \
                         + self.data["quarantine"][indx]
            quarantine = max(quarantine, 0.0)
            # 康复者
            recovered = self.data["infectious_s"][indx] / gamma_s1 \
                        + self.data["infectious_a"][indx] / gamma_a1 \
                        + self.data["quarantine"][indx] / gamma_s1 \
                        - al * self.data["recovered"][indx] \
                        + self.data["recovered"][indx]
            recovered = max(recovered, 0.0)

            self.data["susceptible"].append(float(susceptible))
            self.data["exposed"].append(float(exposed))
            self.data["infectious_s"].append(float(infectious_s))
            self.data["infectious_a"].append(float(infectious_a))
            self.data["quarantine"].append(float(quarantine))
            self.data["recovered"].append(float(recovered))

            self.data["real_patients"].append(self.data["infectious_a"][indx + 1]
                                              + self.data["infectious_s"][indx + 1])

            self.data["predict_total"].append(self.data["infectious_a"][indx + 1]
                                              + self.data["infectious_s"][indx + 1]
                                              + self.data["quarantine"][indx + 1]
                                              + self.data["recovered"][indx + 1])

            # self.data["predict_all"].append(self.data["susceptible"][indx + 1]
            #                                 + self.data["exposed"][indx + 1]
            #                                 + self.data["infectious_s"][indx + 1]
            #                                 + self.data["infectious_a"][indx + 1]
            #                                 + self.data["quarantine"][indx + 1]
            #                                 + self.data["recovered"][indx + 1])

    def train(self):
        if self.beta_is is None or self.beta_ia is None:
            loss_val_min = float('inf')
            for beta_is in np.arange(0.001, 0.900, 0.001):
                for beta_ia in np.arange(0.001, 0.900, 0.001):
                    if not 1.3 * beta_ia < beta_is < 2 * beta_ia:
                        continue
                    loss_val = getLoss(copy.deepcopy(self.data), self.a, self.time, self.real_patients,
                                       r_is=self.r_is, r_ia=self.r_ia, beta_is=beta_is, beta_ia=beta_ia, t=self.t,
                                       alpha=self.alpha, c=self.c, theta_s=self.theta_s, theta_a=self.theta_a,
                                       gamma_s1=self.gamma_s1, gamma_a1=self.gamma_a1, gamma_u=self.gamma_u,
                                       p=self.p, m=self.m, q=self.q, al=self.al)

                    if loss_val_min > loss_val:
                        loss_val_min = loss_val
                        self.beta_is = beta_is
                        self.beta_ia = beta_ia
                        print(self.beta_is, self.beta_ia, loss_val)

        # print(self.beta_is, self.beta_ia)
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
        index = pd.date_range(start='20221207', periods=self.time, name="时间")
        data = pd.DataFrame(data={
            "真实患病人数": self.data["real_patients"]
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

        sht1.write(0, 5, 'quarantine', style0)
        for i in range(1, len(self.data["quarantine"]) + 1):
            sht1.write(i, 5, self.data["quarantine"][i - 1])

        sht1.write(0, 6, 'recovered', style0)
        for i in range(1, len(self.data["recovered"]) + 1):
            sht1.write(i, 6, self.data["recovered"][i - 1])

        sht1.write(0, 7, 'predict_total', style0)
        for i in range(1, len(self.data["predict_total"]) + 1):
            sht1.write(i, 7, self.data["predict_total"][i - 1])

        #
        # sht1.write(0, 14, '每日新增患者', style0)
        # sht1.write(1, 14, self.data["predict_total"][0], style0)
        # for i in range(2, len(self.data["predict_total"]) + 1):
        #     sht1.write(i, 14, self.data["predict_total"][i - 1] - self.data["predict_total"][i - 2])
        #
        # sht1.write(0, 15, '每日新增重症', style0)
        # sht1.write(1, 15, self.data["infectious_u"][0], style0)
        # for i in range(2, len(self.data["infectious_u"]) + 1):
        #     sht1.write(i, 15, int(self.data["infectious_u"][i - 1]) - int(self.data["infectious_u"][i - 2]))
        #
        # sht1.write(0, 16, '每日新增死亡', style0)
        # sht1.write(1, 16, self.data["dead"][0], style0)
        # for i in range(2, len(self.data["dead"]) + 1):
        #     sht1.write(i, 16, int(self.data["dead"][i - 1]) - int(self.data["dead"][i - 2]))
        #
        # sht1.write(0, 17, '每日新增确诊病例', style0)
        # sht1.write(1, 17, self.data["predict_all"][0], style0)
        # for i in range(2, len(self.data["dead"]) + 1):
        #     sht1.write(i, 17, int(self.data["predict_all"][i - 1]) - int(self.data["predict_all"][i - 2]))
        #
        # sht1.write(0, 18, '累计确诊病例', style0)
        # for i in range(1, len(self.data["dead"]) + 1):
        #     sht1.write(i, 18, int(self.data["predict_all"][i - 1]))

        # sht1.write(0, 19, 'beta_is 有症状感染系数', style0)
        # sht1.write(1, 19, self.beta_is, style0)
        # sht1.write(0, 20, 'beta_ia 无症状感染系数', style0)
        # sht1.write(1, 20, self.beta_ia, style0)
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
            r_is, r_ia, beta_is, beta_ia, t, alpha, c, theta_s, theta_a, gamma_s1, gamma_a1, gamma_u, p, m, q, al):
    use = SEIRQD(data, a, time, real_patients,
                 r_is=r_is, r_ia=r_ia, beta_is=beta_is, beta_ia=beta_ia,
                 t=t, alpha=alpha, c=c,
                 theta_s=theta_s, theta_a=theta_a,
                 gamma_s1=gamma_s1, gamma_a1=gamma_a1, gamma_u=gamma_u, p=p, m=m, q=q, al=al)
    use.run()
    return use.loss_huber()
