# 根据所选时间对数据进行处理
import json
import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--city_name', type=str, default='上海', help='Access to urban epidemic information by name')
parser.add_argument('--save_name', type=str, default='shanghai', help='Name of csv file saves in data folder')
args = parser.parse_args()
city_name = args.city_name

# 获取上层目录路径
path1 = os.path.dirname(__file__)
path2 = os.path.dirname(path1)

# 获取 西安 单元格数据
filename = path2 + '/data/' + args.save_name + '.csv'

real = pd.read_csv("getdata/" + city_name + ".csv", encoding='utf-8')
names = ["date", "population shift", "patients"]
population_shift = np.array(real['population shift'])
population_shift = [float(i) for i in population_shift]
time = np.array(real['date'])
real_patients = np.array(real['patients'])

# 2022/04/03
