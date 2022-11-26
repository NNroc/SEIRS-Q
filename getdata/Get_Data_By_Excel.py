import xlrd2
import os
import datetime

readfile = xlrd2.open_workbook(r"天津西安数据.xlsx")
# 获取sheet名称
names = readfile.sheet_names()
print(names)
# 选择所需要的sheet
# 获取sheet对象
sheet_name = "汇总"
obj_sheet = readfile.sheet_by_name(sheet_name)
print(obj_sheet)

dates = []
population_shift = []
patients = []
# # 获取行列
# row = obj_sheet.nrows
# col = obj_sheet.ncols

# 获取上层目录路径
path1 = os.path.dirname(__file__)
path2 = os.path.dirname(path1)

# # 获取 西安 单元格数据
# filename = path2 + '/data/xian.csv'
# for i in range(1, 20):
#     row = obj_sheet.row_values(i)
#     date = xlrd2.xldate_as_tuple(obj_sheet.cell_value(i, 0), readfile.datemode)
#     dates.append(datetime.date(*date[:3]).strftime('%Y/%m/%d'))
#     population_shift.append(int(obj_sheet.cell_value(i, 1) - obj_sheet.cell_value(i, 2)))
#
#     use = (patients[len(patients) - 1] if len(patients) != 0 else 0)
#     patients.append(use + float(obj_sheet.cell_value(i, 5)))

# 获取 上海 数据
filename = path2 + '/data/shanghai.csv'
for i in range(1, 38):
    row = obj_sheet.row_values(i)
    date = xlrd2.xldate_as_tuple(obj_sheet.cell_value(i, 8), readfile.datemode)
    dates.append(datetime.date(*date[:3]).strftime('%Y/%m/%d'))
    population_shift.append(int(obj_sheet.cell_value(i, 9) - obj_sheet.cell_value(i, 10)))

    use = (patients[len(patients) - 1] if len(patients) != 0 else 0)
    patients.append(use + float(obj_sheet.cell_value(i, 12)))

print(dates)
print(population_shift)
print(patients)

with open(filename, 'w', encoding='utf-8') as f:
    print(f)
    f.writelines('date,' + 'population shift,' + 'patients' + '\n')
    for i in range(len(dates)):
        f.writelines((dates[i] + ',' + str(population_shift[i]) + ',' + str(patients[i])) + '\n')
