import xlrd2
import os
import datetime

from model.People import People

readfile = xlrd2.open_workbook(r"北京流入流出2019.xlsx")
# 获取sheet名称
names = readfile.sheet_names()
print(names)
# 选择所需要的sheet
# 获取sheet对象
sheet_name_input = "北京市流入"
sheet_name_output = "北京市流出"
obj_sheet_input = readfile.sheet_by_name(sheet_name_input)
obj_sheet_output = readfile.sheet_by_name(sheet_name_output)
# 获取行列
row_input = obj_sheet_input.nrows
row_output = obj_sheet_output.nrows
# 流入人口
beijing_input = []
# 流出人口
beijing_output = []

dates = []
population_shift = []
patients = []
# 获取上层目录路径
path1 = os.path.dirname(__file__)
path2 = os.path.dirname(path1)
# 获取 北京流入 数据
filename = path2 + '/data/beijing.csv'


def people_have(city_input, date):
    for i in range(len(city_input)):
        if city_input[i].date == date:
            return i
    return -1


# 流入人口
for i in range(1, row_input):
    row = obj_sheet_input.row_values(i)
    date = str(int(obj_sheet_input.cell_value(i, 0)))
    date = datetime.datetime.strptime(date, '%Y%m%d')
    use = people_have(beijing_input, date)
    if use != -1:
        beijing_input[use] = People(date, beijing_input[use].population_shift + int(obj_sheet_input.cell_value(i, 10)))
    else:
        beijing_input.append(People(date, int(obj_sheet_input.cell_value(i, 10))))

# 流出人口
for i in range(1, row_output):
    row = obj_sheet_output.row_values(i)
    date = str(int(obj_sheet_output.cell_value(i, 0)))
    date = datetime.datetime.strptime(date, '%Y%m%d')
    use = people_have(beijing_output, date)
    if use != -1:
        beijing_output[use] = People(date,
                                     beijing_output[use].population_shift + int(obj_sheet_output.cell_value(i, 10)))
    else:
        beijing_output.append(People(date, int(obj_sheet_output.cell_value(i, 10))))

# 排序
beijing_input = sorted(beijing_input, key=lambda x: x.date)
beijing_output = sorted(beijing_output, key=lambda x: x.date)

i, j = 0, 0
while i < len(beijing_input) and j < len(beijing_output):
    if beijing_input[i].date < beijing_output[i].date:
        dates.append(beijing_input[i].date.strftime('%Y/%m/%d'))
        population_shift.append(beijing_input[i].population_shift)
        i = i + 1
    elif beijing_input[i].date > beijing_output[i].date:
        dates.append(beijing_output[i].date.strftime('%Y/%m/%d'))
        population_shift.append(-beijing_output[i].population_shift)
        j = j + 1
    else:
        dates.append(beijing_input[i].date.strftime('%Y/%m/%d'))
        population_shift.append(beijing_input[i].population_shift - beijing_output[i].population_shift)
        i = i + 1
        j = j + 1

while i < len(beijing_input):
    dates.append(beijing_input[i].date.strftime('%Y/%m/%d'))
    population_shift.append(beijing_input[i].population_shift)
    i = i + 1

while j < len(beijing_output):
    dates.append(beijing_output[i].date.strftime('%Y/%m/%d'))
    population_shift.append(-beijing_output[i].population_shift)
    j = j + 1

with open(filename, 'w', encoding='utf-8') as f:
    print(f)
    f.writelines('date,' + 'population shift' + '\n')
    for i in range(len(dates)):
        f.writelines(dates[i] + ',' + str(population_shift[i]) + '\n')
