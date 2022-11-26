import requests
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--city_name', type=str, default='上海', help='Access to urban epidemic information by name')
args = parser.parse_args()

city_name = args.city_name
save_name = city_name + '.csv'
print(save_name)

url = 'https://api.inews.qq.com/newsqa/v1/query/pubished/daily/list?province=' + city_name
print(url)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
}
response = requests.get(url, headers=headers)
code_dic = json.loads(response.content)
code_dic = code_dic['getdata']
# 时间、人口（这里忽略不计）、新增
dates = []
population_shift = []
confirm_adds = []
confirms = []

for mes in code_dic:
    date = str(mes['year']) + '/' + mes['date'][0:2] + '/' + mes['date'][3:5]
    population_s = 0
    confirm_add = mes['confirm_add']
    confirm = mes['confirm']

    dates.append(date)
    population_shift.append(population_s)
    confirm_adds.append(confirm_add)
    confirms.append(confirm)

with open(save_name, 'w', encoding='utf-8') as f:
    print(f)
    # os.linesep代表当前操作系统上的换行符
    f.writelines('date,' + 'population shift,' + 'confirm_adds,' + 'confirm' + '\n')
    for i in range(len(dates)):
        f.writelines((dates[i] + ',' + str(population_shift[i]) + ','
                      + str(confirm_adds[i])) + ',' + str(confirms[i]) + '\n')
