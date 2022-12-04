How to run
python Get_Data_By_Excel.py
python Get_Data_By_Province.py --city_name=上海

python main.py --city_name=xian
python main.py --city_name=shanghai

python main_predict.py --city_name=beijing --folder=month2
python main_predict.py --city_name=beijing_year --folder=year

在使用python3.9读取excel时可能会有如下报错：AttributeError: 'ElementTree' object has no attribute 'getiterator'
需要修改python\Lib\site-packages\xlrd下的xlsx.py文件两个地方266行和316行的的getiterator()改成iter()。