How to run
python Get_Data_By_Excel.py
python Get_Data_By_Province.py --city_name=上海
python main.py --city_name=xian
python main.py --city_name=上海

使用python3.9读取excel时报错AttributeError: 'ElementTree' object has no attribute 'getiterator'
需要修改python\Lib\site-packages\xlrd下的xlsx.py文件两个地方266行和316行的的getiterator()改成iter()。