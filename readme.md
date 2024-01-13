# environment
When using Python 3.9 to read Excel, the following error may occur: AttributeError: 'ElementTree' object has no attribute 'getiterator'
We need to modify the getiterator() on lines 266 and 316 of the xlsx.py file in python\Lib\site-packages\xlrd to iter().
(在使用python3.9读取excel时可能会有如下报错：AttributeError: 'ElementTree' object has no attribute 'getiterator'
需要修改python\Lib\site-packages\xlrd下的xlsx.py文件两个地方266行和316行的的getiterator()改成iter()。)

# Data
```
cd getdata
python Get_Data_By_Excel.py
python Get_Data_By_Province.py --city_name=上海
```

# Run
View the forecast results for Xi'an and Shanghai
```
python main.py --city_name=xian
python main.py --city_name=shanghai
```

# Predict
```
python main_predict.py --city_name=beijing --folder=month2
python main_predict.py --city_name=beijing_year --folder=year
```
