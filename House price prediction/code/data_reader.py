import pandas
import numpy as np
import matplotlib.pyplot as plt

raw_data = pandas.read_csv("D:\\projects\\House price prediction\data\\train.csv")

# raw_data.replace(np.nan, 0)

# def modify_nan(column):
#     first = column[column.first_valid_index()]
#     change_list =
raw_data = raw_data.replace(np.nan, "-10086")


print(raw_data["Alley"])
print(raw_data["Alley"][0])
print(type(raw_data["Alley"][0]))


for column, data in raw_data.items():
    print(data[0])
    print(type(data[0]))
    # if

    raw_data.plot(kind = "scatter",x = column, y = "SalePrice")
    plt.show()
    plt.close()




# for column, data in raw_data.items():
#     # 列名
#     print(column)
#     # 访问列数据
#     print(data)
#     data.plot.bar()
    # 访问列中的第1个元素
    # print(data[0])
