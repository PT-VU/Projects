import pandas
import numpy as np
import matplotlib.pyplot as plt

raw_data = pandas.read_csv("D:\projects\Projects\House_price_prediction\data\\train.csv")




raw_data = raw_data.replace(np.nan, "nagtive")
# print("check", raw_data.iloc[:,:].isnull().sum())

#
# print(raw_data["Alley"])
# print(raw_data["Alley"][0])
# print(type(raw_data["Alley"][0]))


# raw_data.plot(kind = "scatter",x = raw_data["PoolQC"], y = raw_data["SalePrice"])
# plt.show()
# plt.close()


for column, data in raw_data.items():
    # print(data[0])
    # print(type(data[0]))
    # if

    if column == "PoolQC":
        raw_data.plot(kind = "scatter",x = column, y = "SalePrice")
        plt.show()
        plt.close()


