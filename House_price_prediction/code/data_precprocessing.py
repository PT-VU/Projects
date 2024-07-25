import pandas as pd
import numpy as np
import torch,os
from torch.utils.data import Dataset, DataLoader

# raw_data = pandas.read_csv("D:\projects\Projects\House price prediction\data\\train.csv")


# get onehot encoding form data
# def dummy_data(raw_data_loc):
#     # raw_data = pandas.read_csv(raw_data_loc)
#     new_data = pd.get_dummies(raw_data)
#
#     # def get_
#     # for i in range(len(new_data.iloc[0])):
#         # print(i)
#         # print(new_data.iloc[:,i])
#
#     label = new_data.iloc[:,37]
#     data = new_data.drop(columns="SalePrice")
#     data = data*1#.replace({False: 0, True: 1}, inplace=True)
#     # print(data)
#
#
#
#     return data,label
#
# dummy_data("D:\projects\Projects\House price prediction\data\\train.csv")


class Dataset(Dataset):
    # 初始化数据集
    def __init__(self, root, is_train=True):
        # mode = "/train.csv"if is_train else "/test.csv"
        a = pd.read_csv(f"{root}/train.csv").iloc[:,:]
        a = a.fillna(0)
        new_data = pd.get_dummies(a)
        self.dataset = new_data.iloc[:1306,:]*1if is_train else new_data[1306:]*1
        self.dataset.reset_index(drop=True, inplace=True) ##reset index
        # self.dataset.replace(np.nan, "10086")

        # print("check", self.dataset.iloc[:,:30].isnull().sum())
        # print(self.dataset)
        # self.dataset = (self.dataset - self.dataset.min()) / (self.dataset.max() - self.dataset.min())


    # 统计数据集的长度
    def __len__(self):
        return len(self.dataset.iloc[:,0])

    # 每条数据的处理方式
    def __getitem__(self, index):

        # new_data = pd.get_dummies(self.dataset)
        label = self.dataset.loc[index, ["SalePrice"]]
        data = self.dataset.iloc[index,:]
        data = data.drop(columns="SalePrice")




        return torch.tensor(data=data.astype(np.float32).values), torch.tensor(data=label.astype(np.float32).values)