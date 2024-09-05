import utils

from utils import test_csv, train_csv

from utils import iterative_impute, remove_outliers_capping

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from numpy import mean, std, absolute

from sklearn.metrics import mean_absolute_error

import random

def generate_submission(model_type, training_data, test_data, outfile_name, alpha=1.0, l1_ratio = 0.5, max_iter = 1000, tol=0.0001):

    assert any([model_type.lower() == "lasso", model_type.lower() == "elastic-net"]), \
        "invalid model type name. Please enter 'lasso' or 'elastic-net'."

    if model_type.lower() == "lasso":
        model = Lasso(alpha=alpha, random_state=50, max_iter=max_iter, tol=tol)

    elif model_type.lower() == "elastic-net":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=50, max_iter=max_iter, tol=tol)

    x_train = training_data.iloc[:,:-1]
    y_train = training_data.iloc[:,-1]
    
    model.fit(x_train,y_train)

    predictions = model.predict(test_data)
    predictions = pd.DataFrame(predictions, columns=["SalePrice"])
    
    submission_df = pd.concat([utils.test_csv_ids, predictions],axis=1)
    
    print(submission_df.head())

    dirpath = "C:/Users/yixin/Desktop/Workspace/Personal GitHub Repos/House-Prices/Projects/House price prediction/data/"
    submission_df.to_csv(os.path.join(dirpath,outfile_name), index=False)
    print("Submission complete!")


if __name__ == "__main__":
    data_mode = "log"
    model_type = "elastic-net" # select between "lasso" and "elastic-net"
    upper_bond = 0.96
    lower_bond = 0.01
    alpha = 0.016681

    l1_ratio = 0.8
    max_iter = 1000
    tol = 0.01

    train_transformed = utils.data_transform(train_csv, data_mode)
    train_transformed = utils.iterative_impute(train_transformed)
    train_transformed = utils.remove_outliers_capping(train_transformed, upper_bond=upper_bond, lower_bond=lower_bond)
    test_transformed = utils.label_encoder(test_csv)
    test_transformed = utils.data_transform_test(test_transformed, data_mode)
    test_transformed = utils.iterative_impute(test_transformed)

    generate_submission(model_type, train_transformed, test_transformed, "submission.csv", alpha=alpha)

    if model_type.lower() == 'lasso':
        print(f"model:{model_type}, data:{data_mode} transformed training dataset, {upper_bond * 100}% upper-bound and {lower_bond * 100}% lower_bound removed, alpha = {alpha}, "
              f"max iter = {max_iter}, tolerance = {tol}")
    elif model_type.lower() == 'elastic-net':
        print(f"model:{model_type}, data:{data_mode} transformed training dataset, {upper_bond * 100}% upper-bound and {lower_bond * 100}% lower_bound removed, alpha = {alpha}, "
              f"l1 ratio = {l1_ratio}, max iter = {max_iter}, tolerance = {tol}")