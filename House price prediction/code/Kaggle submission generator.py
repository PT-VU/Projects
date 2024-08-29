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

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from numpy import mean, std, absolute

from sklearn.metrics import mean_absolute_error

import random

def generate_submission(training_data, test_data, outfile_name, alpha=1.0):

    model = Lasso(alpha=alpha)
    
    x_train = training_data.iloc[:,:-1]
    y_train = training_data.iloc[:,-1]
    
    model.fit(x_train,y_train)

    predictions = model.predict(test_data)
    predictions = pd.DataFrame(predictions, columns=["SalePrice"])
    
    submission_df = pd.concat([utils.test_csv_ids, predictions],axis=1)
    
    print(submission_df.head())

    dirpath = "C:/Users/yixin/Desktop/Workspace/Personal GitHub Repos/House-Prices/Projects/House price prediction/data/"
    submission_df.to_csv(os.path.join(dirpath,outfile_name))
    print("Submission complete!")

if __name__ == "__main__":

    train_transformed = utils.data_transform(train_csv, "abs-max")
    train_transformed = utils.remove_outliers_capping(train_transformed, upper_bond=0.95, lower_bond=0.01)
    test_transformed = utils.label_encoder(test_csv)
    test_transformed = utils.data_transform_test(test_transformed, "abs-max")
    generate_submission(train_transformed, test_transformed, "submission.csv", alpha=0.000001)




