print("Loading functions and raw data")
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from numpy import mean, std, absolute

from sklearn.metrics import mean_absolute_error

import random

# Load data
data_dirpath = 'C:\\Users\\yixin\\Desktop\\Workspace\\Personal GitHub Repos\\House-Prices\\Projects\\House price prediction\\data'

train_csv = pd.read_csv(os.path.join(data_dirpath, "train.csv"))
test_csv = pd.read_csv(os.path.join(data_dirpath, "test.csv"))

# Maps to store original information of the dataframe (e.g.: datatype, column name)

# We need to store these information because once the dataframe is labelled, the column namd and the datatype will be lost

col_name_map = dict()

for index, col_name in enumerate(train_csv.columns[1:]):
    col_name_map.setdefault(index + 1, col_name)



def change_datatype(df, col_names, new_type):
    for col_name in col_names:
        df[col_name] = df[col_name].astype(new_type)

    return df


def label_encoder(df):
    label_encoder = LabelEncoder()

    for col_name in df.columns:
        if df[col_name].dtype == "O":
            df[col_name] = label_encoder.fit_transform(df[col_name])

    return df


def iterative_impute(df, max_iter=100, random_state=0, verbose=True):
    imp = IterativeImputer(max_iter=max_iter, random_state=random_state, verbose=verbose)

    df = imp.fit_transform(df)
    df = pd.DataFrame(df)

    return df


datatype_map = dict()

for index, col_name in enumerate(train_csv.columns):
    datatype_map.setdefault(index + 1,str(train_csv[col_name].dtype))


def display_boxplot(df):
    for col_nr in df.columns[1:]:

        if datatype_map[col_nr] != "object":
            sns.barplot(x=col_nr, data=df)
        plt.title(f"boxplot of {col_name_map[col_nr]}")
        plt.show()


def display_distplot(df):
    for col_nr in df.columns[1:]:

        if datatype_map[col_nr] != "object":
            sns.distplot(df[col_nr])
        plt.title(f"Distplot of {col_name_map[col_nr]}")
        plt.show()


def display_scatter(df):
    for col_nr in df.columns[1:-1]:

        if datatype_map[col_nr] != "object":
            plt.scatter(df[col_nr], df.iloc[:, -1:])

        plt.title(f"Scatterplot of of {col_name_map[col_nr]}")
        plt.xlabel(col_name_map[col_nr])
        plt.ylabel("House price")
        plt.show()


def data_transform(df, mode="min-max"):
    assert any([mode == "min-max", mode == "abs-max", mode == "z-score", mode == "log"]), "Invalid mode name. \
        Please enter 'min-max', 'abs-max', 'z-score', 'log'."

    df_transformed = df.copy()

    for column in df_transformed.columns[1:-1]:

        # min-max scale
        if mode == "min-max":
            df_transformed[column] = (df_transformed[column] - df_transformed[column].min()) / (
                        df_transformed[column].max() - df_transformed[column].min())

        # Absolute max value
        elif mode == "abs-max":
            df_transformed[column] = df_transformed[column] / df_transformed[column].abs().max()

        # z-score
        elif mode == "z-score":
            df_transformed[column] = (df_transformed[column] - df_transformed[column].mean()) / df_transformed[
                column].std()

        # log normalization
        elif mode == "log":
            df_transformed[column] = np.log(df_transformed[column] + 1)

    return df_transformed


def remove_outliers(df, iqr_range=5):
    df_new = df.copy()

    outlier_index = set()

    for col_name in df_new.columns[1:-1]:
        if datatype_map[col_name] != "object":

            q1 = df_new[col_name].quantile(0.25)
            q3 = df_new[col_name].quantile(0.75)

            iqr = q3 - q1

            outliers = df_new[(df_new[col_name] < (q1 - iqr_range * iqr)) | (df_new[col_name] > (q3 + iqr_range * iqr))]

            for row_index in outliers.index:
                outlier_index.add(row_index)

    df_new = df_new.drop(outlier_index).drop(columns=0)

    return df_new


def remove_outliers_capping(df, lower_bond=0.01, upper_bond=0.99):
    df_new = df.copy()

    outlier_index = set()

    for col_name in df_new.columns[1:-1]:

        if datatype_map[col_name] != "object":
            lower_percentile = df_new[col_name].quantile(lower_bond)
            upper_percentile = df_new[col_name].quantile(upper_bond)

            outliers = df_new[(df_new[col_name] < lower_percentile) | (df_new[col_name] > upper_percentile)]

            for row_index in outliers.index:
                outlier_index.add(row_index)

    df_new = df_new.drop(outlier_index).drop(columns=0)

    return df_new


def test_model(df):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1:]

    alphas = np.logspace(-4, 0, 50)

    result_mean = []
    result_std = []
    result_alpha = []

    for alpha in alphas:
        model = Lasso(alpha=alpha)

        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

        print(f'Mean MAE: {mean(scores):.3f} STD: {std(scores):.3f} alpha={alpha}')

        result_mean.append(mean(scores))
        result_std.append(std(scores))
        result_alpha.append(alpha)

    print(" ")

    print(f"Maximum MAE: {max(result_mean)}, alpha={result_alpha[result_mean.index(max(result_mean))]}")
    print(f"Minimum MAE: {min(result_mean)}, alpha={result_alpha[result_mean.index(min(result_mean))]}")


def check_residuals(data, alpha, savefig_path=""):
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1:]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)

    model_residual = Lasso(alpha=alpha)
    model_residual.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = model_residual.predict(X_test)

    Y_pred = Y_pred.reshape(-1, 1)

    # Calculate the residuals
    residuals = Y_test - Y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(Y_pred, residuals, color='blue', edgecolor='k')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for LASSO Regression (alpha={alpha})')

    if savefig_path:
        plt.savefig(savefig_path)

    plt.show()

### Preprocess and transform data

## Original training data

print("Processing training data")
# Change datatype, impute missing values and encode categorical data
train_csv = change_datatype(train_csv, ["MSSubClass", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd","GarageYrBlt"], str)
train_csv = label_encoder(train_csv)
train_csv = iterative_impute(train_csv)

# Transform data
train_minmax_transformed = data_transform(train_csv, "min-max")
train_absmax_transformed = data_transform(train_csv, "abs-max")
train_zscore_transformed = data_transform(train_csv, "z-score")
train_log_transformed = data_transform(train_csv, "log")

train_log_transformed = remove_outliers_capping(train_log_transformed)
train_minmax_transformed = remove_outliers_capping(train_minmax_transformed)
train_absmax_transformed = remove_outliers_capping(train_absmax_transformed)
train_zscore_transformed = remove_outliers_capping(train_zscore_transformed)

# Preprocess test data
print("Processing test data")
test_csv = label_encoder(test_csv)
test_csv = iterative_impute(test_csv)

test_minmax_transformed = data_transform(test_csv, "min-max")
test_absmax_transformed = data_transform(test_csv, "abs-max")
test_zscore_transformed = data_transform(test_csv, "z-score")
test_log_transformed = data_transform(test_csv, "log")

test_log_transformed = remove_outliers_capping(test_log_transformed)
test_minmax_transformed = remove_outliers_capping(test_minmax_transformed)
test_absmax_transformed = remove_outliers_capping(test_absmax_transformed)
test_zscore_transformed = remove_outliers_capping(test_zscore_transformed)

print("Data Loading and preprocessing complete!")