"""
Module with the functions needed to predict customer churn

Author: Rudi César Comitto Modena
Date: August, 2022
"""

from typing import NoReturn, Optional, List
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import constants as const



os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        pd_df: pandas dataframe
    '''
    pd_df = None

    try:
        pd_df = pd.read_csv(pth)
    except FileNotFoundError:
        print(f"Not able to find the file: {pth}")

    return pd_df


def create_response_variable(pd_df: pd.DataFrame,
                             base_variable: Optional[str] = const.RESPONSE_BASE_VARIABLE,
                             custumer_churn_value: Optional[str] = const.COSTUMER_CHURN_VALUE) -> pd.DataFrame:
    '''
    create the response variable according to a criterion
    input:
            pd_df: pandas dataframe
            base_variable:  string of base variable
            custumer_churn_value:  string with the value of the costumer that will churn

    output:
            pd_df: pandas dataframe with the new column
    '''
    
    pd_df[const.RESPONSE_VARIABLE] = pd_df[base_variable].apply(
        lambda val: 1 if val == custumer_churn_value else 0)

    return pd_df


def plot_histogram(
        pd_df: pd.DataFrame,
        variable: str,
        filename: str,
        width: Optional[int]=const.EDA_FIGURE_WIDTH,
        height: Optional[int]=const.EDA_FIGURE_HEIGHT,
        stat: Optional[str]='count',
        kde: Optional[bool] = False) -> NoReturn:
    '''
    create a histogram and save it as a png file
    input:
            pd_df: pandas dataframe
            variable:  string of variable
            filename:  string of filename
            width: int of figure width
            height: int of figure height
            stat: string of stat of the histogram
            kde: bool of draw kde curve

    output:
            None
    '''
    plt.figure(figsize=(width, height))
    sns.histplot(data=pd_df, x=variable, stat=stat, kde=kde)
    plt.savefig(filename, bbox_inches='tight')


def plot_heatmap(
    pd_df: pd.DataFrame,
    filename: str,
    widht:Optional[int] = const.EDA_FIGURE_WIDTH,
    height:Optional[int] = const.EDA_FIGURE_HEIGHT) -> NoReturn:
    '''
    create a histogram and save it as a png file
    input:
            pd_df: pandas dataframe
            filename:  string of filename
            width: int of figure width
            height: int of figure height

    output:
            None
    '''
    plt.figure(figsize=(widht, height))
    sns.heatmap(pd_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(filename, bbox_inches='tight')


def plot_categories(
        pd_df: pd.DataFrame,
        variable: str,
        filename: str,
        widht: Optional[int] = const.EDA_FIGURE_WIDTH,
        height: Optional[int] = const.EDA_FIGURE_HEIGHT) -> NoReturn:
    '''
    create a histogram and save it as a png file
    input:
            pd_df: pandas dataframe
            variable:  string of variable
            filename:  string of filename
            width: int of figure width
            height: int of figure height

    output:
            None
    '''
    plt.figure(figsize=(widht, height))
    pd_df[variable].value_counts('normalize').plot(
        kind='bar').get_figure().savefig(
        filename, bbox_inches='tight')


def perform_eda(pd_df: pd.DataFrame) -> NoReturn:
    '''
    perform eda on df and save figures to images folder
    input:
            pd_df: pandas dataframe

    output:
            None
    '''

    # Create the response variable
    pd_df = create_response_variable(pd_df=pd_df)

    # Create the plots
    plot_histogram(
        pd_df=pd_df,
        variable=const.RESPONSE_VARIABLE,
        filename=const.EDA_CHURN_DISTRIB_FILENAME)
    plot_histogram(
        pd_df=pd_df,
        variable=const.EDA_CUSTOMER_AGE_VARIABLE,
        filename=const.EDA_CUSTOMER_AGE_DISTRIB_FILENAME)
    plot_histogram(
        pd_df=pd_df,
        variable=const.EDA_TOTAL_TRANSACTION_VARIABLE,
        filename=const.EDA_TOTAL_TRANSACTION_FILENAME,
        stat='density',
        kde=True)
    plot_heatmap(pd_df=pd_df, filename=const.EDA_HEATMAP_FILENAME)
    plot_categories(
        pd_df=pd_df,
        variable=const.EDA_MARITAL_STATUS_VARIABLE,
        filename=const.EDA_MARITAL_STATUS_DISTR_FILENAME)


def encoder_helper(pd_df: pd.DataFrame, category_lst: List[str], response: str = "") -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
one-hot encoding transformation

    input:
            pd_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
            for naming variables or index y column]

    output:
            pd_df: pandas dataframe with new columns for
    '''
    # Rename response variable
    if response:
        pd_df.rename(columns={const.RESPONSE_VARIABLE: response}, inplace=True)

    enc = make_column_transformer((
        OneHotEncoder(handle_unknown='ignore'), category_lst),
        remainder='passthrough')

    np_enc = enc.fit_transform(pd_df)
    return pd.DataFrame(np_enc, columns=enc.get_feature_names())


def perform_feature_engineering(pd_df: pd.DataFrame, response: str = "") -> List[np.ndarray]:
    '''
    input:
                pd_df: pandas dataframe
                response: string of response name [optional argument that could be used
                for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # One-hot encoding categorical variables
    pd_df = encoder_helper(pd_df, const.CAT_COLUMNS, response)

    # Remove columns
    pd_df.drop(columns=const.REMOVE_COLUMNS, inplace=True)

    if response:
        y = pd_df.pop(response)
    else:
        y = pd_df.pop(const.RESPONSE_VARIABLE)

    X = pd_df

    # # Train test split
    return train_test_split(X, y, test_size=const.TEST_SIZE, random_state=const.RANDOM_STATE)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf) -> NoReturn:
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth) -> NoReturn:
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> NoReturn:
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    print(X_train)
    print(y_train)
    # grid search
    rfc = RandomForestClassifier(random_state=const.RANDOM_STATE)

    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver=const.LRC_SOLVER, max_iter=const.LRC_MAX_ITER)

    # cv_rfc = GridSearchCV(estimator=rfc, param_grid=const.PARAM_GRID, cv=const.CROSS_VALID)
    # cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, const.RFC_MODEL_FILENAME)
    joblib.dump(lrc, const.LOGISTIC_MODEL_FILENAME)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)


if __name__ == "__main__":
    # Execute all the steps of the process
    pd_df = import_data(const.DATASET_PATH)
    perform_eda(pd_df)
    train_models(*perform_feature_engineering(pd_df))