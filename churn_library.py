"""
Module with the functions needed to predict customer churn

Author: Rudi César Comitto Modena
Date: August, 2022
"""

from typing import NoReturn, List
import os

import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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
        df: pandas dataframe
    '''
    df_import = None

    try:
        df_import = pd.read_csv(filepath_or_buffer=pth)

    except FileNotFoundError:
        print(f"Not able to find the file: {pth}")

    return df_import


def perform_eda(df: pd.DataFrame) -> NoReturn:
    '''
    perform eda on df and save figures to images folder

    input:
        df: pandas dataframe

    output:
        None
    '''
    # Create the response variable
    response = const.RESPONSE_VARIABLE
    base_response = const.RESPONSE_BASE_VARIABLE
    churn_value = const.COSTUMER_CHURN_VALUE
    df[response] = df[base_response].apply(
        lambda val: 1 if val == churn_value else 0)

    # Create Histogram plots
    list_histograms = [
        {
            'variable': const.RESPONSE_VARIABLE,
            'filename': const.EDA_CHURN_DISTRIB_FILENAME,
            'stat':
            'count',
            'kde': False
        },
        {
            'variable': const.EDA_CUSTOMER_AGE_VAR,
            'filename': const.EDA_CUST_AGE_DIST_FILE,
            'stat': 'count',
            'kde': False
        },
        {
            'variable': const.EDA_TOTAL_TRANSACT_VAR,
            'filename': const.EDA_TOTAL_TRANSACT_FILE,
            'stat': 'density',
            'kde': True
        },
    ]

    for histogram in list_histograms:
        plt.figure(
            figsize=(
                const.EDA_FIGURE_WIDTH,
                const.EDA_FIGURE_HEIGHT))

        sns.histplot(
            data=df,
            x=histogram['variable'],
            stat=histogram['stat'],
            kde=histogram['kde'])

        plt.savefig(
            histogram['filename'],
            bbox_inches='tight')
        plt.close()

    # Plot EDA Total Transaction
    plt.figure(
        figsize=(
            const.EDA_FIGURE_WIDTH,
            const.EDA_FIGURE_HEIGHT))
    sns.heatmap(
        df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)

    plt.savefig(
        const.EDA_HEATMAP_FILENAME,
        bbox_inches='tight')

    plt.close()

    # Plot EDA Marital Status
    plt.figure(
        figsize=(
            const.EDA_FIGURE_WIDTH,
            const.EDA_FIGURE_HEIGHT))
    df[const.EDA_MARITAL_STATUS_VAR].value_counts('normalize').plot(
        kind='bar').get_figure()
    plt.savefig(const.EDA_MARITAL_DIST_FILE, bbox_inches='tight')
    plt.close()


def encoder_helper(
    df: pd.DataFrame,
    category_lst: List[str],
    response: str = "") -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    one-hot encoding transformation

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used
        for naming variables or index y column]

    output:
        df: pandas dataframe with new columns for
    '''
    df_enc = df.copy()
    # Rename response variable
    if response:
        df_enc.rename(
            columns={const.RESPONSE_VARIABLE: response},
            inplace=True)

    enc = make_column_transformer((
        OneHotEncoder(handle_unknown='ignore'), category_lst),
        remainder='passthrough')

    np_enc = enc.fit_transform(X=df_enc)

    return pd.DataFrame(
        data=np_enc,
        columns=enc.get_feature_names())


def perform_feature_engineering(
    df: pd.DataFrame,
    response: str = "") -> List[np.ndarray]:
    '''
    perform one-hot encoding and split the dataset in train and test

    input:
        df: pandas dataframe
        response: string of response name [optional argument that could be used
        for naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''

    # One-hot encoding categorical variables
    df = encoder_helper(
        df=df,
        category_lst=const.CAT_COLUMNS,
        response=response
    )

    # Remove columns
    df.drop(
        columns=const.REMOVE_COLUMNS,
        inplace=True
    )

    if response:
        y_data = df.pop(item=response)
    else:
        y_data = df.pop(item=const.RESPONSE_VARIABLE)

    y_data = y_data.astype('int')

    # # Train test split
    return train_test_split(
        df,
        y_data,
        test_size=const.TEST_SIZE,
        random_state=const.RANDOM_STATE)


def classification_report_image(y_train: np.ndarray,
                                y_test: np.ndarray,
                                y_train_preds_lr: np.ndarray,
                                y_train_preds_rf: np.ndarray,
                                y_test_preds_lr: np.ndarray,
                                y_test_preds_rf: np.ndarray) -> NoReturn:
    '''
    produces classification report for training and testing results and stores
    report as image in images folder

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
    list_reports = [
        {
            'title': 'Random Forest',
            'filename': const.LOGISTIC_RESULTS_FILENAME,
            'y_train_preds': y_train_preds_rf,
            'y_test_preds': y_test_preds_rf
        },
        {
            'title': 'Logistic Regression',
            'filename': const.RFC_RESULTS_FILENAME,
            'y_train_preds': y_train_preds_lr,
            'y_test_preds': y_test_preds_lr
        },
    ]

    for report in list_reports:
        model_title = report['title']
        plt.figure(
            figsize=(
                const.RESULTS_REPORTS_WIDTH,
                const.RESULTS_REPORTS_HEIGHT))
        plt.text(
            0.01,
            1.25,
            f'{model_title} Train',
            const.RESULTS_FONT_SETUP,)
        plt.text(
            0.01,
            0.7,
            str(classification_report(
                y_train,
                report['y_train_preds'])),
            const.RESULTS_FONT_SETUP)

        plt.text(
            0.01,
            0.6,
            f'{model_title} Test',
            const.RESULTS_FONT_SETUP)

        plt.text(
            0.01,
            0.05,
            str(classification_report(
                y_test,
                report['y_test_preds'])),
            const.RESULTS_FONT_SETUP)

        plt.axis('off')
        plt.savefig(
            report['filename'],
            bbox_inches='tight')
        plt.close()


def roc_plot(
    lr_model: LogisticRegression,
    rfc_model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray) -> NoReturn:
    '''
    creates and stores the feature importances in pth
    input:
            lr_model: Logistic Regression model object
            rfc_model: Random Forest Classifier model object
            X_test: X testing data
            y_test: test response values

    output:
             None
    '''
    plt.figure(
        figsize=(
            const.RESULTS_ROC_WIDTH,
            const.RESULTS_ROC_HEIGHT))
    axis_0 = plt.gca()

    _ = plot_roc_curve(
        lr_model,
        X_test,
        y_test,
        ax=axis_0,
        alpha=0.8)

    _ = plot_roc_curve(
        rfc_model,
        X_test,
        y_test,
        ax=axis_0,
        alpha=0.8)

    plt.savefig(
        const.RESULTS_ROC_FILENAME,
        bbox_inches='tight')
    plt.close()


def feature_importance_plot(
    model: RandomForestClassifier,
    X_data: pd.DataFrame,
    output_pth:str) -> NoReturn:
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(
        figsize=(
            const.RESULTS_IMPORTANCE_WIDTH,
            const.RESULTS_IMPORTANCE_HEIGHT))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth, bbox_inches='tight')
    plt.close()


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series) -> NoReturn:
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
    # grid search
    rfc = RandomForestClassifier(random_state=const.RANDOM_STATE)

    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(
        solver=const.LRC_SOLVER,
        max_iter=const.LRC_MAX_ITER)

    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=const.PARAM_GRID,
        cv=const.CROSS_VALID)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # save best model
    joblib.dump(
        cv_rfc.best_estimator_,
        const.RFC_MODEL_FILENAME)

    joblib.dump(
        lrc,
        const.LOGISTIC_MODEL_FILENAME)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train=y_train,
                                y_test=y_test,
                                y_train_preds_lr=y_train_preds_lr,
                                y_train_preds_rf=y_train_preds_rf,
                                y_test_preds_lr=y_test_preds_lr,
                                y_test_preds_rf=y_test_preds_rf)

    roc_plot(
        lr_model=lrc,
        rfc_model=cv_rfc,
        X_test=X_test,
        y_test=y_test)

    feature_importance_plot(
        model=cv_rfc,
        X_data=X_train,
        output_pth=const.RESULTS_IMPORTANCE_FILENAME)


if __name__ == "__main__":
    # Import the Bank Dataset
    df_churn = import_data(const.DATASET_PATH)

    # Performe EDA and save the .png files of the results
    perform_eda(df_churn)

    # Feature engineering
    (X_train_df,
     X_test_df,
     y_train_series,
     y_test_series) = perform_feature_engineering(df_churn)

    # Train and evaluate the performance of Logistic Regression and Random
    # Forest Classifier
    train_models(
        X_train_df,
        X_test_df,
        y_train_series,
        y_test_series)
