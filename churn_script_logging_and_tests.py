"""
Module with the unit tests for the churn_library.py module

Author: Rudi César Comitto Modena
Date: August, 2022
"""

import os
import logging
import pytest
import pandas as pd
import churn_library
import constants as const

# Loggin Setup
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')


@pytest.fixture(autouse=True)
def setup():
    '''
    Remove all files created by the modeling process
    '''
    list_folders = [
        const.EDA_FIGURE_FOLDER,
        const.RESULTS_FIGURE_FOLDER,
        const.MODELS_FOLDER
    ]

    for path in list_folders:
        for filename in os.listdir(path):
            if filename == '.gitkeep':
                continue
            file_path = os.path.join(path, filename)

            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except OSError:
                print(f"Failed to delete {file_path}.")


@pytest.fixture(scope="module")
def import_data():
    '''
    Pytext fixture for the import_data function
    '''
    return churn_library.import_data


@pytest.fixture(scope="module")
def perform_eda():
    '''
    Pytext fixture for the perform_eda function
    '''
    return churn_library.perform_eda


@pytest.fixture(scope="module")
def train_models():
    '''
    Pytext fixture for the train_models function
    '''
    return churn_library.train_models


@pytest.fixture(scope="module")
def encoder_helper():
    '''
    Pytext fixture for the encoder_helper function
    '''
    return churn_library.encoder_helper


@pytest.fixture(scope="module")
def perform_feature_engineering():
    '''
    Pytext fixture for the perform_feature_engineering function
    '''
    return churn_library.perform_feature_engineering

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the
    other test functions
    '''
    try:
        df = import_data(const.DATASET_PATH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        erro_message = ("Testing import_data: The file doesn't appear to"
                            + " have rows and columns")
        logging.error(erro_message)
        raise err

    pytest.df = df


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df = pytest.df
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except AttributeError as err:
        logging.error("Testing perform_eda: Attribute not found")
        raise err

    try:
        assert os.path.exists(const.EDA_CHURN_DISTRIB_FILENAME) is True
        assert os.path.exists(const.EDA_CUST_AGE_DIST_FILE) is True
        assert os.path.exists(const.EDA_TOTAL_TRANSACT_FILE) is True
        assert os.path.exists(const.EDA_HEATMAP_FILENAME) is True
        assert os.path.exists(const.EDA_MARITAL_DIST_FILE) is True

    except AssertionError as err:
        erro_message = ("Testing perform_eda: It appears that there are"
                            + " not created all image files.")
        logging.error(erro_message)
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df = pytest.df.copy()
        df_enc = encoder_helper(
            df=df,
            category_lst=const.CAT_COLUMNS,
            response="test")

        logging.info("Testing test_encoder_helper: SUCCESS")
    except AttributeError as err:
        logging.error("Testing test_encoder_helper: Attribute not found")
        raise err

    try:
        assert df.shape[1] == 23
        assert df_enc.shape[1] == 41
        assert "test" in df_enc.columns

    except AssertionError as err:
        erro_message = ("Testing perform_eda: It appears that there are"
                        + " not created all columns needed.")
        logging.error(erro_message)

        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df = pytest.df
        (X_train,
         X_test,
         y_train,
         y_test) = perform_feature_engineering(df)
        logging.info("Testing test_perform_feature_engineering: SUCCESS")
    except AttributeError as err:
        erro_message = ("Testing test_perform_feature_engineering:"
                        + "  Attribute not found")
        logging.error(erro_message)

        raise err

    try:
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    except AssertionError as err:
        erro_message = ("Testing test_perform_feature_engineering: It"
                            + " appears that the outputs are not of the"
                            + " correct type.")
        logging.error(erro_message)
        raise err

    pytest.X_train = X_train
    pytest.X_test = X_test
    pytest.y_train = y_train
    pytest.y_test = y_test


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        train_models(
            pytest.X_train,
            pytest.X_test,
            pytest.y_train,
            pytest.y_test)

        logging.info("Testing test_train_models: SUCCESS")
    except AttributeError as err:
        logging.error("Testing test_train_models: Attribute not found")
        raise err

    try:
        # Models
        assert os.path.exists(const.RFC_MODEL_FILENAME) is True
        assert os.path.exists(const.LOGISTIC_MODEL_FILENAME) is True

        # Images
        assert os.path.exists(const.LOGISTIC_RESULTS_FILENAME) is True
        assert os.path.exists(const.RFC_RESULTS_FILENAME) is True
        assert os.path.exists(const.RESULTS_ROC_FILENAME) is True
        assert os.path.exists(const.RESULTS_IMPORTANCE_FILENAME) is True

    except AssertionError as err:
        erro_message = ("Testing test_train_models: It appears that there"
                        + " are not created all models or image files.")
        logging.error(erro_message)

        raise err


if __name__ == "__main__":
    list_args = [
        "-v",
        "churn_script_logging_and_tests.py"
    ]

    pytest.main(list_args)
