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


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')


class TestClassChurnLibrary:
    '''
    class with all of the test of the churn_library.py module
    '''
    @classmethod
    @pytest.fixture(autouse=True)
    def setup(cls):
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
                file_path = os.path.join(path, filename)

                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except OSError:
                    print(f"Failed to delete {file_path}.")

    @classmethod
    @pytest.fixture(scope="class")
    def import_data(cls):
        '''
        Pytext fixture for the import_data function
        '''
        return churn_library.import_data

    @classmethod
    @pytest.fixture(scope="class")
    def perform_eda(cls):
        '''
        Pytext fixture for the perform_eda function
        '''
        return churn_library.perform_eda

    @classmethod
    @pytest.fixture(scope="class")
    def train_models(cls):
        '''
        Pytext fixture for the train_models function
        '''
        return churn_library.train_models

    @classmethod
    @pytest.fixture(scope="class")
    def encoder_helper(cls):
        '''
        Pytext fixture for the encoder_helper function
        '''
        return churn_library.encoder_helper

    @classmethod
    @pytest.fixture(scope="class")
    def perform_feature_engineering(cls):
        '''
        Pytext fixture for the perform_feature_engineering function
        '''
        return churn_library.perform_feature_engineering

    @classmethod
    def test_import(cls, import_data):
        '''
        test data import - this example is completed for you to assist with the
        other test functions
        '''
        try:
            pd_df = import_data(const.DATASET_PATH)
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_data: The file wasn't found")
            raise err

        try:
            assert pd_df.shape[0] > 0
            assert pd_df.shape[1] > 0
        except AssertionError as err:
            erro_message = ("Testing import_data: The file doesn't appear to"
                             + " have rows and columns")
            logging.error(erro_message)
            raise err

        pytest.pd_df = pd_df

    @classmethod
    def test_eda(cls, perform_eda):
        '''
        test perform eda function
        '''
        try:
            pd_df = pytest.pd_df
            perform_eda(pd_df)
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

    @classmethod
    def test_encoder_helper(cls, encoder_helper):
        '''
        test encoder helper
        '''
        try:
            pd_df = pytest.pd_df
            pd_df_enc = encoder_helper(
                pd_df=pd_df,
                category_lst=const.CAT_COLUMNS,
                response="test")

            logging.info("Testing test_encoder_helper: SUCCESS")
        except AttributeError as err:
            logging.error("Testing test_encoder_helper: Attribute not found")
            raise err

        try:
            assert pd_df.shape[1] == 23
            assert pd_df_enc.shape[1] == 41
            assert "test" in pd_df_enc.columns

        except AssertionError as err:
            erro_message = ("Testing perform_eda: It appears that there are"
                            + " not created all columns needed.")
            logging.error(erro_message)

            raise err

    @classmethod
    def test_perform_feature_engineering(cls, perform_feature_engineering):
        '''
        test perform_feature_engineering
        '''
        try:
            pd_df = pytest.pd_df
            (X_train,
             X_test,
             y_train,
             y_test) = perform_feature_engineering(pd_df)
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

    @classmethod
    def test_train_models(cls, train_models):
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
        "churn_script_logging_and_tests.py::TestClassChurnLibrary"
    ]

    pytest.main(list_args)
