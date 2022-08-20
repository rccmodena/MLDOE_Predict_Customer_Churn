"""
Module with the unit tests for the churn_library.py module

Author: Rudi César Comitto Modena
Date: August, 2022
"""

import os
import pytest
import logging
import churn_library
import constants as const
import pandas as pd
import numpy as np

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



    @pytest.fixture(scope="class")
    def import_data(self):
        '''
        Pytext fixture for the import_data function
        '''
        return churn_library.import_data


    @pytest.fixture(scope="class")
    def perform_eda(self):
        '''
        Pytext fixture for the perform_eda function
        '''
        return churn_library.perform_eda


    @pytest.fixture(scope="class")
    def train_models(self):
        '''
        Pytext fixture for the train_models function
        '''
        return churn_library.train_models


    @pytest.fixture(scope="class")
    def encoder_helper(self):
        '''
        Pytext fixture for the encoder_helper function
        '''
        return churn_library.encoder_helper


    @pytest.fixture(scope="class")
    def perform_feature_engineering(self):
        '''
        Pytext fixture for the perform_feature_engineering function
        '''
        return churn_library.perform_feature_engineering


    def test_import(self, import_data):
        '''
        test data import - this example is completed for you to assist with the other test functions
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
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err

        pytest.pd_df = pd_df


    def test_eda(self, perform_eda):
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
            assert os.path.exists(const.EDA_CHURN_DISTRIB_FILENAME) == True
            assert os.path.exists(const.EDA_CUSTOMER_AGE_DISTRIB_FILENAME) == True
            assert os.path.exists(const.EDA_TOTAL_TRANSACTION_FILENAME) == True
            assert os.path.exists(const.EDA_HEATMAP_FILENAME) == True
            assert os.path.exists(const.EDA_MARITAL_STATUS_DISTR_FILENAME) == True
            
        except AssertionError as err:
            logging.error(
                "Testing perform_eda: It appears that there are not created all image files.")
            raise err

    def test_encoder_helper(self, encoder_helper):
        '''
        test encoder helper
        '''
        try:
            pd_df = pytest.pd_df            
            pd_df_enc = encoder_helper(pd_df=pd_df, category_lst=const.CAT_COLUMNS, response="test")
            logging.info("Testing test_encoder_helper: SUCCESS")
        except AttributeError as err:
            logging.error("Testing test_encoder_helper: Attribute not found")
            raise err

        try:
            assert pd_df.shape[1] == 23
            assert pd_df_enc.shape[1] == 41
            assert "test" in pd_df_enc.columns

            
        except AssertionError as err:
            logging.error(
                "Testing perform_eda: It appears that there are not created all columns needed.")
            raise err


    def test_perform_feature_engineering(self, perform_feature_engineering):
        '''
        test perform_feature_engineering
        '''
        try:
            pd_df = pytest.pd_df
            X_train, X_test, y_train, y_test = perform_feature_engineering(pd_df)
            logging.info("Testing test_perform_feature_engineering: SUCCESS")
        except AttributeError as err:
            logging.error("Testing test_perform_feature_engineering: Attribute not found")
            raise err

        try:
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(X_test, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
            assert isinstance(y_test, pd.Series)
            
        except AssertionError as err:
            logging.error(
                "Testing test_perform_feature_engineering: It appears that the outputs are not of the correct type.")
            raise err

        pytest.X_train = X_train
        pytest.X_test = X_test
        pytest.y_train = y_train
        pytest.y_test = y_test


    def test_train_models(self, train_models):
        '''
        test train_models
        '''
        try:
            train_models(pytest.X_train, pytest.X_test, pytest.y_train, pytest.y_test)
            logging.info("Testing test_train_models: SUCCESS")
        except AttributeError as err:
            logging.error("Testing test_train_models: Attribute not found")
            raise err

        try:
            # Models
            assert os.path.exists(const.RFC_MODEL_FILENAME) == True
            assert os.path.exists(const.LOGISTIC_MODEL_FILENAME) == True

            # Images
            assert os.path.exists(const.LOGISTIC_RESULTS_FILENAME) == True
            assert os.path.exists(const.RFC_RESULTS_FILENAME) == True
            assert os.path.exists(const.RESULTS_ROC_FILENAME) == True
            assert os.path.exists(const.RESULTS_IMPORTANCE_FILENAME) == True
            
        except AssertionError as err:
            logging.error(
                "Testing test_train_models: It appears that there are not created all models or image files.")
            raise err



if __name__ == "__main__":
    pytest.main(["-v", "churn_script_logging_and_tests.py::TestClassChurnLibrary"])