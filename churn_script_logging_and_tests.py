"""
Module with the unit tests for the churn_library.py module

Author: Rudi César Comitto Modena
Date: August, 2022
"""

import os
import sys
import pytest
import logging
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
            df = import_data(const.DATASET_PATH)
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_eda: The file wasn't found")
            raise err

        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err


    def test_eda(self, perform_eda):
        '''
        test perform eda function
        '''
        # try:
        #     df = import_data(const.DATASET_PATH)
        #     logging.info("Testing import_data: SUCCESS")
        # except FileNotFoundError as err:
        #     logging.error("Testing import_eda: The file wasn't found")
        #     raise err

        # try:
        #     assert df.shape[0] > 0
        #     assert df.shape[1] > 0
        # except AssertionError as err:
        #     logging.error(
        #         "Testing import_data: The file doesn't appear to have rows and columns")
        #     raise err
        assert True == True

    @pytest.fixture(scope="class")
    def encoder_helper(self):
        '''
        Pytext fixture for the encoder_helper function
        '''
        return churn_library.encoder_helper


    def test_encoder_helper(self, encoder_helper):
        '''
        test encoder helper
        '''
        assert True == True


    def test_perform_feature_engineering(self, perform_feature_engineering):
        '''
        test perform_feature_engineering
        '''
        assert True == False



    def test_train_models(self, train_models):
        '''
        test train_models
        '''
        assert True == True



if __name__ == "__main__":
    pytest.main(["-v", "churn_script_logging_and_tests.py::TestClassChurnLibrary"])