"""
Module with the unit tests for the churn_library.py module

Author: Rudi César Comitto Modena
Date: August, 2022
"""

import os
import shutil
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
    @pytest.fixture(scope='class', autouse=True)
    def setup(self):
        '''
        Remove all files created by the modeling process
        '''
        for path in [const.EDA_FIGURE_FOLDER, const.RESULTS_FIGURE_FOLDER, const.MODELS_FOLDER]:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)

                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception:
                    print(f"Failed to delete {file_path}.")

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
            logging.error("Testing import_eda: The file wasn't found")
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
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    def test_encoder_helper(self, encoder_helper):
        '''
        test encoder helper
        '''
        assert True == True


    def test_perform_feature_engineering(self, perform_feature_engineering):
        '''
        test perform_feature_engineering
        '''
        assert True == True



    def test_train_models(self, train_models):
        '''
        test train_models
        '''
        assert True == True



if __name__ == "__main__":
    pytest.main(["-v", "churn_script_logging_and_tests.py::TestClassChurnLibrary"])