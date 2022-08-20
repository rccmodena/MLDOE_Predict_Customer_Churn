"""
Module that serves as a means of providing fixtures for an entire directory.

Author: Rudi César Comitto Modena
Date: August, 2022
"""

import os
import shutil
import pytest
import constants as const

@pytest.fixture(autouse=True)
def setup():
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
