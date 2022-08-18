"""
Module that serves as a means of providing fixtures for an entire directory.

Author: Rudi César Comitto Modena
Date: August, 2022
"""

import pytest

def df_plugin():
    return None


def pytest_configure():
    pytest.df = df_plugin()